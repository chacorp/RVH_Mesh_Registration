"""
Script for lifting 2D OpenPose predictions from multiple views to 3D.

Author: Ilya Petrov
"""
import sys
import json

import argparse
import pickle as pkl
from pathlib import Path

from tqdm import tqdm
import trimesh
import numpy as np
import torch
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras

sys.path.append(".")
from lib.body_objectives import batch_reprojection_loss_vcam
from lib.smpl.wrapper_pytorch import SMPLPyTorchWrapperBatch
from lib.smpl.priors.th_smpl_prior import get_prior
from utils.configs import load_config

from utils.keypoints_3d_estimation.io import normalize_v_np

def initialize_keypoints_3d(centers, num, smpl_models_path, device="cpu"):
    # load 25 keypoints from SMPL
    prior = get_prior(smpl_models_path, gender='male')

    batch_sz = centers.shape[0]
    pose_init = torch.zeros((batch_sz, 72))
    pose_init[:, 3:3+prior.mean.shape[-1]] = prior.mean
    # betas, pose, trans = torch.zeros((batch_sz, 300)), pose_init, centers  # init SMPL with the translation
    betas, pose, trans = torch.zeros((batch_sz, 10)), pose_init, centers  # init SMPL with the translation
    
    #import pdb;pdb.set_trace()
    smpl = SMPLPyTorchWrapperBatch(smpl_models_path, batch_sz, betas, pose, trans).to(device)
    J, face, hands = smpl.get_landmarks()

    if num == 25:
        return J.to(device)
    else:
        joints_all = torch.cat([J, face, hands], axis=1)
        return joints_all


def load_keypoints_2d(keypoints_2d_file, device="cpu", tol=0.3):
    def prepare_keypoints(data_2d, keypoints_num, keypoints_key, device):
        # returns keypoints reshaped as follows: 1 x N x (3*n_views)
        keypoints = []
        for key in sorted(data_2d.keys()):
            keypoints_view = np.array(data_2d[key][keypoints_key], dtype=np.float32)
            if keypoints_view.ndim == 1 or keypoints_view.shape[0] != keypoints_num:
                keypoints_view = np.zeros((keypoints_num, 3), dtype=np.float32)
            mask = keypoints_view[:, 2] < tol
            keypoints_view[mask, 2] = 0. # low confidence keypoints not used
            keypoints.append(keypoints_view)
        # keypoints = [np.array(data_2d[key][keypoints_key][:1], dtype=np.float32) for key in sorted(data_2d.keys())]
        keypoints = np.squeeze(np.array(keypoints))
        keypoints = keypoints[np.newaxis, ...]
        keypoints = np.swapaxes(keypoints, 1, 2)
        keypoints = np.reshape(keypoints, (1, keypoints_num, -1))

        keypoints = torch.from_numpy(keypoints).to(device)

        return keypoints

    if not (keypoints_2d_file.is_file()):
        print(f"No joints for {keypoints_2d_file}")
        return None

    with keypoints_2d_file.open('r') as fp:
        keypoints_2d = json.load(fp)

    # hardcoded sizes of keypoint arrays: body 25, hand 21, face 70
    body_2d = prepare_keypoints(keypoints_2d, 25, "pose_keypoints_2d", device)
    if body_2d is None:
        print(f"Bad joints for {keypoints_2d_file}")

    face_2d = prepare_keypoints(keypoints_2d, 70, "face_keypoints_2d", device)

    hand_l_2d = prepare_keypoints(keypoints_2d, 21, "hand_left_keypoints_2d", device)

    hand_r_2d = prepare_keypoints(keypoints_2d, 21, "hand_right_keypoints_2d", device)

    return body_2d, face_2d, hand_l_2d, hand_r_2d

def compute_j3d_confidence(j2d):
    """
    compute the confidence of lifted 3d joints
    Args:
        j2d: (1, num_joints, 3*cam_views), numpy array

    Returns: (1, num_joints), mean of 2d joints confidence

    """
    confidences = []
    for joints in j2d:
        ind = np.arange(2, joints.shape[-1], 3)
        conf = joints[:, ind] # (N, cam_views)
        mean_conf = np.sum(conf, -1) / np.sum(conf>0, -1)
        mean_conf[np.isnan(mean_conf)] = 0.
        confidences.append(mean_conf)
    return np.array(confidences)


def main(args):
    # Setup the environment
    device = torch.device("cpu") if args.cpu else torch.device("cuda:0")
    np.seterr(all="ignore")

    # Load 2d pose
    body_2d, face_2d, hand_l_2d, hand_r_2d = load_keypoints_2d(args.keypoints_2d_path, device)
    if body_2d is None:
        raise RuntimeError("Incorrect 2D keypoints.")
    keypoints_2d = torch.cat((body_2d, face_2d, hand_l_2d, hand_r_2d), 1) # lift all joints
    # keypoints_2d = body_2d
    keypoints_num = keypoints_2d.shape[1]

    # Load scan to pose SMPL model
    data = trimesh.load(args.input_path, process=False)
    
    #center = np.array(data.vertices.mean(0))[np.newaxis]
    #centers = torch.tensor(center, device=device, dtype=torch.float)
    
    ### normalize and get center
    #import pdb; pdb.set_trace()
    data.vertices = normalize_v_np(data.vertices)
    
    x_cen = (data.vertices[:,0].min() + data.vertices[:,0].max()) * 0.5
    y_cen = (data.vertices[:,1].min() + data.vertices[:,1].max()) * 0.5
    z_cen = (data.vertices[:,2].min() + data.vertices[:,2].max()) * 0.5
    
    centers = torch.tensor([[x_cen, y_cen, z_cen]], device=device, dtype=torch.float)

    # Initialize 3d pose from SMPL
    keypoints_3d = initialize_keypoints_3d(centers, keypoints_num, args.smpl_models_path, device)
    keypoints_3d = keypoints_3d.clone().detach().requires_grad_(True).to(device)

    # Load cameras
    with (args.camera_path).open("rb") as fp:
        rendering_params = pkl.load(fp)
    image_size = rendering_params["image-size"]
    image_size = torch.Tensor([[image_size, image_size]]).to(device)
    cameras = []
    for R, T in zip(rendering_params["R"], rendering_params["T"]):
        cameras.append(FoVPerspectiveCameras(R=[R], T=[T], device=device))

    # Setup optimization
    optimizer = torch.optim.Adam([keypoints_3d], 0.005, betas=(0.9, 0.999))
    iterations, steps_per_iter = 100, 30
    # iterations, steps_per_iter = 1, 1

    import torchvision
    loop = tqdm(range(iterations))
    for it in loop:
        for i in range(steps_per_iter):
            optimizer.zero_grad()
            # Get losses for a forward pass
            #loss, _ = batch_reprojection_loss_vcam(keypoints_2d[:, :keypoints_num], keypoints_3d, cameras, image_size)
            loss, proj = batch_reprojection_loss_vcam(keypoints_2d[:, :keypoints_num], keypoints_3d, cameras, image_size)
#             import pdb;pdb.set_trace()
            
#             image_empty = torch.zeros((3,512,512),dtype=torch.uint8)
#             torchvision.transforms.ToPILImage()(draw_keypoints(image_empty, proj[i], colors="blue", radius=3)).save('test_{:03d}.png'.format(i))
            
#             torchvision.transforms.ToPILImage()(draw_keypoints(image_empty, keypoints_2d[:,:,i*3:i*3+2], colors="blue", radius=3)).save('test_img_{:03d}.png'.format(i))
            
            
#             smpl_joints_proj_xy = cameras[0].transform_points_screen(keypoints_3d, image_size=image_size)[:, :, :2]
#             torchvision.transforms.ToPILImage()(draw_keypoints(image_empty, smpl_joints_proj_xy[:,:25], colors="blue", radius=3)).save('test_smpl_{:03d}.png'.format(i))
            
            loss.backward()
            optimizer.step()
        
        
        l_str = 'loss: {:0.4f}'.format(loss.mean().item())
        loop.set_description(l_str)
#     import pdb;pdb.set_trace()
#     image_empty = torch.zeros((3,512,512),dtype=torch.uint8)
#     for i in range(10):        
#         torchvision.transforms.ToPILImage()(draw_keypoints(image_empty, proj[i], colors="blue", radius=3)).save('test_{:03d}.png'.format(i))
    
    # Save results
    res = keypoints_3d.cpu().detach().numpy()
    conf = compute_j3d_confidence(keypoints_2d.cpu().detach().numpy())
    res = np.concatenate((res, np.expand_dims(conf, -1)), -1) # (1, num_joints, 4)
    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    with args.results_path.open('w') as fp:
        json.dump(res[0].tolist(), fp, indent=4)
        
@torch.no_grad()
def draw_keypoints(
    image,
    keypoints,
    connectivity = None,
    colors= None,
    radius= 2,
    width= 3,
    ):

    """
    Draws Keypoints on given RGB image.
    The values of the input image should be uint8 between 0 and 255.

    Args:
        image (Tensor): Tensor of shape (3, H, W) and dtype uint8.
        keypoints (Tensor): Tensor of shape (num_instances, K, 2) the K keypoints location for each of the N instances,
            in the format [x, y].
        connectivity (List[Tuple[int, int]]]): A List of tuple where,
            each tuple contains pair of keypoints to be connected.
        colors (str, Tuple): The color can be represented as
            PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
        radius (int): Integer denoting radius of keypoint.
        width (int): Integer denoting width of line connecting keypoints.

    Returns:
        img (Tensor[C, H, W]): Image Tensor of dtype uint8 with keypoints drawn.
    """
    from PIL import Image, ImageColor, ImageDraw, ImageFont
#     if not torch.jit.is_scripting() and not torch.jit.is_tracing():
#         _log_api_usage_once(draw_keypoints)
#     if not isinstance(image, torch.Tensor):
#         raise TypeError(f"The image must be a tensor, got {type(image)}")
#     elif image.dtype != torch.uint8:
#         raise ValueError(f"The image dtype must be uint8, got {image.dtype}")
#     elif image.dim() != 3:
#         raise ValueError("Pass individual images, not batches")
#     elif image.size()[0] != 3:
#         raise ValueError("Pass an RGB image. Other Image formats are not supported")

#     if keypoints.ndim != 3:
#         raise ValueError("keypoints must be of shape (num_instances, K, 2)")

    ndarr = image.permute(1, 2, 0).cpu().numpy()
    img_to_draw = Image.fromarray(ndarr)
    draw = ImageDraw.Draw(img_to_draw)
    img_kpts = keypoints.to(torch.int64).tolist()

    for kpt_id, kpt_inst in enumerate(img_kpts):
        for inst_id, kpt in enumerate(kpt_inst):
            x1 = kpt[0] - radius
            x2 = kpt[0] + radius
            y1 = kpt[1] - radius
            y2 = kpt[1] + radius
            draw.ellipse([x1, y1, x2, y2], fill=colors, outline=None, width=0)

        if connectivity:
            for connection in connectivity:
                start_pt_x = kpt_inst[connection[0]][0]
                start_pt_y = kpt_inst[connection[0]][1]

                end_pt_x = kpt_inst[connection[1]][0]
                end_pt_y = kpt_inst[connection[1]][1]

                draw.line(
                    ((start_pt_x, start_pt_y), (end_pt_x, end_pt_y)),
                    width=width,
                )

    return torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1).to(dtype=torch.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Script for lifting 2d keypoints to 3d for virtual camera based setup.")
    # Path to input / output
    parser.add_argument("input_path", type=Path,
                        help="Path to input data (mesh or point cloud)")
    parser.add_argument("--keypoints-2d-path", "-k2", type=Path,
                        help="Path to 2d keypoints predictions for the renderings "
                             "(if not provided 2D_pose.json in scan_path is assumed)")
    parser.add_argument("--results-path", "-r", type=Path, default=None,
                        help="Path to file for resulting 3d pose "
                             "(if not provided, 3D_pose.json in input_path is saved)")
    parser.add_argument("--camera-path", "-cam", type=Path, default=None,
                        help="Path to file with camera parameters "
                             "(if not provided pytorch3d_params_and_cameras.pkl in scan_path is assumed)")

    # Additional parameters
    parser.add_argument("--config-path", "-c", type=Path, default="config.yml",
                        help="Path to yml file with config")
    parser.add_argument("--cpu", action="store_true",
                        help="Perform computations on cpu (default: False)")

    args = parser.parse_args()
    if args.keypoints_2d_path is None:
        args.keypoints_2d_path = args.input_path / "2D_pose.json"
    if args.camera_path is None:
        args.camera_path = args.input_path / "p3d_render_data.pkl"
    if args.results_path is None:
        args.results_path = args.input_path / "3D_pose.json"

    # Load config file
    config = load_config(args.config_path)
    args.smpl_models_path = Path(config["SMPL_MODELS_PATH"])

    main(args)
