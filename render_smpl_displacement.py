# import sys
# import matplotlib.pyplot as plt
import os
from os.path import join

# import time
import json
import argparse

from glob import glob
from PIL import Image
import cv2

# import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv

from pytorch3d.io import save_obj, load_obj, load_ply
from pytorch3d.structures import Meshes, join_meshes_as_batch, packed_to_list
from pytorch3d.renderer.cameras import look_at_view_transform
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    DirectionalLights, 
    PointLights,
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    BlendParams,
    TexturesUV,
    TexturesAtlas,
    TexturesVertex,
)

# for custom rasterizer, shader
from pytorch3d.renderer.mesh.rasterizer import Fragments, RasterizationSettings
from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
from pytorch3d.renderer.mesh.shading import phong_shading
from pytorch3d.renderer.blending import softmax_rgb_blend, hard_rgb_blend
from pytorch3d.ops import interpolate_face_attributes

"""
Reference: 
    https://github.com/facebookresearch/pytorch3d/issues/854
    https://github.com/facebookresearch/pytorch3d/issues/889
    
    ### TODO
    # 1. load smpl
    # 2. apply disp
    # 3. render mesh with texture
    # 4. run DensePose and get predicted iuv
    # 5. map partial texture with iuv

"""
def convert_to_verts_colors(textures_uv: TexturesUV, meshes: Meshes):
    verts_colors_packed = torch.zeros_like(meshes.verts_packed())
    verts_colors_packed[meshes.faces_packed()] = textures_uv.faces_verts_textures_packed()
    return verts_colors_packed

def convert_to_TexturesVertex(textures_uv: TexturesUV, meshes: Meshes):
    verts_colors_packed = convert_to_verts_colors(textures_uv, meshes)
    return TexturesVertex(packed_to_list(verts_colors_packed, meshes.num_verts_per_mesh()))

class CustomMeshRasterizer(nn.Module):
    def __init__(self, cameras=None, raster_settings=None, verts_uvs=None, faces_uvs=None) -> None:
        """
        Args:
            cameras: A cameras object which has a  `transform_points` method
                which returns the transformed points after applying the
                world-to-view and view-to-ndc transformations.
            raster_settings: the parameters for rasterization. This should be a
                named tuple.
        All these initial settings can be overridden by passing keyword
        arguments to the forward function.
        """
        super().__init__()
        if raster_settings is None:
            raster_settings = RasterizationSettings()

        self.cameras = cameras
        self.raster_settings = raster_settings
        self.verts_uvs = verts_uvs
        self.faces_uvs = faces_uvs

    def to(self, device):
        # Manually move to device cameras as it is not a subclass of nn.Module
        if self.cameras is not None:
            self.cameras = self.cameras.to(device)
        return self

    def transform(self, meshes_world, **kwargs):
        """
        Args:
            meshes_world: a Meshes object representing a batch of meshes with
                vertex coordinates in world space.
        Returns:
            meshes_proj: a Meshes object with the vertex positions projected
            in NDC space
        NOTE: keeping this as a separate function for readability but it could
        be moved into forward.
        """
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of MeshRasterizer"
            raise ValueError(msg)

        n_cameras = len(cameras)
        if n_cameras != 1 and n_cameras != len(meshes_world):
            msg = "Wrong number (%r) of cameras for %r meshes"
            raise ValueError(msg % (n_cameras, len(meshes_world)))

        # replace mesh to UV
        if self.faces_uvs and self.verts_uvs:
            # verts_uvs = meshes_world.textures._verts_uvs_list[0] #(V, 2)
            # faces_uvs = meshes_world.textures._faces_uvs_list[0] #(F, 3)
            verts_uvs    = self.verts_uvs[0]
            faces_uvs    = self.faces_uvs[0]
            z_           = torch.zeros(verts_uvs.size(0), 1).to(verts_uvs)
            verts_uvs_z_ = torch.cat(((verts_uvs*2)-1, z_),dim=1) # (V, 3)
            # verts_uvs_z_ = torch.cat(((verts_uvs-0.5)*2, z_),dim=1) # (V, 3)

            meshes_world = Meshes(
                verts    = [verts_uvs_z_], 
                faces    = [faces_uvs]
            )

        verts_world = meshes_world.verts_padded()

        # NOTE: Retaining view space z coordinate for now.
        # TODO: Revisit whether or not to transform z coordinate to [-1, 1] or
        # [0, 1] range.
        eps = kwargs.get("eps", None)
        verts_view = cameras.get_world_to_view_transform(**kwargs).transform_points(
            verts_world, eps=eps
        )
        # view to NDC transform
        to_ndc_transform = cameras.get_ndc_camera_transform(**kwargs)
        projection_transform = cameras.get_projection_transform(**kwargs).compose(
            to_ndc_transform
        )
        verts_ndc = projection_transform.transform_points(verts_view, eps=eps)

        verts_ndc[..., 2] = verts_view[..., 2]
        meshes_ndc = meshes_world.update_padded(new_verts_padded=verts_ndc)
        return meshes_ndc

    def forward(self, meshes_world, **kwargs):
        """
        Args:
            meshes_world: a Meshes object representing a batch of meshes with
                          coordinates in world space.
        Returns:
            Fragments: Rasterization outputs as a named tuple.
        """
        meshes_proj = self.transform(meshes_world, **kwargs)
        raster_settings = kwargs.get("raster_settings", self.raster_settings)

        # By default, turn on clip_barycentric_coords if blur_radius > 0.
        # When blur_radius > 0, a face can be matched to a pixel that is outside the
        # face, resulting in negative barycentric coordinates.
        clip_barycentric_coords = raster_settings.clip_barycentric_coords
        if clip_barycentric_coords is None:
            clip_barycentric_coords = raster_settings.blur_radius > 0.0

        # If not specified, infer perspective_correct and z_clip_value from the camera
        cameras = kwargs.get("cameras", self.cameras)
        if raster_settings.perspective_correct is not None:
            perspective_correct = raster_settings.perspective_correct
        else:
            perspective_correct = cameras.is_perspective()
        if raster_settings.z_clip_value is not None:
            z_clip = raster_settings.z_clip_value
        else:
            znear = cameras.get_znear()
            if isinstance(znear, torch.Tensor):
                znear = znear.min().item()
            z_clip = None if not perspective_correct or znear is None else znear / 2

        # By default, turn on clip_barycentric_coords if blur_radius > 0.
        # When blur_radius > 0, a face can be matched to a pixel that is outside the
        # face, resulting in negative barycentric coordinates.

        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_proj,
            image_size=raster_settings.image_size,
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            bin_size=raster_settings.bin_size,
            max_faces_per_bin=raster_settings.max_faces_per_bin,
            clip_barycentric_coords=clip_barycentric_coords,
            perspective_correct=perspective_correct,
            cull_backfaces=raster_settings.cull_backfaces,
            z_clip_value=z_clip,
            cull_to_frustum=raster_settings.cull_to_frustum,
        )

        return Fragments(
            pix_to_face=pix_to_face,
            zbuf=zbuf,
            bary_coords=bary_coords,
            dists=dists,
        )

class CustomPhongShader(SoftPhongShader):
    """
    Per pixel lighting - the lighting model is applied using the interpolated
    coordinates and normals for each pixel. The blending function returns the
    soft aggregated color using all the faces per pixel.
    To use the default values, simply initialize the shader with the desired
    device e.g.
    .. code-block::
        shader = SoftPhongShader(device=torch.device("cuda:0"))
    """
    def _get_cameras(self, **kwargs):
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of the shader."
            raise ValueError(msg)

        return cameras

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras      = self._get_cameras(**kwargs)
        texels       = meshes.sample_textures(fragments)
        lights       = kwargs.get("lights", self.lights)
        materials    = kwargs.get("materials", self.materials)
        blend_params = kwargs.get("blend_params", self.blend_params)

        # colors = _phong_shading_with_pixels(
        colors = phong_shading(
            meshes=meshes,
            fragments=fragments,
            texels=texels,
            lights=lights,
            cameras=cameras,
            materials=materials,
        )
        # znear = kwargs.get("znear", getattr(cameras, "znear", 1.0))
        # zfar  = kwargs.get("zfar",  getattr(cameras, "zfar", 100.0))
        # images = softmax_rgb_blend(colors, fragments, blend_params, znear=znear, zfar=zfar)
        images = hard_rgb_blend(colors, fragments, blend_params)
        return images

class ToyRenderer(nn.Module):
    def __init__(self,
            image_size   = 512, 
            device       = None,
            opt          = None,
            verts_uvs    = None,
            faces_uvs    = None,
        ):
        super().__init__()
        self.opt = opt
        self.verts_uvs = verts_uvs
        self.faces_uvs = faces_uvs        

        # the number of different viewpoints from which we want to render the mesh.
        # Initialize a camera.  World coordinates +X left, +Y up and +Z in.
        R, T = look_at_view_transform(
            dist = 1, 
            elev = 0, 
            azim = 0, 
            at = ((0.0, 0.0, 0.0),),
            up = ((0.0, 1.0, 0.0),) 
        )

        # We arbitrarily choose one particular view that will be used to visualize results
        select_camera = FoVOrthographicCameras(
            R=R, 
            T=T,
            device=device
        )

        # back ground color
        blend_params = BlendParams(
            # background_color = [0., 0., 0.]
            background_color = [1., 1., 1.]
        )

        raster_settings = RasterizationSettings(
            image_size      = image_size, 
            blur_radius     = 0.0,
            faces_per_pixel = 1,
            bin_size        = 0,
            # max_faces_per_bin = 64,
            # bin_size = None, 
            # max_faces_per_bin = None
        )

        # lights = PointLights(            
        #     ambient_color  = ((0.5, 0.5, 0.5),),
        #     diffuse_color  = ((0.5, 0.5, 0.5),),
        #     specular_color = ((0.0, 0.0, 0.0),),
        #     location       = ((1.0, 1.0, 1.0),),
        #     device         = device
        # )
        lights = DirectionalLights(
            ambient_color   = [[1.0, 1.0, 1.0]], 
            diffuse_color   = [[0.0, 0.0, 0.0]],  
            specular_color  = [[0.0, 0.0, 0.0]], 
            direction       = T, 
            device          = device
        )

        if False: ########### render in world space
            custom_rasterizer = MeshRasterizer(
                cameras         = select_camera, 
                raster_settings = raster_settings
            )
        else: ####### render in Coordinate space
            custom_rasterizer = CustomMeshRasterizer(
                cameras         = select_camera, 
                raster_settings = raster_settings,
                verts_uvs       = self.verts_uvs,
                faces_uvs       = self.faces_uvs,
            )

        # custom_shader = SoftPhongShader(
        custom_shader = CustomPhongShader( ############## render something else
            device       = device,
            cameras      = select_camera,
            lights       = lights,
            blend_params = blend_params
        )

        renderer = MeshRenderer(
            rasterizer = custom_rasterizer,
            shader     = custom_shader
        )
        self.renderer = renderer
        self.camera = select_camera
        self.lights = lights

    def forward(self, mesh, R, T):
            
        rendered = self.renderer(mesh, cameras=self.camera, lights=self.lights, R=R, T=T)

        return rendered

    def save_data(self, path, rendered, name, angle=-1):
        assert path, ValueError('Cannot save image, no valid path specified')
        _rendered = rendered.permute(0,3,1,2)

        render_out, mask_out = path
                
        for idx, img in enumerate(_rendered):
            if angle != -1:
                azimoth = angle
            else:
                azimoth = self.azim[idx] if type(self.azim) == list else self.azim

            save_path = join(render_out, f"{azimoth}") 
            os.makedirs(save_path, exist_ok=True)
            save_mask = join(mask_out,   f"{azimoth}") 
            os.makedirs(save_mask, exist_ok=True)

            # tv.utils.save_image(img[:3], join(save_path, f"{name}.png"))
            mask = (img[3:] > 0).all(0) * 1.0
            img[3][img[3] > 0] = 1.0
            tv.utils.save_image(img, join(save_path, f"{name}.png"))
            tv.utils.save_image(mask, join(save_mask, f"{name}.png"))
        # tv.utils.save_image(_rendered[0, :3], save_img)

def set_seed(seed):
    from torch.backends import cudnn
    import random
    cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):
    # CUDA
    device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size   = 512
    
    save_path    = args.save_path
    
    ### scan fitted/registered mesh
    orign_verts, orign_faces = load_ply(args.register_smpl_ply)
    orign_meshes = Meshes(verts=[orign_verts], faces=[orign_faces]).to(device)

    ### smpl posed mesh
    smpld_verts, smpld_faces = load_ply(args.original_smpl_ply)
    smpld_meshes = Meshes(verts=[smpld_verts], faces=[smpld_faces]).to(device)
    
    ### UV coordiante
    _, faces, aux= load_obj('smpl.obj', load_textures=False)
    
    ### TexturesUV
    verts_uvs    = aux.verts_uvs.to(device)      # (V, 2)
    faces_uvs    = faces.textures_idx.to(device) # (F, 3)
    
    ### TextureMap
    # tex_map      = tv.transforms.ToTensor()(Image.open('./assets/smplx_texture_rainbow.png'))
    # tex_map      = tv.transforms.ToTensor()(Image.open('./assets/_mask.png'))
    # tex_map      = tex_map[:3].permute(1,2,0)[None].to(device)
    # print(tex_map.min(), tex_map.max()) # 0 ~ 1
    # tex_map      = TexturesUV(verts_uvs=[verts_uvs], faces_uvs=[faces_uvs], maps=tex_map)
    
    ### Mask
    pil_mask     = Image.open('mask_soft.png')
    mask         = tv.transforms.ToTensor()(pil_mask)
    mask         = mask.permute(1,2,0)[None].to(device)[..., :3]
    mask         = TexturesUV(verts_uvs=[verts_uvs], faces_uvs=[faces_uvs], maps=mask)
    vert_mask    = torch.zeros_like(orign_meshes.verts_packed()).to(device)
    vert_mask[orign_meshes.faces_packed()] = mask.faces_verts_textures_packed()
    
    ### calculate displacement
    scan         = smpld_meshes._verts_list[0]
    smpl         = orign_meshes._verts_list[0]
    
    displacement = (scan - smpl) * vert_mask
    
    ### Add displacement to mesh
    orign_meshes._verts_list[0] = smpl + displacement
    
    ### Get displacement as VertexColor
    vert_disp    = torch.zeros_like(vert_mask).to(device)
    vert_disp[orign_meshes.faces_packed()] = displacement[orign_meshes.faces_packed()]
    
    ### Amplify value for rendering
    # vert_disp    = vert_disp*100
    
    ### need to normalize vert_disp and save min, max values as json
    disp_min = vert_disp.min()
    disp_max = vert_disp.max()
    disp_mean = vert_disp.mean()
    # import pdb; pdb.set_trace()
    
    disp_min = disp_min.item()
    disp_max = disp_max.item()
    disp_mean = disp_mean.item()
    
    import json
    import os
    basename = os.path.basename(args.smpl_parms)[:-5]+"_disp.json"
    # shutil.copy(args.smpl_parms, f"./results/{basename}")
    
    with open(args.smpl_parms,'r+') as outfile:
        file_data = json.load(outfile)
        
        ### NEED TO CHECK WITH CSH!!!!!!!!!!!!!!!!!!!!!!!!!!!
        file_data["d_range"] = [disp_min, disp_mean, disp_max]
        
        ######################################################
        if True:
            # this part is for convertion between smplx and smpl
            pose_len = len(file_data["pose"])
            for index in range(pose_len-6, pose_len):
                file_data["pose"][index] = 0.0
        ######################################################
        outfile.seek(0)
        
        # convert back to json.        
        with open( f"{save_path}/{basename}", 'w') as final_file:
            json.dump(file_data, final_file, indent = 4)
        
                                         
    print('>>> SMPL parameters saved to', args.smpl_parms)
    import shutil
    
    
    
    vert_disp = vert_disp - vert_disp.min()
    vert_disp = vert_disp / vert_disp.max()
    
    vert_color_disp = TexturesVertex(packed_to_list(vert_disp, orign_meshes.num_verts_per_mesh()))

    orign_meshes.textures       = vert_color_disp ## render using displacement
    # orign_meshes.textures       = tex_map ## render using texture map
    
    renderer = ToyRenderer(
            image_size  = image_size, 
            verts_uvs   = [verts_uvs], 
            faces_uvs   = [faces_uvs], 
            device      = device
        )
    
    R, T = look_at_view_transform(
            dist   =  1, 
            elev   =  0.0, 
            azim   =  0.0, 
            at     =  ((0.0, 0.0, 0.0),),
            up     =  ((0.0, 1.0, 0.0),),
            device = device
        )
    
    img = renderer(orign_meshes, R=R, T=T)
    img  = img.permute(0,3,1,2)[0]

    img[3][img[3] > 0] = 1.0
    tv.utils.save_image(img, f"{save_path}/{basename}_crop_displacement_map_raw.png")
#     file_crop_displacement_map.png
    # convert to npsave_path
    
    np_img = img.permute(1,2,0).cpu().numpy()
    np_img_refine = blur_padding(np_img[:,:,:3], np_img[:,:,3:4], kernel_size=3, iteration=32)
#     (1 - np_mask) * 0.5 
    np_img_refine = (np_img_refine * 255).astype(np.uint8)
    Image.fromarray(np_img_refine).save(f"{save_path}/{basename}_crop_displacement_map.png")
    print('>>> Done.')
    
def blur_padding(np_image, np_mask, kernel_size=3, iteration=24):
    # import pdb; pdb.set_trace()
    kernel = np.ones((kernel_size,kernel_size))
    
    
    # get RGB and mask
    image = (np_image * 255).astype(np.float)
    # image = np_image
    mask  = np_mask
    
    # erode mask
    erode_mask = cv2.erode(mask, kernel, iterations=3)  # make dilation image        
    erode_mask = erode_mask[...,np.newaxis]
    erode_mask = mask
    
    # masking
    image = image * erode_mask
    new_img = image.copy()
    new_msk = mask.copy()
    
    for _ in range(iteration):
        
        # each color
        dilate_r = cv2.dilate(new_img[..., 0, np.newaxis], kernel, iterations=1)  # make dilation image
        dilate_g = cv2.dilate(new_img[..., 1, np.newaxis], kernel, iterations=1)  # make dilation image
        dilate_b = cv2.dilate(new_img[..., 2, np.newaxis], kernel, iterations=1)  # make dilation image
        
        # mask
        dilate_m = cv2.dilate(new_msk, kernel, iterations=1)  # make dilation image        
        dilate_m = dilate_m[...,np.newaxis]
        dilate_m = dilate_m - new_msk
#         dilate_m[2][dilate_m[2] > 0] = 1.0 
        
        # concatenate all channel
        dilate_image = np.concatenate((
                dilate_r[...,np.newaxis], 
                dilate_g[...,np.newaxis], 
                dilate_b[...,np.newaxis]
            ),axis=2)
        
        # mask for only dilated region
        dilate_image = dilate_image * dilate_m
                        
        # update dilated pixels
        new_img = new_img + dilate_image
        new_msk = new_msk + dilate_m
    new_img = cv2.GaussianBlur(new_img, (7, 7), 0)
    new_img = new_img * (1-mask) + image
    
    # return image / 255
    return new_img / 255

#     final_image = np_image* erode_mask + image/255 * (1-erode_mask)
#     return final_image
    


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run Model')
    parser.add_argument('--original_smpl_ply', '-org', type=str, default='./prev_results/07_refine_smpl.ply')
    parser.add_argument('--register_smpl_ply', '-reg', type=str, default='./prev_results/07_refine_smpld.ply')
    parser.add_argument('--smpl_parms', '-p',          type=str, default='./prev_results/07_refine_smpld_.json')
    parser.add_argument('--save_path', '-s',           type=str, help='save path')
    args = parser.parse_args()

    main(args)
            
