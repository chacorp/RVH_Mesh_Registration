import torch
import numpy as np
import PIL.Image as Image

from pytorch3d.io import load_obj

from pytorch3d.structures import Meshes, join_meshes_as_batch, packed_to_list
from pytorch3d.renderer.cameras import look_at_view_transform
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    DirectionalLights, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    BlendParams,
    TexturesUV,
    TexturesAtlas,
    TexturesVertex,
)


def convert_to_textureVertex(textures_uv: TexturesUV, meshes:Meshes):
    verts_colors_packed = torch.zeros_like(meshes.verts_packed())
    verts_colors_packed[meshes.faces_packed()] = textures_uv.faces_verts_textures_packed()
    return TexturesVertex(packed_to_list(verts_colors_packed, meshes.num_verts_per_mesh()))

def load_objs_as_meshes(
        files: list,
        device=None,
        load_textures: bool = True,
        tex_files: list = None,
        disp_files: list = None,
        create_texture_atlas: bool = False,
        texture_atlas_size: int = 4,
        texture_wrap = "repeat",
        path_manager = None,
        ratio_change = False
    ):
    """
    when load meshes we need to put on texture


    Load meshes from a list of .obj files using the load_obj function, and
    return them as a Meshes object. This only works for meshes which have a
    single texture image for the whole mesh. See the load_obj function for more
    details. material_colors and normals are not stored.

    Args:
        files: A list of file-like objects (with methods read, readline, tell,
            and seek), pathlib paths or strings containing file names.
        device: Desired device of returned Meshes. Default:
            uses the current device for the default tensor type.
        load_textures: Boolean indicating whether material files are loaded
        create_texture_atlas, texture_atlas_size, texture_wrap: as for load_obj.
        path_manager: optionally a PathManager object to interpret paths.

    Returns:
        New Meshes object.
    """
    mesh_list = []
    for f_obj, f_tex, f_disp in zip(files, tex_files, disp_files):
        verts, faces, aux = load_obj(
            f_obj,
            load_textures=load_textures,
            create_texture_atlas=create_texture_atlas,
            texture_atlas_size=texture_atlas_size,
            texture_wrap=texture_wrap,
            path_manager=path_manager,
        )
        #import pdb;pdb.set_trace()
        verts = verts*(0.01 if ratio_change else 1.0)
        tex = None
        if create_texture_atlas:
            # TexturesAtlas type
            tex = TexturesAtlas(atlas=[aux.texture_atlas.to(device)])
        else:
            # TexturesUV type
            verts_uvs = aux.verts_uvs.to(device)      # (V, 2)
            faces_uvs = faces.textures_idx.to(device) # (F, 3)

            texture = np.array(Image.open(f_tex))     # (H, W, 3)

            if type(f_disp) != int:
                displacement = np.array(Image.open(f_disp))

            denom = 1/255
            # import pdb;pdb.set_trace()
            tex_map  = torch.tensor(texture[..., :3]).unsqueeze(0) * denom
            tex_map  = tex_map.to(device)

            disp_map = torch.tensor(displacement[..., :3]).unsqueeze(0) * denom
            disp_map = disp_map.to(device)

            # pytorch3D texture attribute
            tex      = TexturesUV(verts_uvs=[verts_uvs], faces_uvs=[faces_uvs], maps=tex_map)
            disp     = TexturesUV(verts_uvs=[verts_uvs], faces_uvs=[faces_uvs], maps=disp_map)

        ## set 'disp' as a mesh texture 
        mesh = Meshes(
            verts    = [verts.to(device)], 
            faces    = [faces.verts_idx.to(device)],
            textures = disp
        )
        # import pdb; pdb.set_trace()

        ### sample colors of each vertices from UV
        # vert_displacement = torch.zeros_like(mesh.verts_packed())
        # vert_displacement[mesh.faces_packed()] = disp.faces_verts_textures_packed()
        
        ## TexturesUV -> TexturesVertex
        textures_vertex = convert_to_textureVertex(mesh.textures, mesh)
        vert_displacement = textures_vertex._verts_features_list[0] # (V, 3)
        
        import pdb;pdb.set_trace()
        """Verts offsets must have dimension (all_v, 3)."""
        mesh = mesh.offset_verts_(vert_displacement)
        
        ## replace 'disp' with 'tex' for rendering
        mesh.textures = tex

        mesh_list.append(mesh)
        
    if len(mesh_list) == 1:
        return mesh_list[0]
    return join_meshes_as_batch(mesh_list)





def load_vertex_mask(
        files: str,
        device=None,
        load_textures: bool = True,
        tex_files: str = None,
        disp_files: str = None,
        create_texture_atlas: bool = False,
        texture_atlas_size: int = 4,
        texture_wrap = "repeat",
        path_manager = None,
        ratio_change = False
    ):
    """
    when load meshes we need to put on texture


    Load meshes from a list of .obj files using the load_obj function, and
    return them as a Meshes object. This only works for meshes which have a
    single texture image for the whole mesh. See the load_obj function for more
    details. material_colors and normals are not stored.

    Args:
        files: A list of file-like objects (with methods read, readline, tell,
            and seek), pathlib paths or strings containing file names.
        device: Desired device of returned Meshes. Default:
            uses the current device for the default tensor type.
        load_textures: Boolean indicating whether material files are loaded
        create_texture_atlas, texture_atlas_size, texture_wrap: as for load_obj.
        path_manager: optionally a PathManager object to interpret paths.

    Returns:
        New Meshes object.
    """
    mesh_list = []
    f_obj, f_tex, f_disp  = files, tex_files, disp_files
    verts, faces, aux = load_obj(
        f_obj,
        load_textures=load_textures,
        create_texture_atlas=create_texture_atlas,
        texture_atlas_size=texture_atlas_size,
        texture_wrap=texture_wrap,
        path_manager=path_manager,
    )
    #import pdb;pdb.set_trace()
    verts = verts*(0.01 if ratio_change else 1.0)
    tex = None
    if create_texture_atlas:
        # TexturesAtlas type
        tex = TexturesAtlas(atlas=[aux.texture_atlas.to(device)])
    else:
        # TexturesUV type
        verts_uvs = aux.verts_uvs.to(device)      # (V, 2)
        faces_uvs = faces.textures_idx.to(device) # (F, 3)

        texture = np.array(Image.open(f_tex))     # (H, W, 3)

        if type(f_disp) != int:
            displacement = np.array(Image.open(f_disp))

        denom = 1/255
        # import pdb;pdb.set_trace()
        tex_map  = torch.tensor(texture[..., :3]).unsqueeze(0) * denom
        tex_map  = tex_map.to(device)

        disp_map = torch.tensor(displacement[..., :3]).unsqueeze(0) * denom
        disp_map = disp_map.to(device)

        # pytorch3D texture attribute
        tex      = TexturesUV(verts_uvs=[verts_uvs], faces_uvs=[faces_uvs], maps=tex_map)
        disp     = TexturesUV(verts_uvs=[verts_uvs], faces_uvs=[faces_uvs], maps=disp_map)
        
        
        ## set 'disp' as a mesh texture 
        mesh = Meshes(
            verts    = [verts.to(device)], 
            faces    = [faces.verts_idx.to(device)],
            textures = disp
        )

        ## TexturesUV -> TexturesVertex
        textures_vertex = convert_to_textureVertex(mesh.textures, mesh)
        vert_displacement = textures_vertex._verts_features_list[0] # (V, 3)
        return vert_displacement

if __name__ == "__main__":
    load_objs_as_meshes(
        files       = ['smpl.obj'], 
        tex_files   = ['data/testset/rp_aaron_posed_005_smpld.json_crop_displacement_map_raw.png'],
        disp_files  = ['data/testset/rp_aaron_posed_005_smpld.json_crop_displacement_map_raw.png']
    )
