import os

import torch
# import gorilla
import numpy as np
import open3d as o3d

from tqdm import tqdm
from flask import current_app


COLOR_DETECTRON2 = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        # 0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        # 0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        # 0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        # 0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        # 0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.857, 0.857, 0.857,
        # 1.000, 1.000, 1.000
    ]).astype(np.float32).reshape(-1, 3) * 255


def to_pcd(ip_file):
    if ip_file.endswith('.pcd'):
        return ip_file
    
    output_file = ip_file.split('.')[0] + '.pcd'
    points = np.loadtxt(ip_file) if ip_file.endswith('txt') else np.load(ip_file)
    
    # TODO: Verify if we really need to do this.
    points[:, -3:] /= 255
    xyz_min = np.amin(points, axis=0)[:3]
    points[:, :3] -= xyz_min
    
    op = o3d.geometry.PointCloud()
    op.points = o3d.utility.Vector3dVector(points[:, :3])
    op.colors = o3d.utility.Vector3dVector(points[:, -3:])
    o3d.io.write_point_cloud(output_file, op)
    
    return output_file


def preprocess(filepath, model, dataset):
    '''
    Preprocesses point cloud files to .pth files compatible with SPFormer.
    Only works with .ply files for now.
    If .pth file, returns filepath of the same as it is already processed.
    '''
    
    if model == 'spformer' and dataset == 'scannetv2':
        if filepath.endswith('.pth'):
            return filepath

        import plyfile
        import segmentator

        if not filepath.endswith('.ply'):
            pcd_file = to_pcd(filepath)
            pcd = o3d.io.read_point_cloud(pcd_file)
            o3d.io.write_point_cloud(pcd_file.split('.')[0]+'.ply', pcd)
            filepath = pcd_file.split('.')[0]+'.ply'

            
        file = plyfile.PlyData.read(filepath)                
        points = np.array([list(x) for x in file.elements[0]])
        coords = np.ascontiguousarray(points[:, :3] - points[:, :3].mean(0))
        colors = np.ascontiguousarray(points[:, 3:6]) / 127.5 - 1

        mesh = o3d.io.read_triangle_mesh(filepath)
        vertices = torch.from_numpy(np.array(mesh.vertices).astype(np.float32))
        faces = torch.from_numpy(np.array(mesh.triangles).astype(np.int64))
        superpoint = segmentator.segment_mesh(vertices, faces).numpy()

        filepath, filename = os.path.split(filepath)
        save_path = os.path.join(filepath.replace('input', 'preprocessed'), filename.split('.')[0] + '.pth')
        torch.save((coords, colors, superpoint), save_path)
        return save_path


def process(save_path, model, dataset):
    '''
    Processes .pth files and saves predicted instances to static/output/filename/pred_instances
    Also saves output to static/output/filename.ply
    If static/output/filename/pred_instances exists, skips prediction.
    website/SPFormer/configs and website/SPFormer/checkpoints dirs are needed for processing.
    '''
    app = current_app
    if model == 'spformer' and dataset == 'scannetv2':
        output = os.path.join(app.config['ROOT_FOLDER'], 'static', 'output', save_path.split('.')[0])
        
        # TODO: Check if this works as intended
        if os.path.isdir(os.path.join(output, 'pred_instances')):
            return output


        from .SPFormer.spformer.model import SPFormer
        from .SPFormer.spformer.dataset import build_dataloader, build_dataset
        from .SPFormer.spformer.utils import get_root_logger, save_pred_instances

        spformer_root = os.path.join(app.config['ROOT_FOLDER'], 'SPFormer')
        config = os.path.join(spformer_root, 'configs', 'spf_scannet.yaml')
        checkpoint = os.path.join(spformer_root, 'checkpoints', 'spf_scannet_512.pth')

        cfg = gorilla.Config.fromfile(config)

        cfg.data.test.prefix = 'preprocessed'
        cfg.data.test.suffix = save_path
        cfg.data.test.data_root = os.path.split(app.config['UPLOAD_FOLDER'])[0]

        gorilla.set_random_seed(cfg.test.seed)
        logger = get_root_logger()

        model = SPFormer(**cfg.model).cuda()
        gorilla.load_checkpoint(model, checkpoint, strict=False)

        dataset = build_dataset(cfg.data.test, logger)
        dataloader = build_dataloader(
            dataset, training=False, **cfg.dataloader.test)

        results, scan_ids, pred_insts = [], [], []

        progress_bar = tqdm(total=len(dataloader))
        with torch.no_grad():
            model.eval()
            for batch in dataloader:
                result = model(batch, mode='predict')
                results.append(result)
                progress_bar.update()
            progress_bar.close()

        for res in results:
            scan_ids.append(res['scan_id'])
            pred_insts.append(res['pred_instances'])

        save_pred_instances(output, 'pred_instance',
                            scan_ids, pred_insts, dataset.NYU_ID)

        xyz, rgb = get_coords_color(output)
        points = xyz[:, :3]
        colors = rgb / 255

        output_ply = os.path.join(output, 'output.ply')
        write_ply(points, colors, None, output_ply)

        return output



    
def get_bounding_boxes(output):

    pcd = o3d.io.read_point_cloud(os.path.join(output, 'output.ply'))
    xyz = np.asarray(pcd.points, dtype=np.float64)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    f = open(os.path.join(output, 'pred_instance', '.txt'), 'r')
    masks = f.readlines()
    masks = [mask.rstrip().split() for mask in masks]
    ins_num = len(masks)
    ins_pointnum = np.zeros(ins_num)
    inst_label = -100 * np.ones(len(pcd.points)).astype(int)
    scores = np.array([float(x[-1]) for x in masks])
    sort_inds = np.argsort(scores)[::-1]
    for i_ in range(len(masks)-1, -1, -1):
        i = sort_inds[i_]
        mask_path = os.path.join(output, 'pred_instance', masks[i][0])
        assert os.path.isfile(mask_path), mask_path
        if (float(masks[i][2])<0.09):
            continue
        mask = np.array(open(mask_path).read().splitlines(), dtype=int)
        ins_pointnum[i] = mask.sum()
        inst_label[mask == 1] = i
    sort_idx = np.argsort(ins_pointnum)[::-1]

    for _sort_id in range(ins_num):
        in_points = np.array(xyz[inst_label == sort_idx[_sort_id]])
        if len(in_points)>0:
            bounding_box = o3d.geometry.AxisAlignedBoundingBox().create_from_points(o3d.utility.Vector3dVector(in_points))
            bounding_box.color = [1.0, 0.0, 0.0]
            vis.add_geometry(bounding_box)
        
    vis.run()
    vis.destroy_window()

def get_coords_color(output):
    '''
    Helper function from SPFormer/tools/visualize.py
    Modified to work without validation labels.
    '''
    app = current_app
    file = os.path.join(app.config['ROOT_FOLDER'], 'static', 'preprocessed', os.path.split(output)[-1].split('.')[0] + '.pth')
    
    xyz, rgb, superpoint = torch.load(file)
    rgb = (rgb + 1) * 127.5

    f = open(os.path.join(output, 'pred_instance', '.txt'), 'r')
    masks = f.readlines()
    masks = [mask.rstrip().split() for mask in masks]
    inst_label_pred_rgb = np.zeros(rgb.shape)  # np.ones(rgb.shape) * 255 #

    ins_num = len(masks)
    ins_pointnum = np.zeros(ins_num)
    inst_label = -100 * np.ones(rgb.shape[0]).astype(int)

    # sort score such that high score has high priority for visualization
    scores = np.array([float(x[-1]) for x in masks])
    sort_inds = np.argsort(scores)[::-1]
    for i_ in range(len(masks) - 1, -1, -1):
        i = sort_inds[i_]
        mask_path = os.path.join(output, 'pred_instance', masks[i][0])
        assert os.path.isfile(mask_path), mask_path
        if (float(masks[i][2]) < 0.09):
            continue
        mask = np.array(open(mask_path).read().splitlines(), dtype=int)
        ins_pointnum[i] = mask.sum()
        inst_label[mask == 1] = i
    sort_idx = np.argsort(ins_pointnum)[::-1]
    for _sort_id in range(ins_num):
        inst_label_pred_rgb[inst_label == sort_idx[_sort_id]] = COLOR_DETECTRON2[_sort_id % len(COLOR_DETECTRON2)]
    rgb = inst_label_pred_rgb

    return xyz, rgb


def write_ply(verts, colors, indices, output_file):
    '''
    Helper function taken directly from SPFormer/tools/visualize.py
    '''
    
    if colors is None:
        colors = np.zeros_like(verts)
    if indices is None:
        indices = []

    file = open(output_file, 'w')
    file.write('ply')
    file.write('\nformat ascii 1.0\n')
    file.write('element vertex {:d}\n'.format(len(verts)))
    file.write('property float x\n')
    file.write('property float y\n')
    file.write('property float z\n')
    file.write('property uchar red\n')
    file.write('property uchar green\n')
    file.write('property uchar blue\n')
    file.write('element face {:d}\n'.format(len(indices)))
    file.write('property list uchar uint vertex_indices\n')
    file.write('end_header\n')
    for vert, color in zip(verts, colors):
        file.write('{:f} {:f} {:f} {:d} {:d} {:d}\n'.format(vert[0], vert[1], vert[2], int(color[0] * 255),
                                                            int(color[1] * 255), int(color[2] * 255)))
    for ind in indices:
        file.write('3 {:d} {:d} {:d}\n'.format(ind[0], ind[1], ind[2]))
    file.close()
