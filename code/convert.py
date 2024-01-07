import numpy as np
import open3d as o3d


g_classes = [
    'ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table',
    'chair', 'sofa', 'bookcase', 'board', 'clutter'
]
g_class2label = {cls: i for i, cls in enumerate(g_classes)}


def txt_to_npy(txt_path, out_filename, class_mapping=g_class2label, default_category="clutter"):
    """
    Convert original dataset files to npy file (each line is XYZRGBL).
    We aggregated all the points from each instance in the room.
    L --> Label which is UNCLASSIFIED for now
    
    Args:
        txt_path: path to original file. e.g. Area_1/office_2/office_2.txt
        out_filename: path to save collected points and labels (each line is XYZRGBL)
    Returns:
        None
    Note:
        the points are shifted before save, the most negative point is now at origin.
    """
    
    points = np.loadtxt(txt_path)
    labels = np.ones((points.shape[0], 1)) * class_mapping[default_category]
    pcd = np.hstack((points, labels))
    xyz_min = np.amin(pcd, axis=0)[:3]
    pcd[:, :3] -= xyz_min
    
    np.save(out_filename, pcd)
    print(f"NPY file saved to {out_filename}")

def txt_to_pcd(txt_path, out_filename, class_mapping=g_class2label, default_category="clutter"):
    """
    Convert original dataset files to pcd file (each line is XYZRGB).
    We aggregated all the points from each instance in the room.
    L --> Label which is UNCLASSIFIED for now
    
    Args:
        txt_path: path to original file. e.g. Area_1/office_2/office_2.txt
        out_filename: path to save collected points and labels (each line is XYZRGBL)
    Returns:
        array of labels
    Note:
        the points are shifted before save, the most negative point is now at origin.
        PCD file format requires rgb colors to be in range [0, 1] and not [0, 255]
        this func returns labels array as it cannot be stored in the open3d point cloud
    """
    
    points = np.loadtxt(txt_path)
    labels = np.ones((points.shape[0], 1)) * class_mapping[default_category]
    
    
    points[: -3:] /= 255
    xyz_min = np.amin(points, axis=0)[:3]
    points[:, :3] -= xyz_min
    
    opcd = o3d.geometry.PointCloud()
    opcd.points = o3d.utility.Vector3dVector(points[:, :3])
    opcd.colors = o3d.utility.Vector3dVector(points[:, -3:])
    
    o3d.io.write_point_cloud(out_filename, opcd)
    print(f"PCD file saved to {out_filename}")
    
    return labels

def npy_to_pcd(npy_path, out_filename, class_mapping=g_class2label, default_category="clutter"):
    points = np.load(npy_path)
    pcd = points[:, :-1]
    labels = points[:, -1]
    
    points[: -3:] = np.round(points[: -3:] / 255)
    
    opcd = o3d.geometry.PointCloud()
    opcd.points = o3d.utility.Vector3dVector(points[:, :3])
    opcd.colors = o3d.utility.Vector3dVector(points[:, -3:])
    
    o3d.io.write_point_cloud(out_filename, opcd)
    print(f"PCD file saved to {out_filename}")

def pcd_to_npy(pcd_path, out_filename, labels, class_mapping=g_class2label, default_category="clutter"):
    pcd = o3d.io.read_point_cloud(pcd_path)
    
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) * 255
    
    if labels is None:
        labels = np.ones((points.shape[0], 1)) * class_mapping[default_category]
    
    npy = np.hstack((points, colors, labels.reshape(-1, 1)))
    
    np.save(out_filename, npy)
    print(f"NPY file saved to {out_filename}")
    
    

def read_npy(npy_file):
    data = np.load(npy_file)
    print(data.shape)
    print(data[:3])

if __name__ == '__main__':
    txt_file = '/media/aether/Ultra Touch/BE/BE_Proj/PCD/data/s3dis_aligned/Area_1/conferenceRoom_1/conferenceRoom_1.txt'
    out_file = '/home/aether/pcd/Pointnet_Pointnet2_pytorch/wip/numpy_test.npy'

    # file processed by their function
    # compare_with = '/media/aether/Ultra Touch/BE/BE_Proj/PCD/data/s3dis_aligned/processed/Area_1_conferenceRoom_1.npy'

    txt_to_npy(txt_file, out_file)

    # read_npy(out_file)
    # output
    # (1136617, 7)
    # [[ 4.933  2.703  2.194 71.    64.    54.    12.   ]
    # [ 4.908  2.716  2.178 68.    64.    52.    12.   ]
    # [ 4.92   2.712  2.175 70.    61.    52.    12.   ]]

    # read_npy(compare_with)
    # output
    # (1136617, 7)
    # [[ 4.933  2.703  2.194 71.    64.    54.     3.   ]
    # [ 4.908  2.716  2.178 68.    64.    52.     3.   ]
    # [ 4.92   2.712  2.175 70.    61.    52.     3.   ]]