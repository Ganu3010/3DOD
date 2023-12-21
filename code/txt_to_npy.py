import numpy as np

def txt_to_npy(txt_path):
    """ Convert original dataset files to npy file (each line is XYZRGBL).
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
    g_classes = [
        'ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table',
        'chair', 'sofa', 'bookcase', 'board', 'clutter'
    ]
    g_class2label = {cls: i for i, cls in enumerate(g_classes)}

    points_list = []

    cls = "clutter"
    points = np.loadtxt(txt_path)
    labels = np.ones((points.shape[0], 1)) * g_class2label[cls]
    points_list.append(np.concatenate([points, labels], 1))

    data_label = np.concatenate(points_list, 0)

    xyz_min = np.amin(data_label, axis=0)[0:3]
    data_label[:, 0:3] -= xyz_min

    return data_label

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