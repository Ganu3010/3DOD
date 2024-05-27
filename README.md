# 3D Object Detection


Currently only compatible with SPFormer and ScanNetv2. <br />

### File extensions supported for Point Cloud Dataset
<ol>
    <li>.txt</li>
    <li>.ply</li>
    <li>.npy</li>
    <li>.pcd</li>
    <li>.pth</li>
</ol>

```
flask --app website run --debug
```
to run the app. <br />

## Installation

Requirements

- Python 3.x
- Pytorch 1.10
- CUDA 10.x or higher

The following installation suppose `python=3.8` `pytorch=1.10` and `cuda=11.4`.

- Create a conda virtual environment

  ```
  conda create -n spformer python=3.8
  conda activate spformer
  ```

- Clone the repository

  ```
  git clone https://github.com/sunjiahao1999/SPFormer.git
  ```

- Install the dependencies

  Install [Pytorch 1.10](https://pytorch.org/)

  ```
  pip install spconv-cu114
  conda install pytorch-scatter -c pyg
  pip install -r requirements.txt
  ```
  To install SPFormer/segmentator, change:
  ```
  set(CMAKE_CXX_STANDARD 14)
  ```
  to
  ```
  set(CMAKE_CXX_STANDARD 17)
  ```
  in segmentator/csrc/CMakeLists.txt
  Follow remaining steps as specified in segmentator.
  Install segmentator from this [repo](https://github.com/Karbo123/segmentator) (We wrap the segmentator in ScanNet).

- Setup, Install spformer and pointgroup_ops.

  ```
  sudo apt-get install libsparsehash-dev
  python setup.py develop
  cd spformer/lib/
  python setup.py develop
  ```

## Data Preparation

### ScanNet v2 dataset

Download the [ScanNet](http://www.scan-net.org/) v2 dataset.

Put the downloaded `scans` and `scans_test` folder as follows.

```
SPFormer
├── data
│   ├── scannetv2
│   │   ├── scans
│   │   ├── scans_test
```

Split and preprocess data

```
cd data/scannetv2
bash prepare_data.sh
```

The script data into train/val/test folder and preprocess the data. After running the script the scannet dataset structure should look like below.

```
SPFormer
├── data
│   ├── scannetv2
│   │   ├── scans
│   │   ├── scans_test
│   │   ├── train
│   │   ├── val
│   │   ├── test
│   │   ├── val_gt
```

## Pretrained Model

Download [SSTNet](https://drive.google.com/file/d/1vucwdbm6pHRGlUZAYFdK9JmnPVerjNuD/view?usp=sharing) pretrained model (We only use the Sparse 3D U-Net backbone for training).

Move the pretrained model to checkpoints.

```
mkdir checkpoints
mv ${Download_PATH}/sstnet_pretrain.pth checkpoints/
```

## Training

Make sure to download the .pth checkpoint and store in SPFormer/checkpoints. <br />
Also make sure to make modifications in SPFormer/configs/spf_scannet.yml to set training configurations like batch_size and number of epochs. <br />

```
cd website/SPFormer
python tools/train.py configs/spf_scannet.yaml
```
## Inference

Download [SPFormer](https://drive.google.com/file/d/1BKuaLTU3TFgekYAssSVxPO0sHWj-LGlH/view?usp=sharing) pretrain model and move it to checkpoints. Its performance on ScanNet v2 validation set is 56.3/73.9/82.9 in terms of mAP/mAP50/mAP25.

```
python tools/test.py configs/spf_scannet.yaml checkpoints/spf_scannet_512.pth
```

## Visualization

Before visualization, you need to write the output results of inference.

```
python tools/test.py configs/spf_scannet.yaml ${CHECKPOINT} --out ${SAVE_PATH}
```

After inference, run visualization by execute the following command. 

```
python tools/visualization.py --prediction_path ${SAVE_PATH}
```
You can visualize by Open3D or visualize saved `.ply` files on MeshLab. Arguments explaination can be found in `tools/visualiztion.py`.

## Acknowledgment
Sincerely thanks for SPFormer repo. This repo is build upon it.





