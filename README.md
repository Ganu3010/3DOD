# 3D Object Detection


Currently only compatible with SPFormer and ScanNetv2. <br />

### File extensions supported for Point Cloud 
<ol>
    <li>.txt</li>
    <li>.ply</li>
    <li>.npy</li>
    <li>.pcd</li>
    <li>.pth</li>
</ol>

### Data Format allowed in files
```
x y z r g b
```
Each point is defined by its position coordinates (x, y, z) and color (r, g, b) attribute.

### Tu run app use command: 
```
flask --app website run --debug
```


## Installation

Requirements

- Python 3.x
- Pytorch 1.10
- CUDA 11.x or higher

The following installation suppose `python=3.8` `pytorch=1.10` and `cuda=11.4`.

- Create a conda virtual environment

  ```
  conda create -n spformer python=3.8
  conda activate spformer
  ```

- Clone the repository

  ```
  git clone https://github.com/SpookyWooky5/SPFormer.git
  ```

- Install the dependencies

  Install [Pytorch 2.1.2](https://pytorch.org/)

  ```
  pip install spconv-cu120 # or appropriate version
  conda install pytorch-scatter -c pyg # if you face errors, refer to https://data.pyg.org/whl/ to install appropriate versions
  pip install -r requirements.txt
  ```
  To install SPFormer/segmentator from this [repo](https://github.com/Karbo123/segmentator) (We wrap the segmentator in ScanNet), and change:
  ```
  set(CMAKE_CXX_STANDARD 14)
  ```
  to
  ```
  set(CMAKE_CXX_STANDARD 17)
  ```
  in segmentator/csrc/CMakeLists.txt
  Follow remaining steps as specified in segmentator.

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

Put the downloaded `scans` and `scans_test` folder as follows. Symbolic linkages also work if data exists on an external drive.

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
cd website/SPFormer
python tools/test.py configs/spf_scannet.yaml checkpoints/spf_scannet_512.pth
```

## Visualization

Before visualization, you need to write the output results of inference.

```
cd website/SPFormer
python tools/test.py configs/spf_scannet.yaml ${CHECKPOINT} --out ${SAVE_PATH}
```

After inference, run visualization by execute the following command. 

```
python tools/visualization.py --prediction_path ${SAVE_PATH}
```
You can visualize by Open3D or visualize saved `.ply` files on MeshLab. Arguments explaination can be found in `tools/visualiztion.py`.

## Implementation of Project 


<div align ="center">
    <img src="https://github.com/Ganu3010/3DOD/assets/81025296/9002769f-e5df-4215-9694-47088ccfc988"> <br/> <br/>
    Fig. 1: Home Page when user is not logged in <br/> <br/>
    <img src="https://github.com/Ganu3010/3DOD/assets/81025296/cf991fa9-f1a0-44de-ab1b-aa9d54557fd5"> <br/> <br/>
    Fig 2: Login Page <br/> <br/>
    <img src="https://github.com/Ganu3010/3DOD/assets/81025296/7e8421e5-7231-40f0-b1a2-79fec3526254"><br/> <br/>
    Fig. 3: Home Page when user is logged in / Upload Page <br/> <br/>
    <img src="https://github.com/Ganu3010/3DOD/assets/81025296/907d3c86-b069-4053-abfb-a2c88598f845"> <br/> <br/>
    Fig. 4: After uploading valid point cloud file, Process Page  <br/> <br/>
    <img src="https://github.com/Ganu3010/3DOD/assets/81025296/96b3ae65-ea1a-49c6-b57d-42c55ae35d1d"> <br/> <br/>
    Fig. 5: Visualization of input file in Open3D, after clicking Visualize<br/>  <br/>
    <img src="https://github.com/Ganu3010/3DOD/assets/81025296/f2ad8c4d-5b61-463e-9e04-295f2f63ecb2"> <br> <br/>
    Fig. 6: Visualization of output with bounding boxes in Open3D, after clicking Predict <br/>  <br/>
    <img src="https://github.com/Ganu3010/3DOD/assets/81025296/35a18e74-0c35-46c4-9905-e7db11c5f64a"> <br/> <br/>
    Fig. 7: File is downloaded after clicking Export button <br/>  <br/>

</div>





## Acknowledgment
Sincerely thanks to the original SPFormer repo. This repo is build upon it.







