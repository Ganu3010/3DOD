# 3D Object Detection


Currently only compatible with SPFormer and ScanNetv2. <br />

```
flask --app website run --debug
```
to run the app. <br />

### TODO
<ol>
    <li>BB generation and storage can be optimized</li>
    <li>utils.get_bouding_boxes and utils.get_cord_colors have overlapping processing</li>
    <li>DB time does not match system time</li>
    <li>Add checkboxes on /process for overwrite_previos and input for confidence interval</li>
    <li>Look for alternatives to Open3D</li>
</ol>

### Installation
requirements.txt is provided just for reference. Please use below points for proper installation.
<ol>
    <li>Flask</li>
    <li>Refer to spookywooky5/SPFormer to install dependencies </li>
    <li>
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
    </li>
    <li>Create checkpoint dir in SPFormer and add the checkpoint in it</li>
</ol>

### Training
SPFormer requires that all data be stored in SPFormer/data/scannetv2. <br />
Please store accordingly or use symbolic linkages as needed.
Make sure to download the .pth checkpoint and store in SPFormer/checkpoints. Refer to SPFormer docs for the file and more details. <br />
Also make sure to make modifications in SPFormer/configs/spf_scannet.yml to set training configurations like batch_size and number of epochs. <br />

```
cd website/SPFormer
python tools/train.py configs/spf_scannet.yaml
```