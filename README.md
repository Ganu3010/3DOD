# 3D Object Detection


Currently only compatible with SPFormer and ScanNetv2. <br />

Dependencies and requirements will be updated soon. <br />


```
flask --app website run --debug
```

### TODO
<ol>
    <li>Add txt, npy, pcd file compatibility in visualize and predict</li>
    <li>Display bounding boxes around visualized object instances</li>
    <li>Display Open3D visualization in same window as website</li>
    <li>Add Export functionality to export predicted bounding boxes</li>
    <li>Create DB to track experiments and point cloud files</li>
    <li>Create page for above DB to easily visualize past experiments and edit if necessary</li>
</ol>

### Installation
<ol>
    <li>Flask</li>
    <li>Refer to spookywooky5/SPFormer ro install dependencies </li>
    <li>To install SPFormer/segmentator, change:
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
