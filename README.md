# 3D Object Detection


Currently only compatible with SPFormer and ScanNetv2. <br />

Dependencies and requirements will be updated soon. <br />


```
flask --app website run --debug
```

### TODO
<ol>
    <li></li>
    <li>Choose better colours for objects</li>
    <li>BB generation and storage can be optimized</li>
    <li>utils.get_bouding_boxes and utils.get_cord_colors have overlapping processing</li>
    <li>experiment with different confidence thresholds</li>
    <li>DB time does not match IST</li>
    <li>Add checkboxes on /process for overwrite_previos and visualizing by instance/class</li>
    <li>Create page for Experiments DB to easily visualize, edit, delete past experiments</li>
    <li><s>Display Open3D visualization in same window as website</s> Look for alternatives to Open3D</li>
</ol>

### Installation
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
