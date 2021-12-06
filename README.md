# CARLA_integration

## Lidar library (lib_lidar.py):
1.  Using CARLA 0.9.11
2.  Detects vehicle (car, truck), and motorcycle (motorcycle, bicycle)
3.  get_bboxes returns | id | cx | cy | cz | distance | length | width | height | orientation | speed |
4.  Provides 2 kind of origin frame : ego vehicle origin, and world origin

Example of get_bboxes (bbox frame Ego Vehicle):
```python
bboxes = lib_lidar.get_bboxes(world, vehicle_lidar, bboxes_old, process_time)
while (id < len(bboxes)):
        vehicle_id      = id
        vehicle_cx      = bboxes[id].cx
        vehicle_cy      = bboxes[id].cy
        vehicle_cz      = bboxes[id].cz
        vehicle_dist    = bboxes[id].dist
        vehicle_l       = bboxes[id].l
        vehicle_w       = bboxes[id].w
        vehicle_h       = bboxes[id].h
        vehicle_orient  = bboxes[id].orient
        vehicle_speed   = bboxes[id].speed
        id = id+1
```
5.  * v0 : returns bboxes of all detected vehicles and motorcycles, and bbox of EV
    * v1 : returns bboxes of all detected vehicles and motorcycles IN FRONT of EV, and bbox of EV
    * v2 : returns bboxes of all detected vehicles and motorcycles IN FRONT of EV
    * v3 : v2 added with new class struct_bbox
    * v4 : v1 added with camera lib, transformation function to World frame

Example of transform_world (bbox frame World):
```python
bboxes = lib_lidar.get_bboxes(world, vehicle_lidar, bboxes_old, process_time)
bboxes_world = lib_lidar.transform_world(bboxes, vehicle_lidar)
while (id < len(bboxes)):
        vehicle_id      = id
        vehicle_cx      = bboxes_world[id].cx
        vehicle_cy      = bboxes_world[id].cy
        vehicle_cz      = bboxes_world[id].cz
        vehicle_dist    = bboxes_world[id].dist  #ego vehicle has 0 distance
        vehicle_l       = bboxes_world[id].l
        vehicle_w       = bboxes_world[id].w
        vehicle_h       = bboxes_world[id].h
        vehicle_orient  = bboxes_world[id].orient
        vehicle_speed   = bboxes_world[id].speed
        id = id+1
```
6. Possible bugs:
  * White jittering in camera window
  * World origin frame has incorrect speed parameter value
  * Pop/push algorithm change arbitrarily from FIFO to LIFO
7. Video demo (print World-origin frame bboxes):

Demo<video src='https://user-images.githubusercontent.com/49227721/143976548-89aa0aab-63f9-47a7-a8e3-311a1052ca13.mp4' width=1920/>

demo can be replicated by excecuting main_lidar_test.py


## Camera library (lib_lidar.py):
Object detection with bounding box application with yoloV3. Reference: https://github.com/umtclskn/Carla_Simulator_YOLOV3_Object_Detection

YoloV3 Tensorflow implementation forked from: https://github.com/YunYang1994/tensorflow-yolov3

### Setup
1. From folder camera_library\, put **yolov3_object_detection.py** and **tensorflow_yolov3** into  
> ..\CARLA_0.9.11\WindowsNoEditor\PythonAPI\examples\

for easier access
2. Download COCO weights from this link:
```
https://github.com/YunYang1994/tensorflow-yolov3/releases/download/v1.0/yolov3_coco.tar.gz
```
extract this file under the below path:

> ..\CARLA_0.9.11\WindowsNoEditor\PythonAPI\examples\tensorflow-yolov3\checkpoint

3. Open cmd at: 
> ..\CARLA_0.9.11\WindowsNoEditor\PythonAPI\examples\tensorflow-yolov3 

and run these command:
```
python convert_weight.py 
python freeze_graph.py
```
4. Open and run CarlaUE4.exe
5. Run spawn actor python file for adding pedestrians or vehicles with:
```
python spawn_npc.py
```
6. Run object detection python file to start detecting vehicles, pedestrians or bicycles. with:
```
python yolov3_object_detection.py
```
![image](https://user-images.githubusercontent.com/49227721/144821441-302cc779-1d84-452e-b242-24e6e2aa818d.png)
![image](https://user-images.githubusercontent.com/49227721/144821459-794be0e6-7d3a-4408-a8fc-1108a4a4788d.png)



