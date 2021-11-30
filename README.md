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

<video src='https://user-images.githubusercontent.com/49227721/143976548-89aa0aab-63f9-47a7-a8e3-311a1052ca13.mp4' width=1920/>









