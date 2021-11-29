# CARLA_integration

## Lidar library (lib_lidar.py):
1.  Using CARLA 0.9.11
2.  Detects vehicle (car, truck), and motorcycle (motorcycle, bicycle)
3.  get_bboxes returns | id | cx | cy | cz | distance | length | width | height | orientation | speed |
Example of get_bboxes:
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
4.  * v0 : returns bboxes of all detected vehicles and motorcycles, and bbox of EV
    * v1 : returns bboxes of all detected vehicles and motorcycles IN FRONT of EV, and bbox of EV
    * v2 : returns bboxes of all detected vehicles and motorcycles IN FRONT of EV
    * v3 : v2 added with new class struct_bbox
5. Video demo:

<video src='https://raw.githubusercontent.com/avitb/CARLA_integration/main/demo_lidar_v3.mp4' width=1920/>

