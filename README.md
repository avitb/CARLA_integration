# CARLA_integration

## Lidar library:
1.  Using CARLA 0.9.11
2.  Detects vehicle (car, truck), and motorcycle (motorcycle, bicycle)
3.  get_bboxes returns | cx | cy | cz | distance | length | width | height | orientation | speed |
4.  * v0 : returns bboxes of all detected vehicles and motorcycles, and bbox of EV
    * v1 : returns bboxes of all detected vehicles and motorcycles IN FRONT of EV, and bbox of EV
    * v2 : returns bboxes of all detected vehicles and motorcycles IN FRONT of EV
