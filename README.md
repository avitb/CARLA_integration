# CARLA_integration

## Lidar library (lib_lidar.py):
1.  Using CARLA 0.9.11
2.  Detects vehicle (car, truck), and motorcycle (motorcycle, bicycle)
3.  get_bboxes returns | cx | cy | cz | distance | length | width | height | orientation | speed |
4.  * v0 : returns bboxes of all detected vehicles and motorcycles, and bbox of EV
    * v1 : returns bboxes of all detected vehicles and motorcycles IN FRONT of EV, and bbox of EV
    * v2 : returns bboxes of all detected vehicles and motorcycles IN FRONT of EV
5. Video demo:

<video src='https://user-images.githubusercontent.com/49227721/142869941-24f73605-4a41-47c8-8f61-a02a3008f253.mp4' width=1920/>

