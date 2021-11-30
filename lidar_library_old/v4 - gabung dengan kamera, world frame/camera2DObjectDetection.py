import glob
import os
import sys
import random
import time
import numpy as np
import cv2
#import torch
try:
    sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

IM_WIDTH = 640
IM_HEIGHT = 480
i_image = 0


def process_img(image, i_image_fungsi):
    i = np.array(image.raw_data)
    #!python detect.py --weights C:/carla9.12/PythonAPI/examples/yolov5/runs/train/exp/weights/best.pt --img 640 --conf 0.25 --source image
    global i_image
    i_image_fungsi = i_image_fungsi + 1
    string_image_save_to_disk = "G:/Akademik/Semester ekstensi/TA2021/SERVER/Integrasi Carla/NEW w kak khansa/Simulasi LIDAR/Bentuk library (CARLA)/v4 - gabung dengan kamera/save/"+str(i_image_fungsi)#"C:/carla9.12/result/"+str( i_image_fungsi )
    i_image = i_image_fungsi
    image_save_to_disk = image.save_to_disk( string_image_save_to_disk )
    # realtime detection
    # string_detect_image = "python detect.py --weights C:/carla9.12/PythonAPI/examples/yolov5/runs/train/exp/weights/best.pt --img 640 --conf 0.25 --source " + string_image_save_to_disk
    # time.sleep(30)
    # os.system( string_detect_image )
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    cv2.imshow("",i3)
    cv2.waitKey(1)
    return i3/255.0 , i_image_fungsi

actor_list = []
try:
    client = carla.Client("localhost", 2000)
    client.set_timeout(5.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    #choose world
    bp = blueprint_library.filter("model3")[0]
    print(bp)
    #choose spawn location
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(bp, spawn_point)
    #autopilot on
    #vehicle.set_autopilot(True)
    vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
    actor_list.append(vehicle)
    #add camera tram
    cam_bp = blueprint_library.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
    cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
    cam_bp.set_attribute("fov", "90")

    spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))

    sensor = world.spawn_actor(cam_bp, spawn_point, attach_to=vehicle)
    actor_list.append(sensor)
    sensor.listen(lambda data : process_img(data, i_image))
    time.sleep(5)
finally:
    for actor in actor_list:
        print("here")
        actor.destroy()
    print("All cleaned up!")

