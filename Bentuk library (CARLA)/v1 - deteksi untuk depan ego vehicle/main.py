import lib_lidar

import glob
import os
import sys
import argparse
import time
from datetime import datetime
import random
import numpy as np
from matplotlib import cm
import open3d as o3d
import math

try:
	sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
		sys.version_info.major,
		sys.version_info.minor,
		'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
	pass

import carla

if __name__ == "__main__":
	argparser = argparse.ArgumentParser(
		description=__doc__)
	argparser.add_argument(
		'--host',
		metavar='H',
		default='localhost',
		help='IP of the host CARLA Simulator (default: localhost)')
	argparser.add_argument(
		'-p', '--port',
		metavar='P',
		default=2000,
		type=int,
		help='TCP port of CARLA Simulator (default: 2000)')
	argparser.add_argument(
		'--no-rendering',
		action='store_true',
		help='use the no-rendering mode which will provide some extra'
		' performance but you will lose the articulated objects in the'
		' lidar, such as pedestrians')
	argparser.add_argument(
		'--semantic',
		action='store_true',
		help='use the semantic lidar instead, which provides ground truth'
		' information')
	argparser.add_argument(
		'--no-noise',
		action='store_true',
		help='remove the drop off and noise from the normal (non-semantic) lidar')
	argparser.add_argument(
		'--no-autopilot',
		action='store_false',
		help='disables the autopilot so the vehicle will remain stopped')
	argparser.add_argument(
		'--show-axis',
		action='store_true',
		help='show the cartesian coordinates axis')
	argparser.add_argument(
		'--filter',
		metavar='PATTERN',
		default='model3',
		help='actor filter (default: "vehicle.*")')
	argparser.add_argument(
		'--upper-fov',
		default=15.0,
		type=float,
		help='lidar\'s upper field of view in degrees (default: 15.0)')
	argparser.add_argument(
		'--lower-fov',
		default=-25.0,
		type=float,
		help='lidar\'s lower field of view in degrees (default: -25.0)')
	argparser.add_argument(
		'--channels',
		default=32.0,
		type=float,
		help='lidar\'s channel count (default: 64)')
	argparser.add_argument(
		'--range',
		default=100.0,
		type=float,
		help='lidar\'s maximum range in meters (default: 100.0)')
	argparser.add_argument(
		'--points-per-second',
		default=500000,
		type=int,
		help='lidar\'s points per second (default: 500000)')
	argparser.add_argument(
		'-x',
		default=0.0,
		type=float,
		help='offset in the sensor position in the X-axis in meters (default: 0.0)')
	argparser.add_argument(
		'-y',
		default=0.0,
		type=float,
		help='offset in the sensor position in the Y-axis in meters (default: 0.0)')
	argparser.add_argument(
		'-z',
		default=0.0,
		type=float,
		help='offset in the sensor position in the Z-axis in meters (default: 0.0)')
	arg = argparser.parse_args()
	#args = argparser.parse_args()

	try:
		#lib_lidar.main_code(args)
		client = carla.Client(arg.host, arg.port)
		client.set_timeout(5.0)
		world = client.get_world()
		#"""

		try:
			original_settings = world.get_settings()
			settings = world.get_settings()
			traffic_manager = client.get_trafficmanager(8000)
			traffic_manager.set_synchronous_mode(True)

			delta = 0.05

			settings.fixed_delta_seconds = delta
			settings.synchronous_mode = True
			settings.no_rendering_mode = arg.no_rendering
			world.apply_settings(settings)

			blueprint_library = world.get_blueprint_library()
			
			#""" SPAWN EGO VEHICLE
			#=====================================================
			vehicle_bp = blueprint_library.filter(arg.filter)[0]
			vehicle_transform = random.choice(world.get_map().get_spawn_points())
			vehicle_lidar = world.spawn_actor(vehicle_bp, vehicle_transform)
			vehicle_lidar.set_autopilot(arg.no_autopilot)
			#"""
			
			# SPAWN LIDAR
			#=====================================================
			lidar_bp = lib_lidar.generate_lidar_bp(arg, world, blueprint_library, delta)
			user_offset = carla.Location(arg.x, arg.y, arg.z)
			lidar_transform = carla.Transform(carla.Location(x=-0.5, z=1.8) + user_offset)
			lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle_lidar)
			
			point_list = o3d.geometry.PointCloud()
			
			if arg.semantic:
				lidar.listen(lambda data: lib_lidar.semantic_lidar_callback(data, point_list))
			else:
				lidar.listen(lambda data: lib_lidar.lidar_callback(data, point_list))

			#GENERATE VISUALIZER
			#=====================================================
			vis = o3d.visualization.Visualizer()
			vis.create_window(
				window_name='Carla Lidar Visualization',
				width=960,
				height=540,
				left=480,
				top=270)
			vis.get_render_option().background_color = [0.05, 0.05, 0.05]
			vis.get_render_option().point_size = 1
			vis.get_render_option().show_coordinate_frame = True
			
			if arg.show_axis:
				lib_lidar.add_open3d_axis(vis)

			frame = 0
			dt0 = datetime.now()
			bboxes_old = {}
			
			while True:
				if frame%6 == 1:
					vis.add_geometry(point_list)
				vis.update_geometry(point_list)
				#vis.add_geometry(point_list)
				
				#RUN DETECT AND VISUALIZATION
				#=====================================================
				lib_lidar.detect_loop(world, frame, lidar, vehicle_lidar, vis, dt0)
				
				vis.poll_events()
				vis.update_renderer()
				# # This can fix Open3D jittering issues:
				time.sleep(0.005)

				world.tick()
                
				process_time = datetime.now() - dt0
				
				#PRINT BBOX ARRAY
				#=====================================================
				bboxes = lib_lidar.get_bboxes(world, vehicle_lidar, bboxes_old, process_time)
				bboxes_old = bboxes
				print(bboxes)
				
				sys.stdout.write('\r' + 'FPS: ' + str(1.0 / process_time.total_seconds()))
				sys.stdout.flush()
				#sys.stdout.write(bboxes+'\n')
				dt0 = datetime.now()
				frame += 1
		finally:
			world.apply_settings(original_settings)
			traffic_manager.set_synchronous_mode(False)

			vehicle_lidar.destroy()
			lidar.destroy()
			vis.destroy_window()

	except KeyboardInterrupt:
		print(' - Exited by user.')