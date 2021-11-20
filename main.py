from __future__ import print_function
from __future__ import division

import lib_lidar
import controller2d_v2
import configparser

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
import logging

try:
	sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
		sys.version_info.major,
		sys.version_info.minor,
		'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
	pass

# Script level imports
sys.path.append(os.path.abspath(sys.path[0] + '/..'))
import carla

"""
Configurable params
"""
ITER_FOR_SIM_TIMESTEP  = 10	 # no. iterations to compute approx sim timestep
WAIT_TIME_BEFORE_START = 5.00   # game seconds (time before controller start)
TOTAL_RUN_TIME		 = 200.00 # game seconds (total runtime before sim end)
TOTAL_FRAME_BUFFER	 = 300	# number of frames to buffer after total runtime
NUM_PEDESTRIANS		= 0	  # total number of pedestrians to spawn
NUM_VEHICLES		   = 0	  # total number of vehicles to spawn
SEED_PEDESTRIANS	   = 0	  # seed for pedestrian spawn randomizer
SEED_VEHICLES		  = 0	  # seed for vehicle spawn randomizer

WEATHERID = {
	"DEFAULT": 0,
	"CLEARNOON": 1,
	"CLOUDYNOON": 2,
	"WETNOON": 3,
	"WETCLOUDYNOON": 4,
	"MIDRAINYNOON": 5,
	"HARDRAINNOON": 6,
	"SOFTRAINNOON": 7,
	"CLEARSUNSET": 8,
	"CLOUDYSUNSET": 9,
	"WETSUNSET": 10,
	"WETCLOUDYSUNSET": 11,
	"MIDRAINSUNSET": 12,
	"HARDRAINSUNSET": 13,
	"SOFTRAINSUNSET": 14,
}
SIMWEATHER = WEATHERID["CLEARNOON"]	 # set simulation weather

PLAYER_START_INDEX = 1	  # spawn index for player (keep to 1)
FIGSIZE_X_INCHES   = 8	  # x figure size of feedback in inches
FIGSIZE_Y_INCHES   = 8	  # y figure size of feedback in inches
PLOT_LEFT		  = 0.1	# in fractions of figure width and height
PLOT_BOT		   = 0.1	
PLOT_WIDTH		 = 0.8
PLOT_HEIGHT		= 0.8

WAYPOINTS_FILENAME = 'trajectory_v2.txt'  # waypoint file to load
DIST_THRESHOLD_TO_LAST_WAYPOINT = 2.0  # some distance from last position before
									   # simulation ends
									   
# Path interpolation parameters
INTERP_MAX_POINTS_PLOT	= 10   # number of points used for displaying
								 # lookahead path
INTERP_LOOKAHEAD_DISTANCE = 20   # lookahead in meters
INTERP_DISTANCE_RES	   = 0.01 # distance between interpolated points

# controller output directory
CONTROLLER_OUTPUT_FOLDER = os.path.dirname(os.path.realpath(__file__)) +\
						   '/controller_output/'

class Timer(object):
	""" Timer Class
	
	The steps are used to calculate FPS, while the lap or seconds since lap is
	used to compute elapsed time.
	"""
	def __init__(self, period):
		self.step = 0
		self._lap_step = 0
		self._lap_time = time.time()
		self._period_for_lap = period

	def tick(self):
		self.step += 1

	def has_exceeded_lap_period(self):
		if self.elapsed_seconds_since_lap() >= self._period_for_lap:
			return True
		else:
			return False

	def lap(self):
		self._lap_step = self.step
		self._lap_time = time.time()

	def ticks_per_second(self):
		return float(self.step - self._lap_step) /\
					 self.elapsed_seconds_since_lap()

	def elapsed_seconds_since_lap(self):
		return time.time() - self._lap_time

def get_current_pose(measurement):
	"""Obtains current x,y,yaw pose from the client measurements
	
	Obtains the current x,y, and yaw pose from the client measurements.

	Args:
		measurement: The CARLA client measurements (from read_data())

	Returns: (x, y, yaw)
		x: X position in meters
		y: Y position in meters
		yaw: Yaw position in radians
	"""
	#x   = measurement.player_measurements.transform.location.x
	#y   = measurement.player_measurements.transform.location.y
	#yaw = math.radians(measurement.player_measurements.transform.rotation.yaw)
	x = measurement[0]['cx']
	y = measurement[0]['cy']
	yaw = measurement[0]['orient']

	return (x, y, yaw)

def get_start_pos(scene):
	"""Obtains player start x,y, yaw pose from the scene
	
	Obtains the player x,y, and yaw pose from the scene.

	Args:
		scene: The CARLA scene object

	Returns: (x, y, yaw)
		x: X position in meters
		y: Y position in meters
		yaw: Yaw position in radians
	"""
	x = scene.player_start_spots[0].location.x
	y = scene.player_start_spots[0].location.y
	yaw = math.radians(scene.player_start_spots[0].rotation.yaw)

	return (x, y, yaw)

def send_control_command(client, throttle, steer, brake, 
						 hand_brake=False, reverse=False):
	"""Send control command to CARLA client.
	
	Send control command to CARLA client.

	Args:
		client: The CARLA client object
		throttle: Throttle command for the sim car [0, 1]
		steer: Steer command for the sim car [-1, 1]
		brake: Brake command for the sim car [0, 1]
		hand_brake: Whether the hand brake is engaged
		reverse: Whether the sim car is in the reverse gear
	"""
	control = VehicleControl()
	# Clamp all values within their limits
	steer = np.fmax(np.fmin(steer, 1.0), -1.0)
	throttle = np.fmax(np.fmin(throttle, 1.0), 0)
	brake = np.fmax(np.fmin(brake, 1.0), 0)

	control.steer = steer
	control.throttle = throttle
	control.brake = brake
	control.hand_brake = hand_brake
	control.reverse = reverse
	client.apply_control(control)

def create_controller_output_dir(output_folder):
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)

def store_trajectory_plot(graph, fname):
	""" Store the resulting plot.
	"""
	create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)

	file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, fname)
	graph.savefig(file_name)

def write_trajectory_file(x_list, y_list, v_list, t_list):
	create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)
	file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, 'trajectory.txt')

	with open(file_name, 'w') as trajectory_file: 
		for i in range(len(x_list)):
			trajectory_file.write('%3.3f, %3.3f, %2.3f, %6.3f\n' %\
								  (x_list[i], y_list[i], v_list[i], t_list[i]))



def exec_waypoint_nav_demo(arg):
	""" Executes waypoint navigation demo.
	"""
	with carla.Client(arg.host, arg.port) as client:
		client.set_timeout(5.0)
		world = client.get_world()
		#"""

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
			
		#############################################
		# Load Configurations
		#############################################

		# Load configuration file (options.cfg) and then parses for the various
		# options. Here we have two main options:
		# live_plotting and live_plotting_period, which controls whether
		# live plotting is enabled or how often the live plotter updates
		# during the simulation run.
		config = configparser.ConfigParser()
		config.read(os.path.join(
				os.path.dirname(os.path.realpath(__file__)), 'options.cfg'))		 
		demo_opt = config['Demo Parameters']

		#############################################
		# Load Waypoints
		#############################################
		# Opens the waypoint file and stores it to "waypoints"
		waypoints_file = WAYPOINTS_FILENAME
		waypoints_np   = None
		with open(waypoints_file) as waypoints_file_handle:
			waypoints = list(csv.reader(waypoints_file_handle, 
										delimiter=',',
										quoting=csv.QUOTE_NONNUMERIC))
			waypoints_np = np.array(waypoints)
			
		# Linear interpolation computations
		# Compute a list of distances between waypoints
		wp_distance = []   # distance array
		for i in range(1, waypoints_np.shape[0]):
			wp_distance.append(
					np.sqrt((waypoints_np[i, 0] - waypoints_np[i-1, 0])**2 +
							(waypoints_np[i, 1] - waypoints_np[i-1, 1])**2))
		wp_distance.append(0)  # last distance is 0 because it is the distance
							   # from the last waypoint to the last waypoint

		# Linearly interpolate between waypoints and store in a list
		wp_interp	  = []	# interpolated values 
							   # (rows = waypoints, columns = [x, y, v])
		wp_interp_hash = []	# hash table which indexes waypoints_np
							   # to the index of the waypoint in wp_interp
		interp_counter = 0	 # counter for current interpolated point index
		for i in range(waypoints_np.shape[0] - 1):
			# Add original waypoint to interpolated waypoints list (and append
			# it to the hash table)
			wp_interp.append(list(waypoints_np[i]))
			wp_interp_hash.append(interp_counter)   
			interp_counter+=1
			
			# Interpolate to the next waypoint. First compute the number of
			# points to interpolate based on the desired resolution and
			# incrementally add interpolated points until the next waypoint
			# is about to be reached.
			num_pts_to_interp = int(np.floor(wp_distance[i] /\
										 float(INTERP_DISTANCE_RES)) - 1)
			wp_vector = waypoints_np[i+1] - waypoints_np[i]
			wp_uvector = wp_vector / np.linalg.norm(wp_vector)
			for j in range(num_pts_to_interp):
				next_wp_vector = INTERP_DISTANCE_RES * float(j+1) * wp_uvector
				wp_interp.append(list(waypoints_np[i] + next_wp_vector))
				interp_counter+=1
		# add last waypoint at the end
		wp_interp.append(list(waypoints_np[-1]))
		wp_interp_hash.append(interp_counter)   
		interp_counter+=1

		#############################################
		# Controller 2D Class Declaration
		#############################################
		# This is where we take the controller2d.py class
		# and apply it to the simulator
		controller = controller2d_v2.Controller2D(waypoints)

		#############################################
		# Determine simulation average timestep (and total frames)
		#############################################
		# Ensure at least one frame is used to compute average timestep
		num_iterations = ITER_FOR_SIM_TIMESTEP
		if (ITER_FOR_SIM_TIMESTEP < 1):
			num_iterations = 1

		"""measurement_data, sensor_data = client.read_data()
		sim_start_stamp = measurement_data.game_timestamp / 1000.0
		# Send a control command to proceed to next iteration.
		# This mainly applies for simulations that are in synchronous mode.
		send_control_command(client, throttle=0.0, steer=0, brake=1.0)
		# Computes the average timestep based on several initial iterations
		sim_duration = 0
		for i in range(num_iterations):
			# Gather current data
			measurement_data, sensor_data = client.read_data()
			# Send a control command to proceed to next iteration
			send_control_command(client, throttle=0.0, steer=0, brake=1.0)
			# Last stamp
			if i == num_iterations - 1:
				sim_duration = measurement_data.game_timestamp / 1000.0 -\
							   sim_start_stamp  
		
		# Outputs average simulation timestep and computes how many frames
		# will elapse before the simulation should end based on various
		# parameters that we set in the beginning.
		SIMULATION_TIME_STEP = sim_duration / float(num_iterations)
		print("SERVER SIMULATION STEP APPROXIMATION: " + \
			  str(SIMULATION_TIME_STEP))
		TOTAL_EPISODE_FRAMES = int((TOTAL_RUN_TIME + WAIT_TIME_BEFORE_START) /\
							   SIMULATION_TIME_STEP) + TOTAL_FRAME_BUFFER"""

		#############################################
		# Frame-by-Frame Iteration and Initialization
		#############################################
		# Store pose history starting from the start position
		process_time = datetime.now() - dt0
		start_time = datetime.now() - dt0
		bboxes_old = {}
		measurement_data = lib_lidar.get_bboxes(world, vehicle_lidar, bboxes_old, process_time)
		print('measurement_data ', measurement_data)
		print('integrasi berhasil: data persepsi diterima')

		start_x, start_y, start_yaw = get_current_pose(measurement_data)
		send_control_command(client, throttle=0.0, steer=0, brake=1.0)
		x_history	 = [start_x]
		y_history	 = [start_y]
		yaw_history   = [start_yaw]
		time_history  = [0]
		speed_history = [0]

		reached_the_end = False
		skip_first_frame = True
		closest_index	= 0  # Index of waypoint that is currently closest to
							  # the car (assumed to be the first index)
		closest_distance = 0  # Closest distance of closest waypoint to car
		for frame in range(TOTAL_EPISODE_FRAMES):
			# Gather current data from the CARLA server
			process_time = datetime.now() - dt0
			# Get timestamp in seconds
			timestamp = (datetime.now() - start_time).total_seconds()
			measurement_data = lib_lidar.get_bboxes(world, vehicle_lidar, bboxes_old, process_time)

			# Update pose, timestamp
			current_x, current_y, current_yaw = \
				get_current_pose(measurement_data)
			current_speed = measurement_data[0]['speed']
			#current_timestamp = float(measurement_data.game_timestamp) / 1000.0
			print('current_x, current_y, current_yaw, speed ', current_x, current_y, current_yaw, current_speed)
			current_timestamp = 3
			# Wait for some initial time before starting the demo
			if current_timestamp <= WAIT_TIME_BEFORE_START:
				send_control_command(client, throttle=0.0, steer=0, brake=1.0)
				continue
			else:
				current_timestamp = current_timestamp - WAIT_TIME_BEFORE_START
			
			# Store history
			x_history.append(current_x)
			y_history.append(current_y)
			yaw_history.append(current_yaw)
			speed_history.append(current_speed)
			time_history.append(current_timestamp) 

			###
			# Controller update (this uses the controller2d.py implementation)
			###

			closest_distance = np.linalg.norm(np.array([
					waypoints_np[closest_index, 0] - current_x,
					waypoints_np[closest_index, 1] - current_y]))
			new_distance = closest_distance
			new_index = closest_index
			while new_distance <= closest_distance:
				closest_distance = new_distance
				closest_index = new_index
				new_index += 1
				if new_index >= waypoints_np.shape[0]:  # End of path
					break
				new_distance = np.linalg.norm(np.array([
						waypoints_np[new_index, 0] - current_x,
						waypoints_np[new_index, 1] - current_y]))
			new_distance = closest_distance
			new_index = closest_index
			while new_distance <= closest_distance:
				closest_distance = new_distance
				closest_index = new_index
				new_index -= 1
				if new_index < 0:  # Beginning of path
					break
				new_distance = np.linalg.norm(np.array([
						waypoints_np[new_index, 0] - current_x,
						waypoints_np[new_index, 1] - current_y]))

			# Once the closest index is found, return the path that has 1
			# waypoint behind and X waypoints ahead, where X is the index
			# that has a lookahead distance specified by 
			# INTERP_LOOKAHEAD_DISTANCE
			waypoint_subset_first_index = closest_index - 1
			if waypoint_subset_first_index < 0:
				waypoint_subset_first_index = 0

			waypoint_subset_last_index = closest_index
			total_distance_ahead = 0
			while total_distance_ahead < INTERP_LOOKAHEAD_DISTANCE:
				total_distance_ahead += wp_distance[waypoint_subset_last_index]
				waypoint_subset_last_index += 1
				if waypoint_subset_last_index >= waypoints_np.shape[0]:
					waypoint_subset_last_index = waypoints_np.shape[0] - 1
					break

			# Use the first and last waypoint subset indices into the hash
			# table to obtain the first and last indicies for the interpolated
			# list. Update the interpolated waypoints to the controller
			# for the next controller update.
			new_waypoints = \
					wp_interp[wp_interp_hash[waypoint_subset_first_index]:\
							  wp_interp_hash[waypoint_subset_last_index] + 1]
			controller.update_waypoints(new_waypoints)

			# Update the other controller values and controls
			controller.update_values(current_x, current_y, current_yaw, 
									 current_speed,
									 current_timestamp, frame)
			controller.update_controls()
			cmd_throttle, cmd_steer, cmd_brake = controller.get_commands()

			# Skip the first frame (so the controller has proper outputs)
			if skip_first_frame and frame == 0:
				pass
			else:
				if frame%6 == 2:
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
				#bboxes_old = {}
				


			# Output controller command to CARLA server
			send_control_command(client,
								 throttle=cmd_throttle,
								 steer=cmd_steer,
								 brake=cmd_brake)

			# Find if reached the end of waypoint. If the car is within
			# DIST_THRESHOLD_TO_LAST_WAYPOINT to the last waypoint,
			# the simulation will end.
			dist_to_last_waypoint = np.linalg.norm(np.array([
				waypoints[-1][0] - current_x,
				waypoints[-1][1] - current_y]))
			if  dist_to_last_waypoint < DIST_THRESHOLD_TO_LAST_WAYPOINT:
				reached_the_end = True
			if reached_the_end:
				break

		# End of demo - Stop vehicle and Store outputs to the controller output
		# directory.
		if reached_the_end:
			print("Reached the end of path. Writing to controller_output...")
		else:
			print("Exceeded assessment time. Writing to controller_output...")
		# Stop the car
		send_control_command(client, throttle=0.0, steer=0.0, brake=1.0)
		# Store the various outputs
		write_trajectory_file(x_history, y_history, speed_history, time_history)

		"""while True:
			if frame%6 == 2:
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
			bboxes_old = {}
				
			process_time = datetime.now() - dt0
				
			#PRINT BBOX ARRAY
			#=====================================================
			exec_waypoint_nav_demo(arg)

			#bboxes = lib_lidar.get_bboxes(world, vehicle_lidar, bboxes_old, process_time)
			#bboxes_old = bboxes
			#print(bboxes)
				
			sys.stdout.write('\r' + 'FPS: ' + str(1.0 / process_time.total_seconds()))
			sys.stdout.flush()
			#sys.stdout.write(bboxes+'\n')
			dt0 = datetime.now()
			frame += 1"""



	world.apply_settings(original_settings)
	traffic_manager.set_synchronous_mode(False)
	vehicle_lidar.destroy()
	lidar.destroy()
	vis.destroy_window()



def main():
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
		exec_waypoint_nav_demo(arg)
		print('Done.')
		return
	#except TCPConnectionError as error:
		#logging.error(error)
		#time.sleep(1)

if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		print(' - Exited by user.')
