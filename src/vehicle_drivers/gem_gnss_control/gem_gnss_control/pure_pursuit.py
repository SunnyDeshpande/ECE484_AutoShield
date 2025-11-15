    #!/usr/bin/env python3

#================================================================
# File name: pure_pursuit_ros2.py
# Description: GNSS waypoints tracker using PID and pure pursuit in ROS2
# Author: Jiaming Zhang, Hang Cui
# Date: 2025-06-03
#================================================================

import os
import csv
import math
import numpy as np
from numpy import linalg as la
import scipy.signal as signal
import pymap3d as pm
import pygame

import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool
from pacmod2_msgs.msg import PositionWithSpeed, VehicleSpeedRpt, GlobalCmd, SystemCmdFloat, SystemCmdInt
from sensor_msgs.msg import NavSatFix
from septentrio_gnss_driver.msg import INSNavGeod
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

# Initialize pygame for joystick
pygame.init()
pygame.joystick.init()
if pygame.joystick.get_count() == 0:
    raise RuntimeError("No joystick connected")
joystick = pygame.joystick.Joystick(0)
joystick.init()


class PID:
    def __init__(self, kp, ki, kd, wg=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.wg = wg
        self.iterm = 0
        self.last_e = 0
        self.last_t = None

    def reset(self):
        self.iterm = 0
        self.last_e = 0
        self.last_t = None

    def get_control(self, t, e):
        if self.last_t is None:
            dt = 0.0
            de = 0.0
        else:
            dt = t - self.last_t
            de = (e - self.last_e) / dt if dt > 0.0 else 0.0

        self.iterm += e * dt
        if self.wg is not None:
            self.iterm = max(min(self.iterm, self.wg), -self.wg)

        self.last_e = e
        self.last_t = t

        return self.kp * e + self.ki * self.iterm + self.kd * de


class OnlineFilter:
    def __init__(self, cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        self.b, self.a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        self.z = signal.lfilter_zi(self.b, self.a)

    def get_data(self, data):
        filted, self.z = signal.lfilter(self.b, self.a, [data], zi=self.z)
        return filted[0]


class PurePursuit(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')
        # Declare parameters with default values
        self.declare_parameter('rate_hz', 20)
        self.declare_parameter('look_ahead', 5.0)
        self.declare_parameter('wheelbase', 2.57)
        self.declare_parameter('offset', 1.26)
        self.declare_parameter('origin_lat', 40.0927422)
        self.declare_parameter('origin_lon', -88.2359639)
        self.declare_parameter('desired_speed', 2.0)
        self.declare_parameter('max_acceleration', 0.5)

        self.declare_parameter('pid/kp', 0.6)
        self.declare_parameter('pid/ki', 0.0)
        self.declare_parameter('pid/kd', 0.1)
        self.declare_parameter('pid/wg', 10)

        self.declare_parameter('filter/cutoff', 1.2)
        self.declare_parameter('filter/fs', 30)
        self.declare_parameter('filter/order', 4)
        self.declare_parameter('vehicle_name', "")
        
        vehicle_name=self.get_parameter('vehicle_name').value
        if (vehicle_name==""):
            self.get_logger().warn("No vehicle_name parameter found. No config file loaded, defaulting to 'e4' parameters.")
        else:
            self.get_logger().info(f"Using vehicle config: {vehicle_name}")


        self.rate_hz = self.get_parameter('rate_hz').value
        self.look_ahead = self.get_parameter('look_ahead').value
        self.wheelbase = self.get_parameter('wheelbase').value
        self.offset = self.get_parameter('offset').value
        self.olat = self.get_parameter('origin_lat').value
        self.olon = self.get_parameter('origin_lon').value

        self.desired_speed = min(5.0,self.get_parameter('desired_speed').value) # desired speed capped at 5 m/s
        self.max_accel = min(2.0, self.get_parameter('max_acceleration').value) # max acceleration capped at 2 m/s^2
        self.pid_speed = PID(
            kp=self.get_parameter('pid/kp').value,
            ki=self.get_parameter('pid/ki').value,
            kd=self.get_parameter('pid/kd').value,
            wg=self.get_parameter('pid/wg').value
            )
        self.speed_filter = OnlineFilter(
            cutoff=self.get_parameter('filter/cutoff').value,
            fs=self.get_parameter('filter/fs').value,
            order=self.get_parameter('filter/order').value,)

        self.goal = 0

        # Subscriptions
        self.create_subscription(NavSatFix, '/navsatfix', self.gnss_callback, 10)
        self.create_subscription(INSNavGeod, '/insnavgeod', self.ins_callback, 10)
        self.create_subscription(Bool, '/pacmod/enabled', self.enable_callback, 10)
        self.create_subscription(VehicleSpeedRpt, '/pacmod/vehicle_speed_rpt', self.speed_callback, 10)

        # Publishers
        self.global_pub = self.create_publisher(GlobalCmd, '/pacmod/global_cmd', 10)
        self.gear_pub = self.create_publisher(SystemCmdInt, '/pacmod/shift_cmd', 10)
        self.brake_pub = self.create_publisher(SystemCmdFloat, '/pacmod/brake_cmd', 10)
        self.accel_pub = self.create_publisher(SystemCmdFloat, '/pacmod/accel_cmd', 10)
        self.turn_pub = self.create_publisher(SystemCmdInt, '/pacmod/turn_cmd', 10)
        self.steer_pub = self.create_publisher(PositionWithSpeed, '/pacmod/steering_cmd', 10)
        self.waypoints_pub = self.create_publisher(Marker, '/visualization/waypoints', 10)
        self.next_waypoint_pub = self.create_publisher(Marker, '/visualization/next_waypoint', 10)

        # Commands
        self.global_cmd = GlobalCmd(enable=False, clear_override = True)
        self.gear_cmd = SystemCmdInt(command=2)  # NEUTRAL
        self.brake_cmd = SystemCmdFloat(command=0.0)
        self.accel_cmd = SystemCmdFloat(command=0.0)
        self.turn_cmd = SystemCmdInt(command=1) # no signal
        self.steer_cmd = PositionWithSpeed(angular_position=0.0, angular_velocity_limit=4.0)

        self.read_waypoints()

        # Initialize
        self.lat = 0.0
        self.lon = 0.0
        self.heading = 0.0
        self.speed = 0.0
        self.gem_enable = False
        self.pacmod_enable = False

        

        self.dist_arr = np.zeros(len(self.path_points_lon_x))

        self.timer = self.create_timer(1.0 / self.rate_hz, self.control_loop)

    def gnss_callback(self, msg):
        self.lat = msg.latitude
        self.lon = msg.longitude

    def ins_callback(self, msg):
        self.heading = msg.heading

    def speed_callback(self, msg):
        self.speed = self.speed_filter.get_data(msg.vehicle_speed)

    def enable_callback(self, msg):
        self.pacmod_enable = msg.data

    def read_waypoints(self):
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, '../waypoints/track.csv')
        with open(filename) as f:
            path_points = [tuple(line) for line in csv.reader(f)]
        self.path_points_lon_x = [float(p[0]) for p in path_points]
        self.path_points_lat_y = [float(p[1]) for p in path_points]
        self.path_points_heading = [float(p[2]) for p in path_points]
        self.wp_size = len(self.path_points_lon_x)

    def heading_to_yaw(self, heading):
        return np.radians(90 - heading) if heading < 270 else np.radians(450 - heading)

    def wps_to_local_xy(self, lon, lat):
        x, y, _ = pm.geodetic2enu(lat, lon, 0, self.olat, self.olon, 0)
        return x, y

    def dist(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def front2steer(self, f_angle):
        f_angle = max(min(f_angle, 35), -35)
        angle = abs(f_angle)
        steer_angle = -0.1084 * angle ** 2 + 21.775 * angle
        return round(steer_angle if f_angle >= 0 else -steer_angle, 2)

    def check_joystick_enable(self):
        pygame.event.pump()
        try:
            lb = joystick.get_button(6)
            rb = joystick.get_button(7)
        except pygame.error:
            self.get_logger().warn("Joystick read failed")
            return 2
        if lb and rb:
            # enable
            return 1
        elif lb and not rb:
            # disable
            return 0
        # others
        return 2

    def get_gem_state(self):
        local_x, local_y = self.wps_to_local_xy(self.lon, self.lat)
        yaw = self.heading_to_yaw(self.heading)
        x = local_x - self.offset * math.cos(yaw)
        y = local_y - self.offset * math.sin(yaw)
        return x, y, yaw

    def publish_visualization_markers(self, target_x, target_y):
        waypoints_marker = Marker()
        waypoints_marker.header.frame_id = "base_link"
        waypoints_marker.header.stamp = self.get_clock().now().to_msg()
        waypoints_marker.ns = "waypoints"
        waypoints_marker.id = 0
        waypoints_marker.type = Marker.POINTS
        waypoints_marker.action = Marker.ADD
        waypoints_marker.scale.x = 2.0
        waypoints_marker.scale.y = 2.0
        waypoints_marker.color.r = 1.0
        waypoints_marker.color.g = 1.0
        waypoints_marker.color.b = 1.0
        waypoints_marker.color.a = 1.0

        for x, y in zip(self.path_points_x, self.path_points_y):
            p = Point()
            p.x = x
            p.y = y
            waypoints_marker.points.append(p)

        self.waypoints_pub.publish(waypoints_marker)

        next_waypoint_marker = Marker()
        next_waypoint_marker.header.frame_id = "base_link"
        next_waypoint_marker.header.stamp = self.get_clock().now().to_msg()
        next_waypoint_marker.ns = "next_waypoint"
        next_waypoint_marker.id = 1
        next_waypoint_marker.type = Marker.SPHERE
        next_waypoint_marker.action = Marker.ADD
        next_waypoint_marker.pose.position.x = target_x
        next_waypoint_marker.pose.position.y = target_y
        next_waypoint_marker.pose.position.z = 0.0
        next_waypoint_marker.scale.x = 2.0
        next_waypoint_marker.scale.y = 2.0
        next_waypoint_marker.scale.z = 2.0
        next_waypoint_marker.color.r = 0.0
        next_waypoint_marker.color.g = 1.0
        next_waypoint_marker.color.b = 0.0
        next_waypoint_marker.color.a = 1.0

        self.next_waypoint_pub.publish(next_waypoint_marker)

    def control_loop(self):
        joy_enable = self.check_joystick_enable()

        if joy_enable == 1 and not self.pacmod_enable:
            # joystick enable when vehicle disbaled 
            self.global_cmd.enable = True
            self.global_cmd.clear_override = True
            self.global_pub.publish(self.global_cmd)
            
            self.gear_cmd.command = 3
            self.gear_pub.publish(self.gear_cmd)
            
            self.brake_cmd.command = 0.0
            self.brake_pub.publish(self.brake_cmd)

            self.accel_cmd.command = 0.0
            self.accel_pub.publish(self.accel_cmd)

            self.turn_cmd.command = 3
            self.turn_pub.publish(self.turn_cmd)
            
            self.get_logger().warn('Pacmod Disabled: Vehicle enabled and forward gear engaged')

        elif joy_enable == 0 and self.pacmod_enable:
            # joystick disable when vehicle enbaled
            self.global_cmd.enable = False
            self.global_pub.publish(self.global_cmd)

            self.turn_cmd.command = 1
            self.turn_pub.publish(self.turn_cmd)

            self.get_logger().warn('Joystick Disabled: Vehicle disabled')

        elif joy_enable != 0 and self.pacmod_enable:
            # exceuate controller
            self.path_points_x = np.array(self.path_points_lon_x)
            self.path_points_y = np.array(self.path_points_lat_y)

            curr_x, curr_y, curr_yaw = self.get_gem_state()
            
            for i in range(self.wp_size):
                self.dist_arr[i] = self.dist((self.path_points_x[i], self.path_points_y[i]), (curr_x, curr_y))

            self.goal = np.argmin(self.dist_arr)
            ld = self.look_ahead + max(0.0, self.speed - 2.5) * 2
            for i in range(self.goal, self.wp_size):
                if self.dist_arr[i] > ld:
                    self.goal = i
                    break

            target_x = self.path_points_x[self.goal]
            target_y = self.path_points_y[self.goal]

            self.get_logger().info(f"Goal index: {self.goal}, Target Position: ({target_x}, {target_y}), current Position: ({curr_x}, {curr_y}), Look-ahead distance: {ld}")

            target_yaw = self.path_points_heading[self.goal]
            alpha = math.atan2(target_y - curr_y, target_x - curr_x) - curr_yaw
            curvature = 0.0 if self.speed < 0.2 else 2.0 * math.sin(alpha) / ld
            steering_angle = math.atan(self.wheelbase * curvature)
            steering_wheel_angle = self.front2steer(math.degrees(steering_angle))

            self.steer_cmd.angular_position = math.radians(steering_wheel_angle)
            self.steer_pub.publish(self.steer_cmd)

            # Speed control
            now = self.get_clock().now().nanoseconds * 1e-9
            speed_error = self.desired_speed - self.speed
            if abs(speed_error) < 0.05:
                speed_error = 0.0
            throttle_cmd = self.pid_speed.get_control(now, speed_error)
            throttle_cmd = max(0.0, min(throttle_cmd, self.max_accel))

            self.accel_cmd.command = throttle_cmd
            self.brake_cmd.command = 0.0
            self.accel_pub.publish(self.accel_cmd)
            self.brake_pub.publish(self.brake_cmd)

            self.global_cmd.enable = True
            self.global_pub.publish(self.global_cmd)

            self.publish_visualization_markers(target_x, target_y)
            self.get_logger().info(f"Pos: ({curr_x:.2f}, {curr_y:.2f}), Target: ({target_x:.2f}, {target_y:.2f}), Speed: {self.speed:.2f}, Throttle: {throttle_cmd:.2f}, Steering: {steering_wheel_angle:.2f}")
            

def main(args=None):
    rclpy.init(args=args)
    pure_pursuit = PurePursuit()
    rclpy.spin(pure_pursuit)
    pure_pursuit.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
