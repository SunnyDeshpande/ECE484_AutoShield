#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from sensor_msgs.msg import NavSatFix
from septentrio_gnss_driver.msg import INSNavGeod, PVTGeodetic  # Replace with correct imports
from pacmod2_msgs.msg import VehicleSpeedRpt, SystemRptFloat
import numpy as np

class GEMRvizMarker(Node):
    def __init__(self):
        super().__init__('gem_rviz_marker')
        
        self.publisher = self.create_publisher(Marker, '/gem_rviz_marker', 10)
        self.timer = self.create_timer(0.1, self.publish_text_marker)  # 10 Hz

        # Subscriptions
        self.create_subscription(NavSatFix, '/navsatfix', self.gps_callback, 10)
        self.create_subscription(INSNavGeod, '/insnavgeod', self.ins_callback, 10)
        self.create_subscription(PVTGeodetic, '/pvtgeodetic', self.rtk_callback, 10)
        self.create_subscription(VehicleSpeedRpt, '/pacmod/vehicle_speed_rpt', self.speed_callback, 10)
        self.create_subscription(SystemRptFloat, '/pacmod/steering_rpt', self.steer_callback, 10)

        self.create_subscription(Marker, '/visualization/waypoints', self.waypoints_callback, 10)
        self.create_subscription(Marker, '/visualization/next_waypoint', self.next_waypoint_callback, 10)

        # Data holders
        self.lat = 0.0
        self.lon = 0.0
        self.yaw = 0.0
        self.speed = 0.0
        self.steer = 0.0
        self.rtk_status = "Disabled"

    def gps_callback(self, msg):
        self.lat = round(msg.latitude, 6)
        self.lon = round(msg.longitude, 6)

    def ins_callback(self, msg):
        self.yaw = round(msg.heading, 2)

    def rtk_callback(self, msg):
        if msg.mode == 4:
            self.rtk_status = "Enabled Fixed"
        elif msg.mode == 5:
            self.rtk_status = "Enabled Float"
        else:
            self.rtk_status = "Disabled"

    def speed_callback(self, msg):
        self.speed = round(msg.vehicle_speed, 2)

    def steer_callback(self, msg):
        self.steer = round(np.degrees(msg.output), 1)

    def waypoints_callback(self, msg: Marker):
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        self.publisher.publish(msg)

    def next_waypoint_callback(self, msg: Marker):
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        self.publisher.publish(msg)

    def publish_text_marker(self):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "gem_status"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = 5.0
        marker.pose.position.y = 9.0
        marker.pose.position.z = 5.0
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.4  # Font size
        marker.color.r = 0.2
        marker.color.g = 1.0
        marker.color.b = 0.8
        marker.color.a = 1.0
        marker.lifetime.sec = 1

        marker.text = (
            f"RTK = {self.rtk_status}\n"
            f"Lat = {self.lat}\n"
            f"Lon = {self.lon}\n"
            f"Yaw = {self.yaw}\n"
            f"Speed[m/s] = {self.speed}\n"
            f"Steer[deg] = {self.steer}"
        )

        self.publisher.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = GEMRvizMarker()
    rclpy.spin(node)
    rclpy.shutdown()
