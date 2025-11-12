#! /usr/bin/env python3

# Python Headers
import os
import cv2 
import csv
import math
import numpy as np
from numpy import linalg as la

# ROS Headers
import rclpy                                     
from rclpy.node import Node    
from cv_bridge import CvBridge, CvBridgeError       
from ament_index_python.packages import get_package_share_directory           

from std_msgs.msg import String, Float64     
from sensor_msgs.msg import Imu, NavSatFix, Image, CameraInfo           
from sensor_msgs.msg import NavSatFix
from septentrio_gnss_driver.msg import INSNavGeod
from visualization_msgs.msg import Marker

image_path = os.path.join(get_package_share_directory('gem_gnss_image'), 'images', 'gnss_map.png')


class GNSSImage(Node):

    global image_path
    
    def __init__(self, name):

        super().__init__(name)                               

        # Read image in BGR format
        self.map_image = cv2.imread(image_path)

        # Create the cv_bridge object
        self.bridge  = CvBridge()
        self.map_image_pub = self.create_publisher(Image, "motion_image", 1)   
        self.timer = self.create_timer(0.05, self.timer_callback)  

        # Subscribe information from sensors
        self.lat     = 0
        self.lon     = 0
        self.heading = 0
        self.gps_sub= self.create_subscription(NavSatFix, '/navsatfix', self.gps_callback, 1)
        self.ins_sub= self.create_subscription(INSNavGeod, '/insnavgeod', self.ins_callback, 1)

        self.create_subscription(Marker, '/visualization/waypoints', self.waypoints_callback, 10)
        self.create_subscription(Marker, '/visualization/next_waypoint', self.next_waypoint_callback, 10)
        self.waypoints = []
        self.next_waypoint = None

        self.lat_start_bt = 40.092722  # 40.09269  
        self.lon_start_l  = -88.236365 # -88.23628
        self.lat_scale    = 0.00057    # 0.00062  
        self.lon_scale    = 0.00136    # 0.00136   

        self.arrow        = 40 
        self.img_width    = 2107
        self.img_height   = 1313

    def gps_callback(self, msg):
        self.lat = msg.latitude
        self.lon = msg.longitude

    def ins_callback(self, msg):
        self.heading = round(msg.heading, 1) 

    def image_heading(self, lon_x, lat_y, heading):
        
        if(heading >=0 and heading < 90):
            angle  = np.radians(90-heading)
            lon_xd = lon_x + int(self.arrow * np.cos(angle))
            lat_yd = lat_y - int(self.arrow * np.sin(angle))

        elif(heading >= 90 and heading < 180):
            angle  = np.radians(heading-90)
            lon_xd = lon_x + int(self.arrow * np.cos(angle))
            lat_yd = lat_y + int(self.arrow * np.sin(angle))  

        elif(heading >= 180 and heading < 270):
            angle = np.radians(270-heading)
            lon_xd = lon_x - int(self.arrow * np.cos(angle))
            lat_yd = lat_y + int(self.arrow * np.sin(angle))

        else:
            angle = np.radians(heading-270)
            lon_xd = lon_x - int(self.arrow * np.cos(angle))
            lat_yd = lat_y - int(self.arrow * np.sin(angle)) 

        return lon_xd, lat_yd  

    def waypoints_callback(self, msg):
        self.waypoints = msg.points

    def next_waypoint_callback(self, msg):
        self.next_waypoint = msg.pose.position

    def timer_callback(self):    

        lon_x = int(self.img_width*(self.lon-self.lon_start_l)/self.lon_scale)
        lat_y = int(self.img_height-self.img_height*(self.lat-self.lat_start_bt)/self.lat_scale)
        lon_xd, lat_yd = self.image_heading(lon_x, lat_y, self.heading)

        pub_image = np.copy(self.map_image)     
        
        if(lon_x >= 0 and lon_x <= self.img_width and lon_xd >= 0 and lon_xd <= self.img_width and 
           lat_y >= 0 and lat_y <= self.img_height and lat_yd >= 0 and lat_yd <= self.img_height):
            cv2.arrowedLine(pub_image, (lon_x, lat_y), (lon_xd, lat_yd), (0, 0, 255), 2)
            cv2.circle(pub_image, (lon_x, lat_y), 12, (0,0,255), 2)

        # draw waypoints
        for wp in self.waypoints:
            lon_x = int(self.img_width * (wp.x - self.lon_start_l) / self.lon_scale)
            lat_y = int(self.img_height - self.img_height * (wp.y - self.lat_start_bt) / self.lat_scale)
            cv2.circle(pub_image, (lon_x, lat_y), 3, (255, 255, 255), -1)

        # draw next waypoint
        if self.next_waypoint:
            lon_x = int(self.img_width * (self.next_waypoint.x - self.lon_start_l) / self.lon_scale)
            lat_y = int(self.img_height - self.img_height * (self.next_waypoint.y - self.lat_start_bt) / self.lat_scale)
            cv2.circle(pub_image, (lon_x, lat_y), 8, (0, 255, 0), -1)

        try:
            # Convert OpenCV image to ROS image and publish
            self.map_image_pub.publish(self.bridge.cv2_to_imgmsg(pub_image, "bgr8"))
        except CvBridgeError as e:
            self.get_logger().info("CvBridge Error: {0}".format(e))                         
                                                            


def main(args=None):                                 
    rclpy.init(args=args)                           
    node = GNSSImage("gem_gnss_image_node")     
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.shutdown()                        


if __name__ == '__main__':
    main()