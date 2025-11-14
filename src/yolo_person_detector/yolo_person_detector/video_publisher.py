#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class VideoPublisher(Node):
    def __init__(self):
        super().__init__('video_publisher')

        # Declare parameters
        self.declare_parameter('video_path', 'video.mp4')
        self.declare_parameter('loop', False)
        # 0.0 = no limit; >0 = limit to first N seconds
        self.declare_parameter('max_seconds', 0.0)

        self.video_path = self.get_parameter('video_path').get_parameter_value().string_value
        self.loop = self.get_parameter('loop').get_parameter_value().bool_value
        self.max_seconds = float(self.get_parameter('max_seconds').value)

        # Open video file
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            self.get_logger().error(f'Failed to open video file: {self.video_path}')
            raise RuntimeError('Could not open video file')

        # Get FPS from video, fall back to 30.0 if unknown
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
            self.get_logger().warn('FPS not found in video, defaulting to 30 FPS')

        self.fps = fps
        self.frame_period = 1.0 / fps

        # Max frames to publish from the start (None = unlimited)
        if self.max_seconds > 0.0:
            self.max_frames = int(self.max_seconds * self.fps + 0.5)
            self.get_logger().info(
                f'Will use only the first {self.max_seconds:.2f}s '
                f'(~{self.max_frames} frames) of the video '
                f'(loop_segment={self.loop})'
            )
        else:
            self.max_frames = None

        self.published_frames = 0

        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Image, '/camera/color/image_raw', 10)

        # Timer to publish frames at video FPS
        self.timer = self.create_timer(self.frame_period, self.timer_callback)

        self.get_logger().info(
            f'Publishing video "{self.video_path}" at {fps:.2f} FPS '
            f'on /camera/color/image_raw (loop={self.loop}, max_seconds={self.max_seconds})'
        )

    def _restart_segment(self):
        """Seek to start, reset frame counter for looping the first N seconds."""
        self.get_logger().info(
            f'Restarting segment: first {self.max_seconds:.2f}s from beginning.'
            if self.max_frames is not None
            else 'Restarting full video from beginning.'
        )
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.published_frames = 0

    def timer_callback(self):
        # Segment limit logic
        if self.max_frames is not None and self.published_frames >= self.max_frames:
            if self.loop:
                # Loop only the first N seconds
                self._restart_segment()
            else:
                # Stop after first N seconds
                self.get_logger().info(
                    f'Reached max duration ({self.max_seconds:.2f}s), '
                    'stopping video publisher.'
                )
                self.timer.cancel()
                rclpy.shutdown()
                return

        if not self.cap.isOpened():
            self.get_logger().error('VideoCapture is not opened anymore.')
            return

        ret, frame = self.cap.read()

        # End-of-file handling (for the "no max_seconds" case or safety)
        if not ret:
            if self.loop:
                # If no max_seconds: loop whole video
                # If max_seconds: we still only replay from the start,
                # and segment-limit logic above will cap it.
                self._restart_segment()
                ret, frame = self.cap.read()
                if not ret:
                    self.get_logger().error('Failed to read frame after restarting. Stopping node.')
                    rclpy.shutdown()
                    return
            else:
                self.get_logger().info('Reached end of video, shutting down.')
                rclpy.shutdown()
                return

        # Convert BGR (OpenCV) to ROS Image
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_color_optical_frame'

        self.publisher.publish(msg)
        self.published_frames += 1


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = VideoPublisher()
        rclpy.spin(node)
    except Exception as e:
        if node is not None:
            node.get_logger().error(f'Exception in VideoPublisher: {e}')
    finally:
        if node is not None and node.cap is not None:
            node.cap.release()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
