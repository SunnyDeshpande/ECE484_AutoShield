import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.qos import QoSProfile, ReliabilityPolicy

class ImageConverter(Node):
    def __init__(self):
        super().__init__('corner_cameras_image_converter')

        self.bridge = CvBridge()

        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # Subscribers for each camera
        self.fl_image_sub = Subscriber(self, Image, "/lucid_vision/camera_fl/image", qos_profile=qos_profile)
        self.fr_image_sub = Subscriber(self, Image, "/lucid_vision/camera_fr/image", qos_profile=qos_profile)
        self.rl_image_sub = Subscriber(self, Image, "/lucid_vision/camera_rl/image", qos_profile=qos_profile)
        self.rr_image_sub = Subscriber(self, Image, "/lucid_vision/camera_rr/image", qos_profile=qos_profile)

        # Time synchronizer
        self.ts = ApproximateTimeSynchronizer(
            [self.fl_image_sub, self.fr_image_sub, self.rl_image_sub, self.rr_image_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.image_callback)

        # Publisher for combined image
        self.corner_image_pub = self.create_publisher(Image, "/camera_corners/combined_image", 10)

    def image_callback(self, fl_image, fr_image, rl_image, rr_image):
        try:
            fl_frame = self.bridge.imgmsg_to_cv2(fl_image, desired_encoding="passthrough")
            fr_frame = self.bridge.imgmsg_to_cv2(fr_image, desired_encoding="passthrough")
            rl_frame = self.bridge.imgmsg_to_cv2(rl_image, desired_encoding="passthrough")
            rr_frame = self.bridge.imgmsg_to_cv2(rr_image, desired_encoding="passthrough")

            # Resize frames to match dimensions if needed
            height = min(fl_frame.shape[0], fr_frame.shape[0], rl_frame.shape[0], rr_frame.shape[0])
            width = min(fl_frame.shape[1], fr_frame.shape[1], rl_frame.shape[1], rr_frame.shape[1])

            fl_frame = cv2.resize(fl_frame, (width, height))
            fr_frame = cv2.resize(fr_frame, (width, height))
            rl_frame = cv2.resize(rl_frame, (width, height))
            rr_frame = cv2.resize(rr_frame, (width, height))

            # Combine horizontally and vertically
            f_frame = np.concatenate((fl_frame, fr_frame), axis=1)
            r_frame = np.concatenate((rl_frame, rr_frame), axis=1)
            frame = np.concatenate((f_frame, r_frame), axis=0)

            self.corner_image_pub.publish(self.bridge.cv2_to_imgmsg(frame, encoding="rgb8"))

        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')
        except Exception as e:
            self.get_logger().error(f'Unhandled exception: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = ImageConverter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

