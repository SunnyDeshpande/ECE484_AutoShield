#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D
from cv_bridge import CvBridge
import numpy as np
import cv2
import time

# Ultralytics YOLOv11
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

COCO_PERSON_CLASS_ID = 0  # "person"


class YoloV11PersonDetector(Node):
    def __init__(self):
        super().__init__('yolo_v11_person_detector')

        # Params
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('model_path', 'yolo11n.pt')
        self.declare_parameter('conf', 0.35)
        self.declare_parameter('iou', 0.45)
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('half', False)
        self.declare_parameter('max_detections', 100)
        self.declare_parameter('x_vel_buffer_length', 3)
        self.declare_parameter('y_vel_buffer_length', 10)

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.publish_debug = self.get_parameter('publish_debug_image').get_parameter_value().bool_value
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.conf = float(self.get_parameter('conf').value)
        self.iou = float(self.get_parameter('iou').value)
        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.half = bool(self.get_parameter('half').value)
        self.max_det = int(self.get_parameter('max_detections').value)
        self.x_vel_buffer_length = int(self.get_parameter('x_vel_buffer_length').value)
        self.y_vel_buffer_length = int(self.get_parameter('y_vel_buffer_length').value)

        if YOLO is None:
            raise RuntimeError("Ultralytics is not installed. pip install ultralytics")

        self.get_logger().info(f"Loading YOLOv11 model: {model_path}")
        self.model = YOLO(model_path)

        self.model.overrides['conf'] = self.conf
        self.model.overrides['iou'] = self.iou
        self.model.overrides['device'] = self.device
        self.model.overrides['imgsz'] = self.imgsz
        self.model.overrides['half'] = self.half
        self.model.overrides['max_det'] = self.max_det
        self.model.overrides['classes'] = [COCO_PERSON_CLASS_ID]

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, image_topic, self.image_cb, qos)

        self.pub_dets = self.create_publisher(Detection2DArray, 'detections', 10)
        self.pub_debug = self.create_publisher(Image, 'detections/image', 10) if self.publish_debug else None

        self.last_fps_t = time.time()
        self.frame_count = 0

        self.x_vel_buffer = []
        self.y_vel_buffer = []

        self.prev_boxes = []          # (x1, y1, x2, y2)
        self.prev_time = None
        self.prev_corners_vel = None  # list of 4 (vx, vy)

        self.get_logger().info("YOLOv11 person detector ready.")

    @staticmethod
    def _point_in_triangle(px, py, x1, y1, x2, y2, x3, y3):
        denom = ( (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3) )
        if abs(denom) < 1e-6:
            return False
        a = ((y2 - y3)*(px - x3) + (x3 - x2)*(py - y3)) / denom
        b = ((y3 - y1)*(px - x3) + (x1 - x3)*(py - y3)) / denom
        c = 1.0 - a - b
        return (0.0 <= a <= 1.0) and (0.0 <= b <= 1.0) and (0.0 <= c <= 1.0)

    def image_cb(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        h, w = cv_image.shape[:2]
        debug_img = cv_image.copy() if self.publish_debug else None

        # fan geometry (for logic + drawing)
        x_center = int(w / 2)
        x_left_third = int(w / 3)
        x_right_third = int(2 * w / 3)
        x_left_sixth = int(w / 6)
        x_right_sixth = int(5 * w / 6)
        delta = int(w / 6)

        top_center = (x_center, 0)
        top_left_yellow = (x_center - delta, 0)
        top_right_yellow = (x_center + delta, 0)

        bottom_left_red = (x_left_third, h - 1)
        bottom_right_red = (x_right_third, h - 1)
        bottom_left_yellow = (x_left_sixth, h - 1)
        bottom_right_yellow = (x_right_sixth, h - 1)

        now = self.get_clock().now()
        t_now = now.nanoseconds * 1e-9
        dt = None
        if self.prev_time is not None:
            dt = t_now - self.prev_time
            if dt <= 0:
                dt = None

        results = self.model.predict(
            source=cv_image,
            verbose=False,
            stream=False
        )

        det_msg = Detection2DArray()
        det_msg.header = msg.header

        r = results[0]
        curr_boxes = []

        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.detach().cpu().numpy()
            confs = r.boxes.conf.detach().cpu().numpy()
            clss = r.boxes.cls.detach().cpu().numpy().astype(int)

            for i in range(xyxy.shape[0]):
                if clss[i] != COCO_PERSON_CLASS_ID:
                    continue

                x1, y1, x2, y2 = xyxy[i]
                score = float(confs[i])

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                bw = (x2 - x1)
                bh = (y2 - y1)

                vx = 0.0
                vy = 0.0
                ax = 0.0
                ay = 0.0

                vel_candidates_scaled = None

                if dt is not None and len(self.prev_boxes) > 0:
                    px_cx_cy = []
                    for (px1, py1, px2, py2) in self.prev_boxes:
                        pcx = (px1 + px2) / 2.0
                        pcy = (py1 + py2) / 2.0
                        px_cx_cy.append((pcx, pcy))

                    dists = [
                        (pcx - cx) ** 2 + (pcy - cy) ** 2
                        for (pcx, pcy) in px_cx_cy
                    ]
                    j = int(np.argmin(dists))
                    px1, py1, px2, py2 = self.prev_boxes[j]

                    prev_corners = [
                        (px1, py1),
                        (px2, py1),
                        (px1, py2),
                        (px2, py2),
                    ]
                    curr_corners = [
                        (x1, y1),
                        (x2, y1),
                        (x1, y2),
                        (x2, y2),
                    ]

                    vel_candidates_px = []
                    for (cx_prev, cy_prev), (cx_curr, cy_curr) in zip(prev_corners, curr_corners):
                        dx = cx_curr - cx_prev
                        dy = cy_curr - cy_prev
                        vx_c = dx / dt
                        vy_c = dy / dt
                        vel_candidates_px.append((vx_c, vy_c))

                    person_height_m = 1.7
                    bbox_height_px = max(bh, 1.0)
                    meters_per_pixel = person_height_m / bbox_height_px

                    vel_candidates_scaled = [
                        (vx_c * meters_per_pixel, vy_c * meters_per_pixel)
                        for (vx_c, vy_c) in vel_candidates_px
                    ]

                    speeds = [
                        vx_s ** 2 + vy_s ** 2
                        for (vx_s, vy_s) in vel_candidates_scaled
                    ]
                    min_v_idx = int(np.argmin(speeds))
                    vx, vy = vel_candidates_scaled[min_v_idx]

                    if (
                        self.prev_corners_vel is not None
                        and len(self.prev_corners_vel) == 4
                    ):
                        accel_speeds = []
                        accel_candidates = []
                        for (vx_curr, vy_curr), (vx_prev, vy_prev) in zip(
                            vel_candidates_scaled, self.prev_corners_vel
                        ):
                            ax_c = (vx_curr - vx_prev) / dt
                            ay_c = (vy_curr - vy_prev) / dt
                            a_speed = ax_c ** 2 + ay_c ** 2
                            accel_speeds.append(a_speed)
                            accel_candidates.append((ax_c, ay_c))

                        min_a_idx = int(np.argmin(accel_speeds))
                        ax, ay = accel_candidates[min_a_idx]

                if vel_candidates_scaled is not None:
                    self.prev_corners_vel = vel_candidates_scaled
                else:
                    self.prev_corners_vel = None

                curr_boxes.append((x1, y1, x2, y2))

                self.x_vel_buffer.append(vx)
                self.y_vel_buffer.append(vy)
                if len(self.x_vel_buffer) > self.x_vel_buffer_length:
                    self.x_vel_buffer.pop(0)
                if len(self.y_vel_buffer) > self.y_vel_buffer_length:
                    self.y_vel_buffer.pop(0)

                det = Detection2D()
                det.header = msg.header

                det.bbox = BoundingBox2D()
                det.bbox.center.position.x = float(cx)
                det.bbox.center.position.y = float(cy)
                det.bbox.size_x = float(bw)
                det.bbox.size_y = float(bh)

                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = "person"
                hyp.hypothesis.score = float(score)
                det.results.append(hyp)

                det_msg.detections.append(det)

                if debug_img is not None:
                    p1 = (int(x1), int(y1))
                    p2 = (int(x2), int(y2))

                    color = (0, 255, 0)
                    crossing = False

                    if abs(ax) < 30.0:
                        in_left_fan = self._point_in_triangle(
                            cx, cy,
                            bottom_left_red[0], bottom_left_red[1],
                            top_center[0], top_center[1],
                            top_left_yellow[0], top_left_yellow[1]
                        )
                        in_right_fan = self._point_in_triangle(
                            cx, cy,
                            bottom_right_red[0], bottom_right_red[1],
                            top_center[0], top_center[1],
                            top_right_yellow[0], top_right_yellow[1]
                        )

                        if (in_left_fan or in_right_fan) and abs(vy/(vx+0.001)) < 0.5:
                            crossing = True
                            color = (0, 0, 255)       # red
                        else:
                            color = (0, 165, 255)     # orange

                    cv2.rectangle(debug_img, p1, p2, color, 2)

                    label = (f"person {score:.2f}, "
                             f"{'CROSSING' if crossing else 'CAUTION'}")
                    cv2.putText(
                        debug_img,
                        label,
                        (p1[0], max(0, p1[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                    info_text = (
                        f"vel_x/vel_y ratio={abs(vy/(vx+0.001)):.1f},"
                        f"img_x_vel={vx:.1f}, img_y_vel={vy:.1f}, "
                        f"img_x_accel={ax:.1f}, img_y_accel={ay:.1f}"
                    )
                    text_org = (p1[0], min(h - 5, p2[1] + 20))
                    cv2.putText(
                        debug_img,
                        info_text,
                        text_org,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                    if crossing:
                        print(f"Detection: vx={vx:.2f}, vy={vy:.2f}, ax={ax:.2f}, ay={ay:.2f}, CROSSING")
                    else:
                        print(f"Detection: vx={vx:.2f}, vy={vy:.2f}, ax={ax:.2f}, ay={ay:.2f}, CAUTION")

        self.prev_boxes = curr_boxes
        self.prev_time = t_now

        if debug_img is not None:
            cv2.line(debug_img, bottom_left_red, top_center, (0, 0, 255), 2)
            cv2.line(debug_img, bottom_right_red, top_center, (0, 0, 255), 2)

            cv2.line(debug_img, bottom_left_yellow, top_left_yellow, (0, 255, 255), 2)
            cv2.line(debug_img, bottom_right_yellow, top_right_yellow, (0, 255, 255), 2)

        self.pub_dets.publish(det_msg)

        if debug_img is not None and self.pub_debug is not None:
            out_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
            out_msg.header = msg.header
            self.pub_debug.publish(out_msg)

        self.frame_count += 1
        if self.frame_count % 30 == 0:
            now_t = time.time()
            fps = 30.0 / (now_t - self.last_fps_t + 1e-9)
            self.last_fps_t = now_t
            self.get_logger().info(f"~{fps:.1f} FPS")


def main():
    rclpy.init()
    node = YoloV11PersonDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
