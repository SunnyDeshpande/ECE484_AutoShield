import os
import csv
import math
import numpy as np
from numpy import linalg as la
import scipy.signal as signal
import pymap3d as pm
import pygame
import cvxpy as cp

import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool
from pacmod2_msgs.msg import PositionWithSpeed, VehicleSpeedRpt, GlobalCmd, SystemCmdFloat, SystemCmdInt
from sensor_msgs.msg import NavSatFix
from septentrio_gnss_driver.msg import INSNavGeod

# Initialize pygame for joystick
pygame.init()
pygame.joystick.init()
if pygame.joystick.get_count() == 0:
    raise RuntimeError("No joystick connected")
joystick = pygame.joystick.Joystick(0)
joystick.init()


class OnlineFilter:
    """
    Butterworth low-pass filter for smoothing noisy sensor data.
    Reduces high-frequency noise in speed measurements.
    """
    def __init__(self, cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        self.b, self.a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        self.z = signal.lfilter_zi(self.b, self.a)

    def get_data(self, data):
        filted, self.z = signal.lfilter(self.b, self.a, [data], zi=self.z)
        return filted[0]


class MPCController(Node):
    """
    Model Predictive Control (MPC) based path tracking controller.
    Uses optimization to predict future vehicle states and compute optimal control inputs.
    """
    def __init__(self):
        super().__init__('mpc_controller_node')
        
        # Declare parameters with default values
        self.declare_parameter('rate_hz', 20)
        self.declare_parameter('wheelbase', 2.57)
        self.declare_parameter('offset', 1.26)
        self.declare_parameter('origin_lat', 40.0927422)
        self.declare_parameter('origin_lon', -88.2359639)
        self.declare_parameter('desired_speed', 2.0)
        self.declare_parameter('max_acceleration', 0.5)
        
        # MPC-specific parameters
        self.declare_parameter('mpc/horizon', 10)  # Prediction horizon (number of steps)
        self.declare_parameter('mpc/dt', 0.1)      # Time step for prediction (seconds)
        self.declare_parameter('mpc/q_x', 10.0)    # Weight for x position error
        self.declare_parameter('mpc/q_y', 10.0)    # Weight for y position error
        self.declare_parameter('mpc/q_yaw', 1.0)   # Weight for heading error
        self.declare_parameter('mpc/q_v', 1.0)     # Weight for velocity error
        self.declare_parameter('mpc/r_accel', 0.1) # Weight for acceleration effort
        self.declare_parameter('mpc/r_steer', 0.1) # Weight for steering effort
        
        self.declare_parameter('filter/cutoff', 1.2)
        self.declare_parameter('filter/fs', 30)
        self.declare_parameter('filter/order', 4)
        self.declare_parameter('vehicle_name', "")
        
        vehicle_name = self.get_parameter('vehicle_name').value
        if vehicle_name == "":
            self.get_logger().warn("No vehicle_name parameter found. Using default parameters.")
        else:
            self.get_logger().info(f"Using vehicle config: {vehicle_name}")

        # Load parameters
        self.rate_hz = self.get_parameter('rate_hz').value
        self.wheelbase = self.get_parameter('wheelbase').value
        self.offset = self.get_parameter('offset').value
        self.olat = self.get_parameter('origin_lat').value
        self.olon = self.get_parameter('origin_lon').value
        self.desired_speed = min(5.0, self.get_parameter('desired_speed').value)
        self.max_accel = min(2.0, self.get_parameter('max_acceleration').value)
        
        # MPC parameters
        self.N = self.get_parameter('mpc/horizon').value  # Prediction horizon
        self.dt = self.get_parameter('mpc/dt').value      # Time step
        self.Q = np.diag([
            self.get_parameter('mpc/q_x').value,
            self.get_parameter('mpc/q_y').value,
            self.get_parameter('mpc/q_yaw').value,
            self.get_parameter('mpc/q_v').value
        ])  # State error weight matrix
        self.R = np.diag([
            self.get_parameter('mpc/r_accel').value,
            self.get_parameter('mpc/r_steer').value
        ])  # Control effort weight matrix
        
        self.speed_filter = OnlineFilter(
            cutoff=self.get_parameter('filter/cutoff').value,
            fs=self.get_parameter('filter/fs').value,
            order=self.get_parameter('filter/order').value
        )

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

        # Commands
        self.global_cmd = GlobalCmd(enable=False, clear_override=True)
        self.gear_cmd = SystemCmdInt(command=2)  # NEUTRAL
        self.brake_cmd = SystemCmdFloat(command=0.0)
        self.accel_cmd = SystemCmdFloat(command=0.0)
        self.turn_cmd = SystemCmdInt(command=1)  # no signal
        self.steer_cmd = PositionWithSpeed(angular_position=0.0, angular_velocity_limit=4.0)

        self.read_waypoints()

        # Initialize vehicle state
        self.lat = 0.0
        self.lon = 0.0
        self.heading = 0.0
        self.speed = 0.0
        self.pacmod_enable = False
        
        # MPC warm-start variables
        self.last_u = None  # Store last optimal control sequence

        self.dist_arr = np.zeros(len(self.path_points_lon_x))
        self.timer = self.create_timer(1.0 / self.rate_hz, self.control_loop)

    def gnss_callback(self, msg):
        """Receive GPS position updates"""
        self.lat = msg.latitude
        self.lon = msg.longitude

    def ins_callback(self, msg):
        """Receive heading/orientation updates from INS"""
        self.heading = msg.heading

    def speed_callback(self, msg):
        """Receive vehicle speed and apply low-pass filter"""
        self.speed = self.speed_filter.get_data(msg.vehicle_speed)

    def enable_callback(self, msg):
        """Monitor PACMod enable status"""
        self.pacmod_enable = msg.data

    def read_waypoints(self):
        """Load waypoints from CSV file"""
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, '../waypoints/track.csv')
        with open(filename) as f:
            path_points = [tuple(line) for line in csv.reader(f)]
        self.path_points_lon_x = [float(p[0]) for p in path_points]
        self.path_points_lat_y = [float(p[1]) for p in path_points]
        self.path_points_heading = [float(p[2]) for p in path_points]
        self.wp_size = len(self.path_points_lon_x)

    def heading_to_yaw(self, heading):
        """Convert compass heading (0-360°, North=0) to yaw angle (radians, East=0)"""
        return np.radians(90 - heading) if heading < 270 else np.radians(450 - heading)

    def wps_to_local_xy(self, lon, lat):
        """Convert GPS coordinates to local ENU (East-North-Up) coordinates"""
        x, y, _ = pm.geodetic2enu(lat, lon, 0, self.olat, self.olon, 0)
        return x, y

    def dist(self, p1, p2):
        """Euclidean distance between two points"""
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def front2steer(self, f_angle):
        """
        Convert front wheel angle to steering wheel angle.
        Uses quadratic calibration curve for the vehicle.
        """
        f_angle = max(min(f_angle, 35), -35)
        angle = abs(f_angle)
        steer_angle = -0.1084 * angle ** 2 + 21.775 * angle
        return round(steer_angle if f_angle >= 0 else -steer_angle, 2)

    def check_joystick_enable(self):
        """
        Check joystick buttons for enable/disable commands.
        Returns: 1 (enable), 0 (disable), 2 (no change)
        """
        pygame.event.pump()
        try:
            lb = joystick.get_button(6)  # Left bumper
            rb = joystick.get_button(7)  # Right bumper
        except pygame.error:
            self.get_logger().warn("Joystick read failed")
            return 2
        if lb and rb:
            return 1  # Enable
        elif lb and not rb:
            return 0  # Disable
        return 2  # No change

    def get_gem_state(self):
        """
        Get current vehicle state in local coordinates.
        Accounts for GPS sensor offset from vehicle center.
        Returns: (x, y, yaw) in local frame
        """
        local_x, local_y = self.wps_to_local_xy(self.lon, self.lat)
        yaw = self.heading_to_yaw(self.heading)
        # Adjust for sensor offset
        x = local_x - self.offset * math.cos(yaw)
        y = local_y - self.offset * math.sin(yaw)
        return x, y, yaw

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def get_reference_trajectory(self, start_idx):
        """
        Extract reference trajectory for MPC horizon.
        Returns: numpy array of shape (N, 4) with [x, y, yaw, v] for each step
        """
        ref_traj = []
        for i in range(self.N):
            idx = min(start_idx + i, self.wp_size - 1)
            ref_traj.append([
                self.path_points_lon_x[idx],
                self.path_points_lat_y[idx],
                self.normalize_angle(np.radians(self.path_points_heading[idx])),
                self.desired_speed
            ])
        return np.array(ref_traj)

    def solve_mpc(self, curr_state, ref_trajectory):
        """
        Solve MPC optimization problem.
        
        Args:
            curr_state: Current vehicle state [x, y, yaw, v]
            ref_trajectory: Reference trajectory (N x 4)
        
        Returns:
            (acceleration, steering_angle) or None if solve fails
        """
        # Decision variables
        x = cp.Variable((4, self.N + 1))  # State: [x, y, yaw, v]
        u = cp.Variable((2, self.N))      # Control: [acceleration, steering_angle]
        
        # Cost function
        cost = 0
        constraints = [x[:, 0] == curr_state]  # Initial condition
        
        for k in range(self.N):
            # State tracking cost: penalize deviation from reference
            state_error = x[:, k] - ref_trajectory[k]
            # Normalize yaw error to [-pi, pi]
            state_error[2] = cp.sin(state_error[2])  # Approximate angle normalization
            cost += cp.quad_form(state_error, self.Q)
            
            # Control effort cost: penalize aggressive control
            cost += cp.quad_form(u[:, k], self.R)
            
            # Vehicle dynamics (kinematic bicycle model)
            # x_{k+1} = x_k + v_k * cos(yaw_k) * dt
            # y_{k+1} = y_k + v_k * sin(yaw_k) * dt
            # yaw_{k+1} = yaw_k + (v_k / L) * tan(delta_k) * dt
            # v_{k+1} = v_k + a_k * dt
            
            # Linearized dynamics for convex optimization
            yaw_k = curr_state[2] if k == 0 else ref_trajectory[k-1][2]
            v_k = curr_state[3] if k == 0 else ref_trajectory[k-1][3]
            
            constraints += [
                x[0, k+1] == x[0, k] + v_k * math.cos(yaw_k) * self.dt,
                x[1, k+1] == x[1, k] + v_k * math.sin(yaw_k) * self.dt,
                x[2, k+1] == x[2, k] + (v_k / self.wheelbase) * u[1, k] * self.dt,
                x[3, k+1] == x[3, k] + u[0, k] * self.dt
            ]
            
            # Control constraints
            constraints += [
                u[0, k] >= -self.max_accel,           # Min acceleration
                u[0, k] <= self.max_accel,            # Max acceleration
                u[1, k] >= -np.radians(35),           # Min steering angle
                u[1, k] <= np.radians(35),            # Max steering angle
                x[3, k+1] >= 0.0,                     # Min speed
                x[3, k+1] <= self.desired_speed * 1.5 # Max speed
            ]
        
        # Terminal cost: penalize final state error
        final_error = x[:, self.N] - ref_trajectory[-1]
        final_error[2] = cp.sin(final_error[2])
        cost += cp.quad_form(final_error, self.Q * 2)
        
        # Solve optimization problem
        problem = cp.Problem(cp.Minimize(cost), constraints)
        
        try:
            # Use warm-start if available
            if self.last_u is not None:
                u.value = self.last_u
            
            problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                # Extract first control input (receding horizon principle)
                accel_cmd = u[0, 0].value
                steer_cmd = u[1, 0].value
                
                # Store for warm-start next iteration
                self.last_u = u.value
                
                return accel_cmd, steer_cmd
            else:
                self.get_logger().warn(f"MPC solve failed: {problem.status}")
                return None, None
                
        except Exception as e:
            self.get_logger().error(f"MPC solver error: {e}")
            return None, None

    def control_loop(self):
        """Main control loop called at rate_hz frequency"""
        joy_enable = self.check_joystick_enable()

        # Handle enable request
        if joy_enable == 1 and not self.pacmod_enable:
            self.global_cmd.enable = True
            self.global_cmd.clear_override = True
            self.global_pub.publish(self.global_cmd)
            
            self.gear_cmd.command = 3  # FORWARD
            self.gear_pub.publish(self.gear_cmd)
            
            self.brake_cmd.command = 0.0
            self.brake_pub.publish(self.brake_cmd)

            self.accel_cmd.command = 0.0
            self.accel_pub.publish(self.accel_cmd)

            self.turn_cmd.command = 3  # LEFT signal
            self.turn_pub.publish(self.turn_cmd)
            
            self.get_logger().warn('Vehicle enabled and forward gear engaged')

        # Handle disable request
        elif joy_enable == 0 and self.pacmod_enable:
            self.global_cmd.enable = False
            self.global_pub.publish(self.global_cmd)

            self.turn_cmd.command = 1  # No signal
            self.turn_pub.publish(self.turn_cmd)
            self.get_logger().warn('Vehicle disabled')

        # Execute MPC controller
        elif joy_enable != 0 and self.pacmod_enable:
            self.path_points_x = np.array(self.path_points_lon_x)
            self.path_points_y = np.array(self.path_points_lat_y)

            curr_x, curr_y, curr_yaw = self.get_gem_state()
            curr_state = np.array([curr_x, curr_y, curr_yaw, self.speed])
            
            # Find closest waypoint ahead of vehicle
            min_dist = float('inf')
            self.goal = 0
            for i in range(self.wp_size):
                self.dist_arr[i] = self.dist((self.path_points_x[i], self.path_points_y[i]), (curr_x, curr_y))
                dx = self.path_points_x[i] - curr_x
                dy = self.path_points_y[i] - curr_y
                angle_to_wp = math.atan2(dy, dx)
                angle_diff = self.normalize_angle(angle_to_wp - curr_yaw)
                
                # Only consider waypoints ahead (within 90 degrees)
                if abs(angle_diff) < math.pi/2 and self.dist_arr[i] < min_dist:
                    min_dist = self.dist_arr[i]
                    self.goal = i

            # Get reference trajectory for MPC
            ref_trajectory = self.get_reference_trajectory(self.goal)
            
            # Solve MPC optimization
            accel_cmd, steer_angle = self.solve_mpc(curr_state, ref_trajectory)
            
            if accel_cmd is not None and steer_angle is not None:
                # Convert steering angle to steering wheel angle
                steering_wheel_angle = self.front2steer(math.degrees(steer_angle))
                
                # Publish steering command
                self.steer_cmd.angular_position = math.radians(steering_wheel_angle)
                self.steer_pub.publish(self.steer_cmd)
                
                # Publish throttle/brake commands
                if accel_cmd >= 0:
                    self.accel_cmd.command = min(accel_cmd, self.max_accel)
                    self.brake_cmd.command = 0.0
                else:
                    self.accel_cmd.command = 0.0
                    self.brake_cmd.command = min(-accel_cmd, 1.0)
                
                self.accel_pub.publish(self.accel_cmd)
                self.brake_pub.publish(self.brake_cmd)
                
                self.global_cmd.enable = True
                self.global_pub.publish(self.global_cmd)
                
                self.get_logger().info(
                    f"MPC - Goal: {self.goal}/{self.wp_size}, "
                    f"Pos: ({curr_x:.2f}, {curr_y:.2f}), "
                    f"Target: ({ref_trajectory[0][0]:.2f}, {ref_trajectory[0][1]:.2f}), "
                    f"Speed: {self.speed:.2f}, "
                    f"Accel: {accel_cmd:.3f}, "
                    f"Steer: {steering_wheel_angle:.2f}°"
                )
            else:
                # MPC failed, apply safe defaults
                self.brake_cmd.command = 0.3
                self.accel_cmd.command = 0.0
                self.brake_pub.publish(self.brake_cmd)
                self.accel_pub.publish(self.accel_cmd)
                self.get_logger().error("MPC solve failed, applying light brake")


def main(args=None):
    rclpy.init(args=args)
    mpc_controller = MPCController()
    rclpy.spin(mpc_controller)
    mpc_controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()