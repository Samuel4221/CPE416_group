"""
@file ekf_node.py
@brief Extended Kalman Filter (EKF) fusion of IMU + LiDAR odometry for Project 2.

This node fuses:
  - IMU yaw rate (sensor_msgs/msg/Imu) for prediction
  - LiDAR odometry (nav_msgs/msg/Odometry) for correction

State vector:
  x = [ x, y, theta ]^T

Subscriptions:
  - /oak/camera/imu_data   (sensor_msgs/msg/Imu)
  - /kiss-icp/odom         (nav_msgs/msg/Odometry)

Publications:
  - /ekf_odom              (nav_msgs/msg/Odometry)
  - /tf                    (optional) odom_ekf -> base_link
"""

import math
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Quaternion, Vector3
import tf2_ros


class EkfNode(Node):
    """
    @class EkfNode
    @brief EKF-based pose estimator fusing IMU yaw rate and LiDAR odometry.

    The node mirrors the structure of the Roomba controller:
      - Publishers created in the constructor
      - Subscribers created in the constructor
      - A periodic timer_callback() that runs the main loop

    Predict step:
      - Uses IMU angular velocity around z (yaw rate) to integrate heading.

    Update step:
      - Uses LiDAR odometry (x, y, theta) as a measurement to correct the state.

    @publisher /ekf_odom
      nav_msgs/msg/Odometry containing the fused [x, y, theta] pose.

    @publisher /tf
      geometry_msgs/msg/TransformStamped from frame "odom_ekf" to "base_link".

    @subscriber /oak/camera/imu_data
      sensor_msgs/msg/Imu from the OAK camera IMU.

    @subscriber /kiss-icp/odom
      nav_msgs/msg/Odometry from LiDAR odometry (e.g., KISS-ICP).
    """

    def __init__(self):
        """
        @brief Construct a new EkfNode.

        Initializes publishers, subscribers, EKF state and covariance, and
        the periodic timer used to drive predict/update.
        """
        super().__init__('ekf_node')

        # --- Publishers ---
        self.ekf_pub = self.create_publisher(Odometry, '/ekf_odom', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # --- Subscribers ---
        self.imu_sub = self.create_subscription(
            Imu,
            '/oakd/imu/data',
            self.imu_callback,
            50
        )

        self.lidar_sub = self.create_subscription(
            Odometry,
            '/kiss/odometry',
            self.lidar_callback,
            10
        )
        # --- EKF state: x = [x, y, theta]^T ---
        self.state = np.zeros((3, 1), dtype=float)
        self.P = np.eye(3, dtype=float) * 1e-3  # small initial covariance

        # Process noise (Q) and measurement noise (R) parameters
        # Tune these values for your system.
        self.qx = 1e-6     # process noise in x
        self.qy = 1e-6     # process noise in y
        self.qtheta = 1e-4 # process noise in theta

        self.rx = 1e-2     # measurement noise in x
        self.ry = 1e-2     # measurement noise in y
        self.rtheta = 1e-1 # measurement noise in theta

        # IMU message queue and last IMU time for dt computation
        self.imu_queue = deque()
        self.last_imu_time = None  # seconds (float)

        # Latest LiDAR odometry message (for update)
        self.latest_lidar_odom = None

        # Timer period in seconds (e.g. 50 ms -> 20 Hz)
        self.period = 0.05
        self.timer = self.create_timer(self.period, self.timer_callback)

        self.get_logger().info("EKF node initialized.")

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def imu_callback(self, msg: Imu):
        """
        @brief IMU callback that enqueues incoming IMU messages.

        This callback must NOT run EKF math directly; it just stores
        the data so that the timer loop can process it deterministically.

        @param msg Incoming sensor_msgs/msg/Imu message.
        """
        self.imu_queue.append(msg)

    def lidar_callback(self, msg: Odometry):
        """
        @brief LiDAR odometry callback that stores the latest odom message.

        The LiDAR odometry provides a noisy measurement of the robot pose.
        We keep only the most recent message to use in the update step.

        @param msg Incoming nav_msgs/msg/Odometry message.
        """
        self.latest_lidar_odom = msg

    def timer_callback(self):
        """
        @brief Periodic EKF loop: predict using IMU, then update using LiDAR.

        This function is called at a fixed rate (self.period). Each tick:
          1. Drains the IMU queue and runs predict() for each message.
          2. Runs update_from_lidar() if a LiDAR odometry message is available.
          3. Publishes the fused Odometry and TF.
        """
        # 1) PREDICT: integrate all IMU messages since last tick
        while self.imu_queue:
            imu_msg = self.imu_queue.popleft()
            self.predict(imu_msg)

        # 2) UPDATE: use the latest LiDAR odometry measurement
        if self.latest_lidar_odom is not None:
            self.update_from_lidar(self.latest_lidar_odom)

        # 3) Publish fused odometry and TF
        self.publish_ekf_odom()

    # -------------------------------------------------------------------------
    # EKF math
    # -------------------------------------------------------------------------

    def predict(self, imu_msg: Imu):
        """
        @brief EKF prediction step using IMU yaw rate.

        The motion model here is very simple: we only integrate the yaw angle
        using angular_velocity.z from the IMU. x and y are left unchanged
        and will be corrected by LiDAR odometry.

        State update:
            theta_new = theta + omega_z * dt

        Covariance update:
            P = F * P * F^T + Q   with F ~ I (identity) in this simple model.

        @param imu_msg IMU message containing angular velocity.
        """
        t = imu_msg.header.stamp
        current_time = float(t.sec) + float(t.nanosec) * 1e-9

        if self.last_imu_time is None:
            # First IMU message, just store the time
            self.last_imu_time = current_time
            return

        dt = current_time - self.last_imu_time
        self.last_imu_time = current_time

        if dt <= 0.0:
            return  # ignore out-of-order or identical timestamps

        # Extract yaw rate (rad/s)
        omega_z = imu_msg.angular_velocity.z  # z-axis angular velocity

        # Current state
        x, y, theta = self.state.flatten()

        # Integrate heading
        theta_new = self._wrap_angle(theta + omega_z * dt)

        # Update state (x, y unchanged in this minimal model)
        self.state = np.array([[x],
                           [y],
                           [theta_new]], dtype=float)

        # State transition Jacobian (approximately identity here)
        F = np.eye(3, dtype=float)

        # Process noise covariance
        Q = np.diag([self.qx, self.qy, self.qtheta])

        # Covariance prediction
        self.P = F @ self.P @ F.T + Q

    def update_from_lidar(self, odom_msg: Odometry):
        """
        @brief EKF update step using LiDAR odometry.

        The LiDAR odometry message is treated as a direct noisy measurement of
        [x, y, theta]. The measurement model is:
            z = h(x) = x
        so H = I (identity).

        @param odom_msg LiDAR odometry message (nav_msgs/msg/Odometry).
        """
        # Extract position
        z_x = odom_msg.pose.pose.position.x
        z_y = odom_msg.pose.pose.position.y

        # Extract yaw from quaternion
        q = odom_msg.pose.pose.orientation
        z_theta = self._yaw_from_quaternion(q)

        z = np.array([[z_x],
                      [z_y],
                      [z_theta]], dtype=float)

        # Measurement model: h(x) = x (identity)
        H = np.eye(3, dtype=float)

        # Measurement noise covariance
        R = np.diag([self.rx, self.ry, self.rtheta])

        # Innovation y = z - h(x)
        y = z - self.state
        # Wrap yaw innovation into [-pi, pi]
        y[2, 0] = self._wrap_angle(y[2, 0])

        # Innovation covariance S = HPH^T + R
        S = H @ self.P @ H.T + R

        # Kalman gain K = P H^T S^-1
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        self.state = self.state + K @ y

        # Covariance update: P = (I - K H) P
        I = np.eye(3, dtype=float)
        self.P = (I - K @ H) @ self.P

    # -------------------------------------------------------------------------
    # Publishing / helpers
    # -------------------------------------------------------------------------

    def publish_ekf_odom(self):
        """
        @brief Publish fused EKF odometry and TF transform.

        Publishes the current state as nav_msgs/msg/Odometry on /ekf_odom and
        broadcasts a transform from odom_ekf to base_link.
        """
        odom = Odometry()
        now = self.get_clock().now().to_msg()

        # Header
        odom.header.stamp = now
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame

        # Pose
        x, y, theta = self.state.flatten()
        odom.pose.pose.position.x = float(x)
        odom.pose.pose.position.y = float(y)
        odom.pose.pose.position.z = 0.0

        q = self._quaternion_from_yaw(theta)
        odom.pose.pose.orientation = q

        # Pack 3x3 covariance into a 6x6 pose covariance matrix
        cov = [0.0] * 36
        cov[0]  = float(self.P[0, 0])  # xx
        cov[1]  = float(self.P[0, 1])  # xy
        cov[5]  = float(self.P[0, 2])  # x-yaw
        cov[6]  = float(self.P[1, 0])  # yx
        cov[7]  = float(self.P[1, 1])  # yy
        cov[11] = float(self.P[1, 2])  # y-yaw
        cov[30] = float(self.P[2, 0])  # yaw-x
        cov[31] = float(self.P[2, 1])  # yaw-y
        cov[35] = float(self.P[2, 2])  # yaw-yaw
        odom.pose.covariance = cov

        self.ekf_pub.publish(odom)

        # TF transform
        tf_msg = TransformStamped()
        tf_msg.header.stamp = now
        tf_msg.header.frame_id = self.odom_frame
        tf_msg.child_frame_id = self.base_frame
        tf_msg.transform.translation.x = float(x)
        tf_msg.transform.translation.y = float(y)
        tf_msg.transform.translation.z = 0.0
        tf_msg.transform.rotation = q

        self.tf_broadcaster.sendTransform(tf_msg)

    def _wrap_angle(angle: float) -> float:
        """
        @brief Wrap an angle to the range [-pi, pi].

        @param angle Input angle (radians).
        @return Wrapped angle (radians).
        """
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def _quaternion_from_yaw(yaw: float) -> Quaternion:
        """
        @brief Convert a yaw angle to a Quaternion (z-rotation only).

        @param yaw Yaw angle (radians).
        @return Quaternion representing rotation about Z by yaw.
        """
        q = Quaternion()
        q.w = math.cos(yaw * 0.5)
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw * 0.5)
        return q

    def _yaw_from_quaternion(q: Quaternion) -> float:
        """
        @brief Extract yaw angle from a Quaternion.

        Assumes a standard ROS ENU convention and that roll and pitch
        are small / not used for this 2D ground robot.

        @param q Input Quaternion.
        @return Yaw angle (radians).
        """
        # yaw (z-axis rotation) from quaternion
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)


def main(args=None):
    """
    @brief Main entry point: initialize ROS 2 and spin the EKF node.
    """
    rclpy.init(args=args)
    node = EkfNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
