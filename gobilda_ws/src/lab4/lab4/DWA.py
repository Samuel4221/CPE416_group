#!/usr/bin/env python3
import math
from typing import Optional

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from .DWA_FUNCTIONS import DWAPlanner, RobotState


class DwaLocalPlannerNode(Node):
    def __init__(self):
        super().__init__('dwa_local_planner')

        # --- Parameters ---
        self.declare_parameter('max_vel_x', 0.5)
        self.declare_parameter('min_vel_x', 0.0)
        self.declare_parameter('max_vel_theta', 1.5)
        self.declare_parameter('min_vel_theta', -1.5)
        self.declare_parameter('acc_lim_x', 0.5)
        self.declare_parameter('acc_lim_theta', 1.0)

        self.declare_parameter('sim_time', 2.0)
        self.declare_parameter('sim_dt', 0.1)
        self.declare_parameter('vx_samples', 10)
        self.declare_parameter('vtheta_samples', 20)

        self.declare_parameter('heading_weight', 1.0)
        self.declare_parameter('obstacle_weight', 1.5)
        self.declare_parameter('velocity_weight', 0.5)

        params = {
            'max_vel_x': self.get_parameter('max_vel_x').value,
            'min_vel_x': self.get_parameter('min_vel_x').value,
            'max_vel_theta': self.get_parameter('max_vel_theta').value,
            'min_vel_theta': self.get_parameter('min_vel_theta').value,
            'acc_lim_x': self.get_parameter('acc_lim_x').value,
            'acc_lim_theta': self.get_parameter('acc_lim_theta').value,
            'sim_time': self.get_parameter('sim_time').value,
            'sim_dt': self.get_parameter('sim_dt').value,
            'vx_samples': self.get_parameter('vx_samples').value,
            'vtheta_samples': self.get_parameter('vtheta_samples').value,
            'heading_weight': self.get_parameter('heading_weight').value,
            'obstacle_weight': self.get_parameter('obstacle_weight').value,
            'velocity_weight': self.get_parameter('velocity_weight').value,
        }

        # --- Internal state ---
        self.current_state: Optional[RobotState] = None
        self.current_goal: Optional[PoseStamped] = None
        self.current_scan: Optional[LaserScan] = None

        # --- Planner core ---
        self.planner = DWAPlanner(params)

        # --- ROS interfaces ---
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_cb, 10)

        # control loop at 10 Hz
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('DWA local planner node started')

    # ----------------- Callbacks -----------------

    def odom_cb(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # yaw from quaternion
        q = msg.pose.pose.orientation
        yaw = self.yaw_from_quat(q.x, q.y, q.z, q.w)

        vx = msg.twist.twist.linear.x
        vtheta = msg.twist.twist.angular.z

        self.current_state = RobotState(x=x, y=y, yaw=yaw,
                                        vx=vx, vtheta=vtheta)

    def scan_cb(self, msg: LaserScan):
        self.current_scan = msg

    def goal_cb(self, msg: PoseStamped):
        self.current_goal = msg

    # ----------------- Main control loop -----------------

    def control_loop(self):
        if self.current_state is None:
            return
        if self.current_goal is None:
            return
        if self.current_scan is None:
            return

        cmd = self.planner.compute_cmd(
            state=self.current_state,
            goal=self.current_goal,
            scan=self.current_scan
        )

        self.cmd_pub.publish(cmd)

    # ----------------- Helpers -----------------

    @staticmethod
    def yaw_from_quat(x, y, z, w) -> float:
        # standard yaw extraction
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)


def main(args=None):
    rclpy.init(args=args)
    node = DwaLocalPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    

if __name__ == '__main__':
    main()
