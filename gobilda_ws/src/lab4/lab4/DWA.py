#!/usr/bin/env python3
import math
from typing import Optional, List, Tuple
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry, OccupancyGrid


class DwaLocalPlannerNode(Node):
    def __init__(self):
        super().__init__('dwa_local_planner')

        # --- ROS interfaces ---
        # Publishers 
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        # Subscribers
        self.lidar_sub = self.create_subscription(
            Odometry,
            '/kiss/odometry',
            self.lidar_callback,
            10
        )
        self.localcostmap_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.localcostmap_callback,
            10
        )
        #--- instance variables ---
        #--- Hard coded Goal State ---
        self.goal_state = (2, 0, 0, 0, 0)  # x, y, yaw, vx, vtheta
        self.current_state = (0,0,0,0,0)   # x, y, yaw, vx, vtheta
        #--- Subscribers messages---
        self.latest_local_costmap = None
        self.latest_odometry = None
        #--- DWA physical limitations ---
        #assume robot limits are the same regardless of their sign
        self.vel_lim_x = 0.5
        self.vel_lim_theta = 1.5
        self.acc_lim_x = 0.5
        self.acc_lim_theta = 1.0
        #--- DWA simulation parameters ---
        self.sim_time = 2.0 # seconds
        self.sim_dt = 0.1 # seconds
        self.vx_samples = 10
        self.vtheta_samples = 20
        #--- DWA weights ---
        self.heading_weight = 1.0
        self.obstacle_weight = 1.5
        self.velocity_weight = 0.5

        # control loop at 10 Hz
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('DWA local planner node started')

    # ----------------- Callbacks -----------------

    def lidar_callback(self, msg: Odometry):
        """
        @brief kiss/odom callback that stores the latest lidar odometry messages.

        The odometry provides the current estimated pose and velocity of the robot.
        We keep only the most recent message to use in the planning step.
        
        @param msg Incoming nav_msgs/msg/Odometry message.
        """
        self.latest_odometry = msg
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # yaw from quaternion
        q = msg.pose.pose.orientation
        yaw = self.yaw_from_quat(q.x, q.y, q.z, q.w)

        vx = msg.twist.twist.linear.x
        vtheta = msg.twist.twist.angular.z

        self.current_state = (x, y, yaw, vx, vtheta)

    def local_costmap_callback(self, msg: OccupancyGrid):
        """
        @brief Local costmap callback that stores the latest costmap message.

        The local costmap provides a representation of the environment around the robot.
        We keep only the most recent message to use in the planning step.

        @param msg Incoming nav_msgs/msg/OccupancyGrid message.
        """
        self.latest_local_costmap = msg

    # ----------------- Main control loop -----------------

    def control_loop(self):
        if self.latest_odometry is None:
            return
        if self.latest_local_costmap is None:
            return

        cmd = self.compute_next_velo(
            state=self.current_state,
            goal=self.goal_state,
            costmap=self.latest_local_costmap
        )

        self.cmd_pub.publish(cmd)

    # ----------------- Helpers -----------------

    @staticmethod
    def yaw_from_quat(x, y, z, w) -> float:
        # standard yaw extraction
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)
    #function to compute next velocity
    def compute_next_velo(self,
                    state: Tuple[float, float, float, float, float],
                    goal: Tuple[float, float, float, float, float],
                    costmap: OccupancyGrid) -> Twist:
        """
        Main entry point: given robot state, goal, and costmap,
        compute the best (v, w) command using DWA.
        """
        # 1. Build dynamic window
        v_min, v_max, w_min, w_max = self.compute_dynamic_window(state)

        # 2. Sample velocity space
        samples = self.sample_velocities(v_min, v_max, w_min, w_max)

        # 3. Convert OccupancyGrid to obstacle points
        obstacles = self.costmap_to_points(costmap)

        best_score = -float('inf')
        best_cmd = Twist()

        for v, w in samples:
            # 4. Forward simulate this (v, w)
            traj = self.rollout_trajectory(state, v, w)

            # 5. Score this trajectory
            score = self.evaluate_trajectory(traj, goal, obstacles, v, w)

            if score > best_score:
                best_score = score
                best_cmd.linear.x = v
                best_cmd.angular.z = w

        return best_cmd

    # ------------- Core pieces -------------
    #compute dynamic window
    def compute_dynamic_window(self, state: Tuple[float, float, float, float, float]) -> Tuple[float, float, float, float]:
        """
        Compute the (v, w) ranges allowed over the next time step,
        based on current velocity and acceleration limits.
        """
        x, y, yaw, vx, vtheta = state
        
        max_v = self.vel_lim_x
        min_v = -self.vel_lim_x
        max_w = self.vel_lim_theta
        min_w = -self.vel_lim_theta

        acc_v = self.acc_lim_x
        acc_w = self.acc_lim_theta
        dt = self.sim_dt

        # velocity limits due to acceleration
        v_min = max(min_v, vx - acc_v * dt)
        v_max = min(max_v, vx + acc_v * dt)

        w_min = max(min_w, vtheta - acc_w * dt)
        w_max = min(max_w, vtheta + acc_w * dt)

        return v_min, v_max, w_min, w_max

    def sample_velocities(self,
                          v_min: float, v_max: float,
                          w_min: float, w_max: float) -> List[Tuple[float, float]]:
        """
        Uniformly sample (v, w) in the dynamic window.
        """
        vx_samples = int(self.vx_samples)
        w_samples = int(self.vtheta_samples)

        vs = []
        if vx_samples <= 1:
            vs = [(v_min + v_max) / 2.0]
        else:
            dv = (v_max - v_min) / max(vx_samples - 1, 1)
            vs = [v_min + i * dv for i in range(vx_samples)]

        ws = []
        if w_samples <= 1:
            ws = [(w_min + w_max) / 2.0]
        else:
            dw = (w_max - w_min) / max(w_samples - 1, 1)
            ws = [w_min + i * dw for i in range(w_samples)]

        samples = []
        for v in vs:
            for w in ws:
                samples.append((v, w))
        return samples

    def rollout_trajectory(self,
                           state: Tuple[float, float, float, float, float],
                           v: float, w: float) -> List[Tuple[float, float, float]]:
        """
        Simulate robot motion under (v, w) for sim_time seconds.
        Returns list of (x, y, yaw).
        """
        sim_time = self.sim_time
        dt = self.sim_dt

        x, y, yaw, vx, vtheta = state

        traj = []
        t = 0.0
        while t < sim_time:
            x += v * math.cos(yaw) * dt
            y += v * math.sin(yaw) * dt
            yaw += w * dt
            traj.append((x, y, yaw))
            t += dt

        return traj

    # ------------- Cost evaluation -------------

    def evaluate_trajectory(self,
                            traj: List[Tuple[float, float, float]],
                            goal: Tuple[float, float, float, float, float],
                            obstacles: List[Tuple[float, float]],
                            v: float,
                            w: float) -> float:
        """
        Combine multiple cost terms into a single score.
        Higher score is better.
        """
        if not traj:
            return -float('inf')

        heading_w = self.heading_weight
        obstacle_w = self.obstacle_weight
        velocity_w = self.velocity_weight

        heading_cost = self.heading_cost(traj, goal)
        obstacle_cost = self.obstacle_cost(traj, obstacles)
        velocity_cost = self.velocity_cost(v)

        # NOTE: for DWA it's common to *maximize* score, so we treat
        # good heading and velocity as positive, and obstacles as negative.
        score = (
            heading_w * heading_cost +
            obstacle_w * obstacle_cost +
            velocity_w * velocity_cost
        )
        return score

    def heading_cost(self,
                     traj: List[Tuple[float, float, float]],
                     goal: Tuple[float, float, float, float, float]) -> float:
        """
        Reward trajectories that end close to the goal.
        """
        gx, gy, _, _, _ = goal
        x_end, y_end, _ = traj[-1]

        dist = math.hypot(gx - x_end, gy - y_end)
        # invert so closer => larger cost
        return -dist

    def obstacle_cost(self,
                      traj: List[Tuple[float, float, float]],
                      obstacles: List[Tuple[float, float]]) -> float:
        """
        Penalize trajectories that come close to obstacles.
        You can refine this a lot later.
        """
        if not obstacles:
            return 0.0

        min_dist = float('inf')
        for x, y, _ in traj:
            for ox, oy in obstacles:
                d = math.hypot(ox - x, oy - y)
                if d < min_dist:
                    min_dist = d

        # If collision (too close), invalidate trajectory
        if min_dist < 0.1:  # TODO: tune collision radius
            return -float('inf')

        # Otherwise, reward being farther from obstacles
        return min_dist

    def velocity_cost(self, v: float) -> float:
        """
        Prefer higher forward velocity (you can adjust this).
        """
        return v

    # ------------- OccupancyGrid → obstacle list -------------

    def costmap_to_points(self, costmap: OccupancyGrid) -> List[Tuple[float, float]]:
        """
        Convert OccupancyGrid into obstacle points in the robot frame.
        Extract occupied cells from the costmap.
        """
        points: List[Tuple[float, float]] = []
        
        width = costmap.info.width
        height = costmap.info.height
        resolution = costmap.info.resolution
        origin_x = costmap.info.origin.position.x
        origin_y = costmap.info.origin.position.y
        
        # Iterate through the costmap and find occupied cells
        for i, value in enumerate(costmap.data):
            if value > 50:  # Occupied threshold
                # Convert index to x, y coordinates
                grid_x = i % width
                grid_y = i // width
                
                # Convert to world coordinates
                x = origin_x + (grid_x + 0.5) * resolution
                y = origin_y + (grid_y + 0.5) * resolution
                points.append((x, y))
        
        return points



def main(args=None):
    rclpy.init(args=args)
    node = DwaLocalPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    

if __name__ == '__main__':
    main()
