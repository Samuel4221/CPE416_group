from dataclasses import dataclass
from typing import Dict, List, Tuple

import math

from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan


@dataclass
class RobotState:
    x: float
    y: float
    yaw: float
    vx: float
    vtheta: float


class DWAPlanner:
    def __init__(self, params: Dict):
        self.params = params

    # ------------- Public API -------------

    def compute_cmd(self,
                    state: RobotState,
                    goal: PoseStamped,
                    scan: LaserScan) -> Twist:
        """
        Main entry point: given robot state, goal, and sensor data,
        compute the best (v, w) command using DWA.
        """
        # 1. Build dynamic window
        v_min, v_max, w_min, w_max = self.compute_dynamic_window(state)

        # 2. Sample velocity space
        samples = self.sample_velocities(v_min, v_max, w_min, w_max)

        # 3. Convert LaserScan to simple obstacle representation
        obstacles = self.scan_to_points(scan)

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

    def compute_dynamic_window(self, state: RobotState) -> Tuple[float, float, float, float]:
        """
        Compute the (v, w) ranges allowed over the next time step,
        based on current velocity and acceleration limits.
        """
        max_v = self.params['max_vel_x']
        min_v = self.params['min_vel_x']
        max_w = self.params['max_vel_theta']
        min_w = self.params['min_vel_theta']

        acc_v = self.params['acc_lim_x']
        acc_w = self.params['acc_lim_theta']
        dt = self.params['sim_dt']

        # velocity limits due to acceleration
        v_min = max(min_v, state.vx - acc_v * dt)
        v_max = min(max_v, state.vx + acc_v * dt)

        w_min = max(min_w, state.vtheta - acc_w * dt)
        w_max = min(max_w, state.vtheta + acc_w * dt)

        return v_min, v_max, w_min, w_max

    def sample_velocities(self,
                          v_min: float, v_max: float,
                          w_min: float, w_max: float) -> List[Tuple[float, float]]:
        """
        Uniformly sample (v, w) in the dynamic window.
        """
        vx_samples = int(self.params['vx_samples'])
        w_samples = int(self.params['vtheta_samples'])

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
                           state: RobotState,
                           v: float, w: float) -> List[Tuple[float, float, float]]:
        """
        Simulate robot motion under (v, w) for sim_time seconds.
        Returns list of (x, y, yaw).
        """
        sim_time = self.params['sim_time']
        dt = self.params['sim_dt']

        x = state.x
        y = state.y
        yaw = state.yaw

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
                            goal: PoseStamped,
                            obstacles: List[Tuple[float, float]],
                            v: float,
                            w: float) -> float:
        """
        Combine multiple cost terms into a single score.
        Higher score is better.
        """
        if not traj:
            return -float('inf')

        heading_w = self.params['heading_weight']
        obstacle_w = self.params['obstacle_weight']
        velocity_w = self.params['velocity_weight']

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
                     goal: PoseStamped) -> float:
        """
        Reward trajectories that end close to the goal.
        """
        gx = goal.pose.position.x
        gy = goal.pose.position.y
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

    # ------------- LaserScan → obstacle list -------------

    def scan_to_points(self, scan: LaserScan) -> List[Tuple[float, float]]:
        """
        Convert LaserScan into simple obstacle points in the robot frame.
        For now, we just generate (x, y) in base_link.
        """
        angles = []
        a = scan.angle_min
        while a <= scan.angle_max + 1e-6:
            angles.append(a)
            a += scan.angle_increment

        points: List[Tuple[float, float]] = []
        for r, theta in zip(scan.ranges, angles):
            if math.isinf(r) or math.isnan(r):
                continue
            # simple polar → Cartesian in robot frame
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            points.append((x, y))

        return points
