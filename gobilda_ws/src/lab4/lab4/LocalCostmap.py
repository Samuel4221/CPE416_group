#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
# Needed imports
from math import cos, sin, isfinite
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, Point, Quaternion

class LocalCostmap(Node):
    def __init__(self):
        super().__init__('local_costmap')
        
        # Map config
        self.map_width = 300            # cells
        self.map_height = 300           # cells
        self.map_resolution = 0.05      # resolution * cells  => 15 m x 15 m map
        
        # Occupancy conventions
        self.UNKNOWN = -1
        self.FREE = 0
        self.OCCUPIED = 100
        
        # Outgoing OccupancyGrid message (we'll fill .info and .header)
        self.publish_map = OccupancyGrid()
        self._init_map_info()
        
        # Precompute robot's cell (center of grid)
        self.cx = self.map_width // 2
        self.cy = self.map_height // 2
        
        # Store the most recent laser scan
        self.latest_scan = None
        
        # Create publisher for occupancy grid
        self.publisher_ = self.create_publisher(OccupancyGrid, '/map', 10)
        
        # Subscribe to laser scan
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10
        )
        
        # Timer to periodically publish the map
        self.timer = self.create_timer(0.1, self.build_occupancy_grid)  # 10 Hz
        
        self.get_logger().info('Local Costmap Node Started')
        self.get_logger().info(f'Map: {self.map_width}x{self.map_height} cells = {self.map_width*self.map_resolution}x{self.map_height*self.map_resolution}m')

    ''' Initialize static OccupancyGrid.info and origin so the robot is at the map center. '''
    def _init_map_info(self):
        self.publish_map.info.resolution = self.map_resolution
        self.publish_map.info.width = self.map_width
        self.publish_map.info.height = self.map_height
        
        # Place (0,0) of the grid so that the robot (base frame origin) is at the center cell.
        # That means the map origin (bottom-left corner in world coords) is shifted negative by half-size.
        origin = Pose()
        origin.position.x = - (self.map_width * self.map_resolution) / 2.0
        origin.position.y = - (self.map_height * self.map_resolution) / 2.0
        origin.position.z = 0.0
        origin.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        self.publish_map.info.origin = origin

    ''' Meters in robot frame -> map indices (mx, my). Returns None if out of bounds. '''
    # Input: (x, y) coordinates of a point in the Cartesian plane
    # Output: Corresponding cell in the occupancy grid
    def world_to_map(self, x_m, y_m):
        # Convert world coordinates (meters) to map cell coordinates
        # Robot is at center, so we need to offset by the center position
        mx = int(x_m / self.map_resolution) + self.cx
        my = int(y_m / self.map_resolution) + self.cy
        
        # Check bounds
        if 0 <= mx < self.map_width and 0 <= my < self.map_height:
            return mx, my
        else:
            return None

    # Bresenham's Line Algorithm: inclusive endpoints
    # Input: 2-points on the Cartesian plane (i.e. a line)
    # (The first point is the robot origin, while the second is a single beam's endpoint)
    # Output: All the cells that the beam crosses. i.e. the free cells.
    def bresenham_line_algorithm(self, x0, y0, x1, y1):
        """
        Bresenham's line algorithm to trace all cells from (x0,y0) to (x1,y1)
        Returns list of (x, y) tuples representing cell coordinates
        """
        cells = []
        
        # Calculate deltas
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        
        # Determine direction of steps
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        
        # Initialize error
        err = dx - dy
        
        x, y = x0, y0
        
        while True:
            # Add current cell to list
            cells.append((x, y))
            
            # Check if we've reached the end
            if x == x1 and y == y1:
                break
            
            # Calculate error for next step
            e2 = 2 * err
            
            # Step in x direction
            if e2 > -dy:
                err -= dy
                x += sx
            
            # Step in y direction
            if e2 < dx:
                err += dx
                y += sy
        
        return cells
    
    ''' Cache the most recent LaserScan'''
    def laser_callback(self, msg: LaserScan):
        self.latest_scan = msg

    # Input: x & y coordinates (endpoint of laser beam in meters);
    # Output: list of free cells along the ray (excludes the last cell)
    def raytrace(self, x_m, y_m):
        """
        Compute free cells for a single beam from robot center to endpoint
        This function should call self.bresenham_line_algorithm
        """
        # Convert endpoint from meters to map cells
        endpoint = self.world_to_map(x_m, y_m)
        
        if endpoint is None:
            return []
        
        end_x, end_y = endpoint
        
        # Get all cells along the ray using Bresenham
        all_cells = self.bresenham_line_algorithm(self.cx, self.cy, end_x, end_y)
        
        # Return all cells except the last one (the last one is the occupied cell)
        free_cells = all_cells[:-1] if len(all_cells) > 1 else []
        
        return free_cells, endpoint

    ''' Build and Publish the Occupancy Grid from the most recent LiDAR Scan '''
    def build_occupancy_grid(self):
        # First, check that the scan data is ready
        if self.latest_scan is None:
            return
        
        scan = self.latest_scan
        
        # Initialize the grid with UNKNOWN values
        grid_data = [self.UNKNOWN] * (self.map_width * self.map_height)
        
        # Second, iterate through beams to create the map!
        angle = scan.angle_min
        
        for i, range_val in enumerate(scan.ranges):
            # Skip invalid readings
            if not isfinite(range_val):
                angle += scan.angle_increment
                continue
            
            if range_val < scan.range_min or range_val > scan.range_max:
                angle += scan.angle_increment
                continue
            
            # Calculate endpoint in meters (robot frame)
            x_m = range_val * cos(angle)
            y_m = range_val * sin(angle)
            
            # Get free cells and endpoint
            result = self.raytrace(x_m, y_m)
            if not result:
                angle += scan.angle_increment
                continue
                
            free_cells, endpoint = result
            
            # Mark free cells
            for (cell_x, cell_y) in free_cells:
                # Convert 2D cell coords to 1D array index (row-major order)
                index = cell_y * self.map_width + cell_x
                if 0 <= index < len(grid_data):
                    grid_data[index] = self.FREE
            
            # Mark occupied cell (endpoint)
            if endpoint is not None:
                end_x, end_y = endpoint
                end_index = end_y * self.map_width + end_x
                if 0 <= end_index < len(grid_data):
                    grid_data[end_index] = self.OCCUPIED
            
            angle += scan.angle_increment
        
        # Populate OccupancyGrid message
        self.publish_map.data = grid_data
        self.publish_map.header.stamp = self.get_clock().now().to_msg()
        
        # Set frame to match your robot frame that LaserScan is in (commonly "base_link" or "laser")
        # The simulation and hardware will have different names for this frame
        # You may need to change this based on your robot's configuration
        self.publish_map.header.frame_id = 'diff_drive/lidar_link'
        
        # Publish
        self.publisher_.publish(self.publish_map)

def main(args=None):
    rclpy.init(args=args)
    
    # Node creation and spin
    local_costmap = LocalCostmap()
    rclpy.spin(local_costmap)
    
    # Node cleanup
    local_costmap.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()