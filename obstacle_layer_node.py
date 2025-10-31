import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose
import numpy as np
import math


class ObstacleLayerNode(Node):
    def __init__(self):
        super().__init__('obstacle_layer_node')
        
        # Map parameters
        self.map_width = 15.0  # meters
        self.map_height = 15.0  # meters
        self.resolution = 0.05  # meters per cell (5cm)
        
        # Calculate grid dimensions
        self.grid_width = int(self.map_width / self.resolution)
        self.grid_height = int(self.map_height / self.resolution)
        
        # Initialize occupancy grid (-1 = unknown, 0 = free, 100 = occupied)
        self.grid = np.full(self.grid_width * self.grid_height, -1, dtype=np.int8)
        
        # Robot is at the center of the map
        self.robot_x = self.map_width / 2.0
        self.robot_y = self.map_height / 2.0
        
        # Create publisher for occupancy grid
        self.map_publisher = self.create_publisher(OccupancyGrid, '/map', 10)
        
        # Subscribe to laser scan data
        self.scan_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        
        self.get_logger().info('Obstacle Layer Node started')
        self.get_logger().info(f'Map size: {self.map_width}x{self.map_height}m')
        self.get_logger().info(f'Grid size: {self.grid_width}x{self.grid_height} cells')
        self.get_logger().info(f'Resolution: {self.resolution}m/cell')
    
    def world_to_grid(self, x, y):
        """Convert world coordinates (meters) to grid coordinates (cells)"""
        grid_x = int(x / self.resolution)
        grid_y = int(y / self.resolution)
        return grid_x, grid_y
    
    def grid_to_index(self, grid_x, grid_y):
        """Convert grid coordinates to 1D array index"""
        if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
            return grid_y * self.grid_width + grid_x
        return None
    
    def bresenham_line(self, x0, y0, x1, y1):
        """
        Bresenham's line algorithm to get all cells along a line
        Returns a list of (x, y) grid coordinates
        """
        cells = []
        
        # Calculate direction and error term
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        error = dx + dy
        
        x, y = x0, y0
        
        while True:
            cells.append((x, y))
            
            # Check if we've reached the endpoint
            if x == x1 and y == y1:
                break
            
            # Calculate error term for next step
            e2 = 2 * error
            
            # Decide whether to step in x direction
            if e2 >= dy:
                if x == x1:
                    break
                error = error + dy
                x = x + sx
            
            # Decide whether to step in y direction
            if e2 <= dx:
                if y == y1:
                    break
                error = error + dx
                y = y + sy
        
        return cells
    
    def update_grid_with_scan(self, scan_msg):
        """Update the occupancy grid based on laser scan data"""
        angle = scan_msg.angle_min
        
        for i, range_val in enumerate(scan_msg.ranges):
            # Skip invalid readings
            if math.isnan(range_val) or math.isinf(range_val):
                angle += scan_msg.angle_increment
                continue
            
            if range_val < scan_msg.range_min or range_val > scan_msg.range_max:
                angle += scan_msg.angle_increment
                continue
            
            # Calculate endpoint in world coordinates
            endpoint_x = self.robot_x + range_val * math.cos(angle)
            endpoint_y = self.robot_y + range_val * math.sin(angle)
            
            # Convert to grid coordinates
            robot_grid_x, robot_grid_y = self.world_to_grid(self.robot_x, self.robot_y)
            endpoint_grid_x, endpoint_grid_y = self.world_to_grid(endpoint_x, endpoint_y)
            
            # Get all cells along the ray using Bresenham's algorithm
            ray_cells = self.bresenham_line(robot_grid_x, robot_grid_y, 
                                           endpoint_grid_x, endpoint_grid_y)
            
            # Update cells along the ray
            for j, (grid_x, grid_y) in enumerate(ray_cells):
                index = self.grid_to_index(grid_x, grid_y)
                
                if index is not None:
                    # Mark cells along the ray as free space (except the last one)
                    if j < len(ray_cells) - 1:
                        self.grid[index] = 0  # Free space
                    else:
                        # Mark the endpoint as occupied
                        self.grid[index] = 100  # Occupied space
            
            angle += scan_msg.angle_increment
    
    def publish_map(self):
        """Publish the occupancy grid"""
        grid_msg = OccupancyGrid()
        
        # Set header
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = 'map'
        
        # Set map metadata
        grid_msg.info = MapMetaData()
        grid_msg.info.resolution = self.resolution
        grid_msg.info.width = self.grid_width
        grid_msg.info.height = self.grid_height
        
        # Set origin (robot at center means origin is at bottom-left)
        grid_msg.info.origin = Pose()
        grid_msg.info.origin.position.x = 0.0
        grid_msg.info.origin.position.y = 0.0
        grid_msg.info.origin.position.z = 0.0
        grid_msg.info.origin.orientation.w = 1.0
        
        # Set grid data
        grid_msg.data = self.grid.tolist()
        
        # Publish
        self.map_publisher.publish(grid_msg)
    
    def scan_callback(self, msg):
        """Callback function for laser scan messages"""
        # Reset grid to unknown for fresh update
        # (For a simple implementation, you might want to keep previous data)
        # self.grid.fill(-1)
        
        # Update grid with new scan data
        self.update_grid_with_scan(msg)
        
        # Publish the updated map
        self.publish_map()


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleLayerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()