import rclpy
from rclpy.node import Node
import math

from std_msgs.msg import Float32, Bool
from sensor_msgs.msg import LaserScan


# Object orientated nodes for ROS in python
class ClearanceCalculator(Node): 

    def __init__(self):
        # Init the node with a name (this is the name that appears when running)
        # 'ros2 node list'
        super().__init__('clearance_calculator')
        #First we want to subscribe to the LIDAR data from the robot
        self.lidar_subscriber_ = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.check_forward_sub = self.create_subscription(Bool, '/gobilda/check_forward', self.check_callback, 10)
        # Create a publisher object
        # Arguments include the type of message, the name of the topic,
        # and the queue size
        self.publisher_ = self.create_publisher(Float32, '/gobilda/min_clearance', 10)
        #Next initialize fields needed for the node
        self.check_forward = True  # default to checking forward view for obstacles
        self.min_clearance = Float32() # closest obstacle distance in robots current trajectory
        self.min_clearance.data = 10.0 # initialize to 10.0 meters
        self.allowed_window = 0.35 # this is the allowed y distance from the center line of the robot
        # if the obstacle is outside this window we ignore it
    #callback to check the current direction to check for obstacles
    def check_callback(self, msg: Bool):
        self.check_forward = msg.data

    # Callback when LIDAR data is received we want to process it
    # to determine the minimum distance to an obstacle in front of or behind the robot
    def lidar_callback(self, msg):
        # Reset the minimum clearance to a large value
        self.min_clearance.data = 10.0
        # in front checking mode, we want to look at the front 120 degrees of the LIDAR data this would be
        # from -75 to +75 degrees. Our lidar data is in radians and goes from -pi to +pi
        if self.check_forward:
            start_index = int((- math.radians(75) - msg.angle_min) / msg.angle_increment)
            num_indices = int((math.radians(150)) / msg.angle_increment)
            end_index = start_index + num_indices
            for i in range(start_index, end_index, 3): # step by 3 to reduce computation
                self.minimum_clearance_updater(msg, i) # update the minimum clearance based on this index using the helper function
                
        else: # in back checking mode, we want to look at the back 60 degrees of the LIDAR data this would be 
              # from -180 to -150 and 150 to 180 degrees
            start_index = 0
            half_numindices = int((math.radians(60)) / msg.angle_increment)
            first_end_index = half_numindices
            second_start_index = msg.ranges.__len__() - half_numindices
            end_index = msg.ranges.__len__()
            for i in range(start_index, first_end_index, 3): # step by 3 to reduce computation
                self.minimum_clearance_updater(msg, i)
            for i in range(second_start_index, end_index, 3): # step by 3 to reduce computation
                self.minimum_clearance_updater(msg, i)
        # After processing all the relevant LIDAR data, we want to publish the minimum clearance
        self.publisher_.publish(self.min_clearance)

    def minimum_clearance_updater(self, msg: Float32, idx):
        """Helper function to update the minimum clearance based on a given index of LIDAR data."""
        if msg.ranges[idx] < msg.range_min or msg.ranges[idx] > msg.range_max:
            return  # ignore invalid readings
        angle = msg.angle_min + idx * msg.angle_increment # calculate the angle of the lidar data for this index
        distance = msg.ranges[idx] # get the distance reading at this index
        y = abs(distance * math.sin(angle)) # calculate the y component
        if y < self.allowed_window:
            x = abs(distance * math.cos(angle)) # calculate the x component
            # which is the normal distance to the obstacle in front of the robot
            if x < self.min_clearance.data: # check if the current reading is less than the current minimum clearance
                self.min_clearance.data = x # if so update the minimum clearance


def main(args=None):
    rclpy.init(args=args)

    clearance_calculator = ClearanceCalculator()

    # Spin function *important*
    # makes sure that the program does not terminate
    # immediately
    rclpy.spin(clearance_calculator)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    clearance_calculator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
