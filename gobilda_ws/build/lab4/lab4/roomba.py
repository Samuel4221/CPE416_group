import rclpy
from rclpy.node import Node

# The bridge in your logs maps ROS TwistStamped -> Gazebo Twist.
# So publish TwistStamped (or change the bridge mapping to plain Twist).
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float32, Bool


class Roomba(Node):
    """
    Simple Roomba-like controller:
      0 Stopped  -> wait for clearance, then go Forward
      1 Forward  -> if too close, Backward
      2 Backward -> back up for ~1s, then Turn
      3 Turn     -> when clear enough, Forward
    Publishes:
      - /gobilda/cmd_vel (TwistStamped)   : velocity commands (remapped in launch to /cmd_vel)
      - /gobilda/check_forward (Bool)     : tells clearance node to check front (True) or back (False)
    Subscribes:
      - /gobilda/min_clearance (Float32)  : minimum clearance (meters) based on LIDAR
    """

    def __init__(self):
        super().__init__('roomba')
        # publishers
        self.cmd_pub = self.create_publisher(TwistStamped, '/gobilda/cmd_vel', 10)
        self.check_forward_pub = self.create_publisher(Bool, '/gobilda/check_forward', 10)

        # subscriber
        self.clearance_sub = self.create_subscription(
            Float32, '/gobilda/min_clearance', self.clearance_callback, 10)
        # --- parameters / state ---
        self.cmd = TwistStamped()
        self.check_forward = Bool()
        self.check_forward.data = True  # start by checking front
        self.closest_object = 2.0  # meters initially set to close
        self.period = 0.1  # 100 ms timer runs at (~10 Hz)
        self.state = 0       # 0 Stopped, 1 Fwd, 2 Back, 3 Turn
        self.back_time = 0.0
        self.turn_time = 0.0 # time spent turning
        self.lidar_data = False #Flag to determine if we are getting data or not from the Lidar
        self.counter  = 5 #amount of timer periods since we got lidar data 
#start it at 5 so we dont go straight to moving froward with no data
        # thresholds (tune as needed)
        self.min_clear = 1.0    # meters, too close -> back up
        self.turn_clear = 3.0   # meters, clear enough after turning -> go forward

        # velocities (m/s and rad/s)
        self.fwd_speed = 0.005
        self.back_speed = -0.005
        self.turn_speed = 0.05

        # periodic control loop
        self.timer = self.create_timer(self.period, self.timer_callback)

    def clearance_callback(self, msg: Float32):
        """FSM transitions based on minimum clearance ahead (or behind when backing)."""
        self.closest_object = msg.data # in meters from clearance calculator
        self.lidar_data = True #set flag


    def timer_callback(self):
        """Publish check_forward flag and velocity each tick, update timed transitions."""
        # at the top of timer_callback, just for a test
        self.get_logger().info(f"{self.state}")
        if self.lidar_data: #if we have gotten lidar data
            self.counter = 0 #set counter to zero
            self.lidar_data = False #reset lidar data flag
        else:  #if we have not gotten lidar data
            self.counter += 1 #increment time since lidar data 
            if self.counter >= 5: #if its been over 0.5 seconds 
                self.state = 0 #stop the robot 
                self.cmd.twist.linear.x = 0.0
                self.cmd.twist.angular.z = 0.0

        self.cmd.header.stamp = self.get_clock().now().to_msg()
        # FSM logic
        if self.state == 0:  # Stopped
            # If theres enough clearance to go forward, start moving forward
            if self.counter < 5 and self.closest_object > self.min_clear: 
                self.cmd.twist.linear.x = self.fwd_speed
                self.cmd.twist.angular.z = 0.0
                self.state = 1  # start moving forward

        elif self.state == 1:  # Forward
            # If we get too close to an obstacle, start backing up
            if self.closest_object <= self.min_clear:
                self.back_time = 0.0
                self.cmd.twist.angular.z = 0.0
                self.cmd.twist.linear.x = self.back_speed
                self.state = 2  # back up
                

        elif self.state == 2:  # Backward
            self.back_time += self.period # accumulate backing time
            # After ~1 second of backing up, start turning
            if self.back_time >= 1.0:
                self.back_time = 0.0
                self.cmd.twist.linear.x = 0.0
                self.cmd.twist.angular.z = self.turn_speed
                self.state = 3  # start turning after ~1 second
            #even while backing up, if we are to close to an obstacle, go to turning
            elif self.closest_object <= self.min_clear:
                self.back_time = 0.0
                self.cmd.twist.linear.x = 0.0
                self.cmd.twist.angular.z = self.turn_speed
                self.state = 3  # start turning if still too close

        elif self.state == 3:  # Turn in place
            self.turn_time += self.period
            # After ~0.5 seconds of turning, check if we're clear to go forward
            if self.turn_time > 0.750:  # accumulate turning time
                if self.closest_object >= self.turn_clear:
                    self.cmd.twist.linear.x = self.fwd_speed
                    self.cmd.twist.angular.z = 0.0
                    self.state = 1  # go forward when sufficiently clear
                    self.turn_time = 0.0
                if self.turn_time > 2:
                    self.state = 0
                    self.turn_time = 0
                    self.cmd.twist.linear.x = 0.0
                    self.cmd.twist.angular.z = 0.0

        # Tell clearance node whether to check front (True) or back (False)
        self.check_drive_direction()
        self.cmd_pub.publish(self.cmd)

        
    def check_drive_direction(self):
        """Helper to publish whether to check front or back for obstacles."""
        # Tell clearance node whether to check front (True) or back (False)
        check_forward= Bool()
        check_forward.data = (self.state != 2)  # check front unless backing up
        self.check_forward_pub.publish(check_forward)

def main(args=None):
    rclpy.init(args=args)
    roomba = Roomba()
    # Spin function *important*
    # makes sure that the program does not terminate
    # immediately
    rclpy.spin(roomba)
     # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    roomba.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
