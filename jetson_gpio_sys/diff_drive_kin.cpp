#include <fstream>
#include <string>
#include <algorithm>

#include <stdexcept>
#include <iostream>
#include <chrono>

#include <thread>
#include <signal.h>
#include <atomic>
#include <cmath>
#include <assert.h>

// You can compile using the following command:
// g++ -std=c++17 -Wno-psabi diff_drive_kin.cpp -o test_motors

bool almost_equal(double a, double b, double epsilon = 1e-4) {
    return std::abs(a - b) < epsilon;
}

class Motor {
//  DON'T TOUCH THIS CODE
private:
    std::string pwm_chip_path_;
    std::string pwm_channel_path_;
    int pwm_channel_;

public:
    Motor(int pwm_chip_number, int pwm_channel) : pwm_channel_(pwm_channel) {
        pwm_chip_path_ = "/sys/class/pwm/pwmchip" + std::to_string(pwm_chip_number) + "/";
        pwm_channel_path_ = pwm_chip_path_ + "pwm" + std::to_string(pwm_channel_) + "/";
        
        std::cout << "Initializing hardware PWM on chip " << pwm_chip_number 
                  << ", channel " << pwm_channel << "..." << std::endl;
         
        // Export the PWM channel
        std::ofstream export_file(pwm_chip_path_ + "export");
        if (!export_file) {
            throw std::runtime_error("Failed to open export file for writing");
        }
        export_file << pwm_channel_;
        export_file.close();
        
        // Wait for kernel to create the PWM directory (important!)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Check if PWM channel directory was created
        std::ifstream check_file_2(pwm_channel_path_ + "period");
        if (!check_file_2) {
            throw std::runtime_error("PWM channel directory not created after export");
        }
        
        // Set period (20ms for 50Hz = 20,000,000 nanoseconds)
        std::ofstream period_file(pwm_channel_path_ + "period");
        if (!period_file) {
            throw std::runtime_error("Failed to open period file");
        }
        period_file << 20000000;
        period_file.close();
        
        // Set initial duty cycle to neutral (1.5ms)
        std::ofstream duty_file(pwm_channel_path_ + "duty_cycle");
        if (!duty_file) {
            throw std::runtime_error("Failed to open duty_cycle file");
        }
        duty_file << 1500000; // 1.5ms in nanoseconds
        duty_file.close();
        
        // Enable PWM
        std::ofstream enable_file(pwm_channel_path_ + "enable");
        if (!enable_file) {
            throw std::runtime_error("Failed to open enable file");
        }
        enable_file << 1;
        enable_file.close();
        
        std::cout << "Hardware PWM initialized successfully!" << std::endl;
    }
    
    ~Motor() {
        try {
            // Return to neutral before cleanup
            trySetVelocity(1500);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
            // Disable PWM
            std::ofstream enable_file(pwm_channel_path_ + "enable");
            enable_file << 0;
            enable_file.close();
            
            // Unexport the PWM channel
            std::ofstream unexport_file(pwm_chip_path_ + "unexport");
            unexport_file << pwm_channel_;
            unexport_file.close();
            
            std::cout << "Hardware PWM cleaned up" << std::endl;
        } catch (...) {
            // Silent cleanup in destructor
        }
    }
    
    bool trySetVelocity(int pulse_width_us) {
        // Convert microseconds to nanoseconds
        int duty_cycle_ns = pulse_width_us * 1000;
        
        std::ofstream duty_file(pwm_channel_path_ + "duty_cycle");
        if (!duty_file) {
            std::cerr << "Failed to open duty_cycle file for writing" << std::endl;
            return false;
        }
        duty_file << duty_cycle_ns;
        return !duty_file.fail();
    }
};

class DiffDriveRobot{
// DON'T TOUCH THIS CODE
private:
    std::unique_ptr<Motor> left_motor_;
    std::unique_ptr<Motor> right_motor_;

    // Variables for the conversion to motor controller
    // signals
    const int neutral_pw_ = 1500;
    const double max_radians_ = 12.0;
    const int top_fwd = 1650, top_rev = 1300;
    const int min_us = 1000, max_us = 2000;
    // Deadband values for the tested motor controller
    const int deadband_fwd = 60;
    const int deadband_rev = 60;
    const double cmd_deadband = 0.20;

    // Velocity limits for the GoBilda
    const double min_vel = -0.25;
    const double max_vel = 0.25;
    
    const double min_ang = -0.785;
    const double max_ang = 0.785;


public:
     DiffDriveRobot(std::unique_ptr<Motor> left, std::unique_ptr<Motor> right)
        : left_motor_(std::move(left)), right_motor_(std::move(right)) {
        if (!left_motor_ || !right_motor_) throw std::invalid_argument("null unique_ptr");
    }

    std::pair<double, double> InverseKinematics(const double v, const double w){
        // ENTER YOUR CODE HERE
        // Implement the inverse kinematic equations
        // From Chapter 4 in the book and worksheets:
        // i.e.
        // robot_linear_velocity = 0.5 * radius * (w_r + w_l)
        // and robot_angular_velocity = (raduis / length) * (w_r - w_l)
        // .first = left_motor, .second = right_motor
        
        // Measured Values in meters for our Gobilda Robots
        const double Radius = 0.044;
        const double Length = 0.4318;

        // you can return a pair variable like this
        // left motor angular velocity should be .first in pair
        // return {L, R};
    }

    void sendCommands(double wl, double wr, int current_step){
        // DON'T TOUCH THIS CODE
        double expected_straight = 5.68182;
        double expected_turn = 3.85185;

        // TEST_CASES
        if (current_step == 0){
            // Move forward
            std::cout << "wl: " << wl << '\n';
            assert(almost_equal(wl, expected_straight));
            assert(almost_equal(wr, expected_straight));
            std::cout << "\nTest case 1 successfull!" << '\n';
        }

        else if (current_step == 1){
            // Move backwards
            assert(almost_equal(wl, -expected_straight));
            assert(almost_equal(wr, -expected_straight));
            std::cout << "\nTest case 2 successfull!" << '\n';
        }

        else if (current_step == 2){
            // Turn left
            assert(almost_equal(wl, -expected_turn));
            assert(almost_equal(wr, expected_turn));
            std::cout << "\nTest case 3 successfull!" << '\n';
        }

        else if (current_step == 3){
            // Turn right
            assert(almost_equal(wl, expected_turn));
            assert(almost_equal(wr, -expected_turn));
            std::cout << "\nTest case 4 successfull!" << '\n';
        }

        int pulse_left  = ConvertRadPulseWidth(-wl);
        int pulse_right = ConvertRadPulseWidth(wr);

        // Clamp (motor controller safety)
        pulse_left  = std::clamp(pulse_left,  min_us, max_us);
        pulse_right = std::clamp(pulse_right, min_us, max_us);
        
        // And send to the lower level controller
        left_motor_->trySetVelocity(pulse_left);
        right_motor_->trySetVelocity(pulse_right);
        
        return;
    }

    int ConvertRadPulseWidth(double rad_s) {
        // DON'T TOUCH THIS CODE
        // Treat small commands as neutral
        if (std::fabs(rad_s) < cmd_deadband) return neutral_pw_;

        // Add limits to the motion
        const bool fwd = rad_s > 0.0;
        const double mag = std::min(std::fabs(rad_s), max_radians_);

        // Separate spans (your endpoints are asymmetric)
        const double span_fwd = (top_fwd - neutral_pw_);
        const double span_rev = (neutral_pw_ - top_rev);

        // Bigger static "punch" to beat friction (tune 90–150 µs)
        const double kS_fwd = 90;
        const double kS_rev = 100;

        // Realistic effective wheel max under load
        const double omega_wheel_max_eff = 15.0;

        const double usable_fwd = span_fwd - kS_fwd;
        const double usable_rev = span_rev - kS_rev;
        const double denom      = (omega_wheel_max_eff - cmd_deadband);
        const double kV_fwd     = usable_fwd / denom;
        const double kV_rev     = usable_rev / denom;

        double pulse = neutral_pw_;
        if (fwd) pulse += kS_fwd + kV_fwd * (mag - cmd_deadband);
        else     pulse -= kS_rev + kV_rev * (mag - cmd_deadband);

        return static_cast<int>(std::llround(std::clamp(pulse, double(min_us), double(max_us))));
    }
};

// Global flag for clean shutdown
std::atomic<bool> running{true};

// Signal handler for Ctrl+C
void signalHandler(int signum) {
    std::cout << "\nInterrupt signal (" << signum << ") received. Shutting down..." << std::endl;
    running = false;
}

int main(int argc, char* argv[]) {
    // Register signal handler
    signal(SIGINT, signalHandler);

    // Note:
    // std::cerr << "Run 'ls /sys/class/pwm/' to see available PWM chips" << std::endl;
    try {
        std::cout << "=== Hardware PWM Motor Test ===" << '\n';
        // Create Motor Objects
        // The channel and chip number were taken from the Jetson Orin /sys/class files 
        auto left_motor= std::make_unique<Motor>(0, 0);
        auto right_motor  = std::make_unique<Motor>(3, 0);

        // Create Robot Object
        DiffDriveRobot gobilda_robot(std::move(left_motor), std::move(right_motor));

        // Create a path for the robot to execute
        // These commands should get the robot roughly to the pose (x = 1.0, y = 0.5, theta = 0)
        const std::vector vel_commands = {  0.25, 0.0,   // move robot forward
                                            -0.25, 0.0,   // move backwards
                                            0.0, 0.785,  // make robot turn counter-clockwise (top view)
                                            0.0, -0.785, } ; // make robot tuen clockwise (top view)
        // Send commands to the robot
        // 8 total commands
        size_t total_commands = 8;
        std::pair<double, double> angular_vels;
        int current_step = 0;
        
        for (size_t i = 0; i < total_commands; i+=2){
            std::cout << "\nSending command " << i/2 << " to the robot!\n";
            std::cout << "Sending linear speed: " << vel_commands[i];
            std::cout << ", Sending angular speed: " << vel_commands[i+1] << '\n';

            // Compute the desired rad/s for each wheel given the target speeds
            // Inverse Kinematics
            angular_vels = gobilda_robot.InverseKinematics(vel_commands[i], vel_commands[i+1]);
            
            // Send motor speeds
            gobilda_robot.sendCommands(angular_vels.first, angular_vels.second, current_step);
            current_step++;

            // Wait for 2 seconds
            std::this_thread::sleep_for(std::chrono::seconds(2));  // wait 2 seconds
        }

        if (running) {
            std::cout << "\nMotor Test completed successfully!" << std::endl;
        }
        
        // Send neutral singals to the robot after finishing to stop!
        std::cout << "\nSending STOP to robot!" << std::endl;
        gobilda_robot.InverseKinematics(0.0, 0.0);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
