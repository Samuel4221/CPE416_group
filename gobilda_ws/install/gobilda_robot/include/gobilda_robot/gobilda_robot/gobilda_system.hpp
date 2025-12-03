#ifndef GOBILDA_ROBOT__GOBILDA_SYSTEM_HPP_
#define GOBILDA_ROBOT__GOBILDA_SYSTEM_HPP_

#include <memory>
#include <string>
#include <vector>

#include "hardware_interface/handle.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "rclcpp/clock.hpp"
#include "rclcpp/duration.hpp"
#include "rclcpp/macros.hpp"
#include "rclcpp/time.hpp"
#include "rclcpp_lifecycle/node_interfaces/lifecycle_node_interface.hpp"
#include "rclcpp_lifecycle/state.hpp"
#include "gobilda_robot/motor.hpp"

namespace gobilda_robot
{
class GobildaSystemHardware : public hardware_interface::SystemInterface
{
public:
  RCLCPP_SHARED_PTR_DEFINITIONS(GobildaSystemHardware)

  hardware_interface::CallbackReturn on_init(
    const hardware_interface::HardwareInfo & info) override;

  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;

  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

  hardware_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::return_type read(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

  hardware_interface::return_type write(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  // Store the command for the simulated robot
  std::vector<double> hw_commands_;
  std::vector<double> hw_positions_;
  std::vector<double> hw_velocities_;
  std::vector<std::unique_ptr<Motor>> motors_; // Change to pointers
  // Code assumes that Right motor is on pin 32
  // and Left motor is on pin 15
  std::vector<int> pwm_chip_numbers_ = {0, 3};

  // Calibrated values 
  // TODO: (consider making ROS params)
  // These values were tuned for a specific Gobilda number
  // They most likely will need to be tuned for every robot/
  // motor controller. NOTE that I used thr 312 rpm motors
  // for these values.
  const double neutral_us           = 1480.0;
  const double deadband_fwd_us      = 40.0;
  const double deadband_rev_us      = 40.0;
  const double gain_fwd_us_per_rads = 27.0;
  const double gain_rev_us_per_rads = 20.0;
  const double cmd_deadband_rad_s   = 0.05;
  const double top_fwd_us           = 1950.0;  // cap forward
  const double top_rev_us           = 1050.0;  // cap reverse
  const double min_us               = 1000.0;
  const double max_us               = 2000.0;
};

}  // namespace gobilda_robot

#endif //GOBILDA_ROBOT__GOBILDA_SYSTEM_HPP
