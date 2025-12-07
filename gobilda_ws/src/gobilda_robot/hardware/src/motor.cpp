#include "gobilda_robot/motor.hpp"
// non_blocking motor.cpp
#include <fstream>
#include <chrono>

Motor::Motor(int pwm_chip, int pwm_channel) 
    : pwm_chip_number_(pwm_chip), pwm_channel_(pwm_channel) {

    pwm_chip_path_ = "/sys/class/pwm/pwmchip" + std::to_string(pwm_chip_number_) + "/";
    pwm_channel_path_ = pwm_chip_path_ + "pwm" + std::to_string(pwm_channel_) + "/";
    pwm_duty_cycle_path_ = pwm_channel_path_ + "duty_cycle";
        
    // 1) Export if necessary
    if (!std::ifstream(pwm_channel_path_ + "period")) {
        std::ofstream(pwm_chip_path_ + "export") << pwm_channel_;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // 2) Open FDs after sysfs nodes exist
    period_fd_ = open((pwm_channel_path_ + "period").c_str(),  O_WRONLY | O_CLOEXEC);
    duty_fd_   = open((pwm_channel_path_ + "duty_cycle").c_str(), O_WRONLY | O_CLOEXEC);
    enable_fd_ = open((pwm_channel_path_ + "enable").c_str(),  O_WRONLY | O_CLOEXEC);
    if (period_fd_ < 0 || duty_fd_ < 0 || enable_fd_ < 0) throw std::runtime_error("open fd");

    // 3) Configure period + neutral
    {
        char buf[16]; int len = snprintf(buf, sizeof(buf), "%d", period_ns_);
        pwrite(period_fd_, buf, len, 0);
        len = snprintf(buf, sizeof(buf), "%d", duty_cycle_ns_);
        pwrite(duty_fd_, buf, len, 0);
        pwrite(enable_fd_, "1", 1, 0);
    }

    running_ = true;
    pwm_thread_ = std::thread(&Motor::hardwareThread, this);
}

Motor::~Motor() {
    stop();
}

bool Motor::trySetVelocity(int pulse_width_us) {
    // NON-BLOCKING: Just update the target value
    target_pulse_.store(pulse_width_us);
    return true;
}

void Motor::stop() {
    running_ = false;
    if (pwm_thread_.joinable()) pwm_thread_.join();
    if (enable_fd_ >= 0) pwrite(enable_fd_, "0", 1, 0);
    if (period_fd_ >= 0) close(period_fd_);
    if (duty_fd_   >= 0) close(duty_fd_);
    if (enable_fd_ >= 0) close(enable_fd_);
    period_fd_ = duty_fd_ = enable_fd_ = -1;
}

void Motor::hardwareThread() {
    // Get time now, so that I can sleep for variable time
    // needed to hit my control loop
    auto next = std::chrono::steady_clock::now();
    while (running_) {
        int current_pulse = target_pulse_.load();
        // BLOCKING call in a separate thread
        setVelocityHardware(current_pulse);
        // Sleep until next iteration so that we hit timing rate
        next += std::chrono::milliseconds(20);
        std::this_thread::sleep_until(next);
    }
}

bool Motor::setVelocityHardware(int pulse_width_us) {
    // This is the original BLOCKING implementation
    // but now it runs in a separate thread
    // TODO: Clamp + write-if-changed
    const int duty_ns = pulse_width_us * 1000;

    char buf[16];
    int len = snprintf(buf, sizeof(buf), "%d", duty_ns);
    if (len <= 0) return false;
    // Write to the duty cycle file, which controls the PWM pins
    if (pwrite(duty_fd_, buf, len, 0) < 0) { /* set io_error_ */ return false; }
    return true;
}