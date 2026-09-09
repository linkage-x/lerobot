#include "controllers/pure_torque.h"
#include "panda.h"

PureTorque::PureTorque() : motion_finished_(false) {
  tau_d_.setZero();
}

franka::Torques PureTorque::step(const franka::RobotState &robot_state,
                                 franka::Duration &duration) {
  // Get the desired torques (thread-safe)
  Vector7d tau_d;
  mux_.lock();
  tau_d = tau_d_;
  mux_.unlock();
  
  // Directly convert to franka::Torques without any processing
  // No filtering, no damping, no gravity compensation
  // Pure pass-through of user commands
  franka::Torques torques = VectorToArray(tau_d);
  torques.motion_finished = motion_finished_;
  
  return torques;
}

void PureTorque::setControl(const Vector7d &torque) {
  std::lock_guard<std::mutex> lock(mux_);
  tau_d_ = torque;
}

Vector7d PureTorque::getTorques() {
  std::lock_guard<std::mutex> lock(mux_);
  return tau_d_;
}

void PureTorque::start(const franka::RobotState &robot_state,
                      std::shared_ptr<franka::Model> model) {
  motion_finished_ = false;
  tau_d_.setZero();  // Start with zero torques for safety
}

void PureTorque::stop(const franka::RobotState &robot_state,
                     std::shared_ptr<franka::Model> model) {
  motion_finished_ = true;
}

bool PureTorque::isRunning() { 
  return !motion_finished_; 
}