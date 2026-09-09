#pragma once

#include <franka/robot.h>
#include <Eigen/Dense>
#include <atomic>
#include <mutex>
#include <memory>

#include "controllers/controller.h"
#include "utils.h"

/**
 * Pure torque controller without any filtering or damping.
 * Directly passes through the desired torques to the robot.
 */
class PureTorque : public TorqueController {
 public:
  PureTorque();
  
  franka::Torques step(const franka::RobotState &robot_state,
                      franka::Duration &duration) override;
  
  void start(const franka::RobotState &robot_state,
            std::shared_ptr<franka::Model> model) override;
  
  void stop(const franka::RobotState &robot_state,
           std::shared_ptr<franka::Model> model) override;
  
  bool isRunning() override;
  
  const std::string name() override { return "PureTorque"; }
  
  /**
   * Set the desired torques directly.
   * @param torque Desired joint torques in Nm
   */
  void setControl(const Vector7d &torque);
  
  /**
   * Get current desired torques.
   */
  Vector7d getTorques();

 private:
  std::mutex mux_;
  Vector7d tau_d_;  // Desired torques
  std::atomic<bool> motion_finished_;
};