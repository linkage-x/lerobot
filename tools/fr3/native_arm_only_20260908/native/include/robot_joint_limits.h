#pragma once
#include <cstdint>
#include <stdexcept>
#include "constants.h"

struct RobotJointLimits {
  Vector7d lower;
  Vector7d upper;
  const char* name;
};

// Conservative FR3 envelope used before Robot System 5.9.0. Values agree with
// upstream panda-py 9038c06/include/constants.h and the existing P0 FR3 URDF.
// This isolated candidate is validated only for the observed FCI protocol 9
// (Desk identifies Arm3R, system 5.8.1). Reject other versions, never guess.
inline RobotJointLimits robotJointLimitsForServerVersion(uint16_t version) {
  if (version != 9) {
    throw std::runtime_error(
        "FR3 5.8.1 limits candidate requires verified FCI server version 9; "
        "unsupported robot/version, motion must not start.");
  }
  Vector7d lower, upper;
  lower << -2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159;
  upper << 2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159;
  return {lower, upper, "FR3 pre-5.9 (verified protocol 9)"};
}
