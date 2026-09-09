#pragma once
// ISOLATED, OFFLINE-TESTED CANDIDATE. Not an approved hardware replay controller.
// Reuses JointPosition PD and Panda's existing torque-rate limits/virtual walls.
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <franka/exception.h>
#include <limits>
#include <vector>
#include "controllers/joint_position.h"

class TimedReplayCandidate : public JointPosition {
 public:
  using Coeff = Eigen::Matrix<double, 6, 8>;
  using V8 = Eigen::Matrix<double, 8, 1>;
  using M4 = Eigen::Matrix4d;
  TimedReplayCandidate(const std::vector<double>& times,
                       const std::vector<Coeff>& coefficients,
                       const std::vector<M4>& origins,
                       const std::vector<Eigen::Vector3d>& axes, const M4& tail)
      : JointPosition((Vector7d() << 300, 300, 300, 200, 100, 80, 40).finished(),
                      (Vector7d() << 30, 30, 30, 20, 15, 12, 8).finished(), 1.),
        times_(times), coeff_(coefficients), origins_(origins), axes_(axes), tail_(tail) {
    setTime(0.);
    if (times_.size() < 2 || coeff_.size() + 1 != times_.size() || times_[0] != 0. ||
        origins_.size() != 7 || axes_.size() != 7) throw std::invalid_argument("Invalid timed plan shape");
    for (size_t i = 0; i < times_.size(); ++i)
      if (!std::isfinite(times_[i]) || (i && times_[i] <= times_[i-1]))
        throw std::invalid_argument("Invalid timed plan clock");
    for (const auto& c : coeff_) if (!c.allFinite()) throw std::invalid_argument("Nonfinite coefficients");
    for (size_t i = 0; i < 7; ++i) {
      if (!validPose(origins_[i]) || !axes_[i].allFinite() || std::abs(axes_[i].norm()-1.) > 1e-9)
        throw std::invalid_argument("Invalid FK chain");
    }
    if (!validPose(tail_)) throw std::invalid_argument("Invalid fixed TCP transform");
    for (size_t i = 1; i < coeff_.size(); ++i) {
      for (int d = 0; d <= 2; ++d) {
        const V8 a = segment(i-1, 1., d), b = segment(i, 0., d);
        // Width is C1 PCHIP; the seven arm coordinates are at least C2.
        if ((a.head<7>() - b.head<7>()).cwiseAbs().maxCoeff() > 1e-7 ||
            (d <= 1 && std::abs(a[7]-b[7]) > 1e-7))
          throw std::invalid_argument("Discontinuous trajectory");
      }
    }
    for (double t : {0., times_.back()})
      if (evaluate(t, 1).head<7>().norm() > 1e-7 || evaluate(t, 2).head<7>().norm() > 1e-7)
        throw std::invalid_argument("Arm must start/end at rest");
    if (times_.back() > 600.) throw std::invalid_argument("Bounded logger supports at most 600 seconds");
    // Preallocate before control. No file I/O, Python calls, allocations or
    // history copying in the callback. Published rows are immutable.
    telemetry_.resize(static_cast<size_t>(std::ceil((times_.back()+3.)/.005))+10);
  }

  static bool validPose(const M4& t) {
    return t.allFinite() && (t.row(3) - Eigen::RowVector4d(0,0,0,1)).norm() < 1e-8 &&
        (t.topLeftCorner<3,3>().transpose()*t.topLeftCorner<3,3>() - Eigen::Matrix3d::Identity()).norm() < 1e-7 &&
        std::abs(t.topLeftCorner<3,3>().determinant()-1.) < 1e-7;
  }

  V8 evaluate(double time, int derivative=0) const {
    if (!std::isfinite(time) || time < 0 || derivative < 0 || derivative > 3)
      throw std::invalid_argument("Invalid trajectory query");
    if (time > times_.back() && derivative) return V8::Zero();
    const double t = std::min(time, times_.back());
    const size_t i = std::min<size_t>(std::upper_bound(times_.begin(), times_.end(), t)-times_.begin()-1, coeff_.size()-1);
    return segment(i, (t-times_[i])/(times_[i+1]-times_[i]), derivative);
  }

  M4 fk(const Vector7d& q) const {
    M4 t = M4::Identity();
    for (size_t i = 0; i < 7; ++i) {
      M4 r = M4::Identity();
      r.topLeftCorner<3,3>() = Eigen::AngleAxisd(q[i], axes_[i]).toRotationMatrix();
      t = (t * origins_[i] * r).eval();
    }
    return t * tail_;
  }

  // Single sensor-producer only. seq must identify a NEW device measurement,
  // not merely a repeatedly-read SDK cache. Its receipt uses the arm clock.
  void feedGripper(uint64_t seq, double received_at_arm_s, double width_m) {
    if (!std::isfinite(received_at_arm_s) || !std::isfinite(width_m) || width_m < 0 ||
        width_m > .088739804924 || received_at_arm_s < 0 || received_at_arm_s > getTime()+1e-6 ||
        seq <= producer_seq_ || received_at_arm_s < producer_stamp_)
      throw std::invalid_argument("Gripper measurement is invalid, old or future-dated");
    producer_seq_ = seq; producer_stamp_ = received_at_arm_s;
    sample_version_.fetch_add(1, std::memory_order_acq_rel);
    width_.store(width_m, std::memory_order_relaxed);
    stamp_.store(received_at_arm_s, std::memory_order_relaxed);
    sample_version_.fetch_add(1, std::memory_order_release);
  }

  void abortExternal() { external_fault_.store(true); }
  void acknowledgeGripperCommand(double command_arm_time, double width_m) {
    if (!std::isfinite(command_arm_time) || command_arm_time < 0 || command_arm_time > getTime()+1e-6 ||
        !std::isfinite(width_m) || std::abs(width_m-evaluate(command_arm_time)[7]) > 1e-9)
      throw std::invalid_argument("Gripper acknowledgement does not match the arm-clock target");
    if (command_arm_time >= times_.back()) final_ack_stamp_.store(command_arm_time);
  }
  int faultCode() const { return fault_.load(); }
  double duration() const { return times_.back(); }
  bool completed() const { return completed_.load(); }
  Eigen::Matrix<double, Eigen::Dynamic, 48> telemetry() const {
    const size_t n = telemetry_count_.load(std::memory_order_acquire);
    Eigen::Matrix<double, Eigen::Dynamic, 48> out(n,48);
    for (size_t i=0; i<n; ++i)
      for (size_t j=0; j<48; ++j) out(i,j)=telemetry_[i][j];
    return out;
  }

  void start(const franka::RobotState& state, std::shared_ptr<franka::Model> model) override {
    // Faults latch permanently on this object. Never auto-recover or auto-restart.
    if (fault_.load() || started_) throw franka::Exception("Timed replay candidate cannot be restarted");
    setTime(0.);
    updateSensor();
    const Vector7d q = Eigen::Map<const Vector7d>(state.q.data());
    const Vector7d dq = Eigen::Map<const Vector7d>(state.dq.data());
    const auto target = evaluate(0.);
    const M4 actual = Eigen::Map<const M4>(state.O_T_EE.data());
    if (!q.allFinite() || !dq.allFinite() || !validPose(actual) ||
        (q-target.head<7>()).cwiseAbs().maxCoeff() > .01 || dq.cwiseAbs().maxCoeff() > .01)
      fail(1, "Start state is not the first frame at rest; start transit is not implemented");
    if (!poseClose(actual, fk(q), .002, .0174532925199433)) fail(2, "Controller TCP and URDF are inconsistent");
    if (sensor_stamp_ != 0. || std::abs(sensor_width_-target[7]) > .002)
      fail(3, "Fresh initial gripper position is required");
    JointPosition::start(state, model);
    started_ = true;
  }

  franka::Torques step(const franka::RobotState& state, franka::Duration& period) override {
    if (fault_.load()) throw franka::Exception("Timed replay fault is latched");
    if (!started_) fail(4, "Controller was not initialized");
    const double t = getTime(), dt = period.toSec();
    if (!std::isfinite(t) || t < last_time_ || t-last_time_ > .010001 || dt > .010001 || dt < 0)
      fail(5, "Control clock discontinuity / excessive callback gap");
    last_time_ = t;
    record(state, t);
    if (external_fault_.load()) fail(6, "External supervisor or gripper transport failed");
    updateSensor();
    if (t-sensor_stamp_ > .2) fail(7, "Gripper measurement watchdog expired");
    const auto target = evaluate(t), vel = evaluate(t, 1), acc = evaluate(t, 2), jerk = evaluate(t, 3);
    const Vector7d q = Eigen::Map<const Vector7d>(state.q.data());
    const Vector7d dq = Eigen::Map<const Vector7d>(state.dq.data());
    const M4 actual = Eigen::Map<const M4>(state.O_T_EE.data());
    const Vector7d lower = (Vector7d() << -2.7437,-1.7837,-2.9007,-3.0421,-2.8065,.5445,-3.0159).finished();
    const Vector7d upper = (Vector7d() << 2.7437,1.7837,2.9007,-.1518,2.8065,4.5169,3.0159).finished();
    const Vector7d band = (Vector7d() << .24,.18,.18,.18,.0698,.0698,.0698).finished();
    if (!q.allFinite() || !dq.allFinite() || !validPose(actual)) fail(8, "Invalid measured arm state");
    if ((target.head<7>().array() <= (lower+band).array()).any() ||
        (target.head<7>().array() >= (upper-band).array()).any() ||
        (q.array() <= (lower+band).array()).any() || (q.array() >= (upper-band).array()).any())
      fail(9, "Actual or desired joints entered a virtual-wall band");
    if (vel.head<7>().cwiseAbs().maxCoeff() > .1200001 || acc.head<7>().cwiseAbs().maxCoeff() > .4000001 ||
        jerk.head<7>().cwiseAbs().maxCoeff() > 5.000001 || std::abs(vel[7]) > .0100001 || target[7] < -1e-12 || target[7] > .088739804924)
      fail(10, "Timed plan limits violated");
    if (!poseClose(actual, fk(q), .002, .0174532925199433)) fail(2, "Controller TCP and URDF diverged");
    track_bad_s_ = (q-target.head<7>()).cwiseAbs().maxCoeff() > .05 ? track_bad_s_ + dt : 0.;
    tcp_bad_s_ = !poseClose(actual, fk(target.head<7>()), .01, .0872664625997165) ? tcp_bad_s_ + dt : 0.;
    grip_bad_s_ = std::abs(sensor_width_-target[7]) > .005 ? grip_bad_s_ + dt : 0.;
    if (track_bad_s_ >= .1-1e-9) fail(11, "Persistent joint tracking error");
    if (tcp_bad_s_ >= .1-1e-9) fail(12, "Persistent configured-TCP tracking error");
    if (grip_bad_s_ >= .5-1e-9) fail(13, "Persistent gripper tracking error");
    setControl(target.head<7>(), vel.head<7>());
    const double final_ack = final_ack_stamp_.load();
    if (t >= times_.back() && final_ack >= times_.back() && sensor_stamp_ >= final_ack &&
        (q-target.head<7>()).cwiseAbs().maxCoeff() < .01 &&
        dq.cwiseAbs().maxCoeff() < .001 && std::abs(sensor_width_-target[7]) < .002) {
      completed_.store(true);
      JointPosition::stop(state, nullptr);
    }
    if (t > times_.back()+2.) fail(14, "End position did not settle");
    return JointPosition::step(state, period);
  }

  const std::string name() override { return "Timed Replay STAGED - physical validation pending"; }

 private:
  std::vector<double> times_;
  std::vector<Coeff> coeff_;
  std::vector<M4> origins_;
  std::vector<Eigen::Vector3d> axes_;
  M4 tail_;
  std::atomic<int> fault_{0};
  std::atomic<bool> completed_{false};
  std::vector<std::array<double,48>> telemetry_;
  std::atomic<size_t> telemetry_count_{0};
  double last_record_s_=-1.;
  std::atomic<bool> external_fault_{false};
  std::atomic<uint64_t> sample_version_{0};
  std::atomic<double> width_{0.}, stamp_{-1.};
  std::atomic<double> final_ack_stamp_{-1.};
  uint64_t producer_seq_=0;
  double producer_stamp_=-1., sensor_width_=0., sensor_stamp_=-1.;
  bool started_=false;
  double last_time_=0., track_bad_s_=0., tcp_bad_s_=0., grip_bad_s_=0.;

  V8 segment(size_t i, double u, int derivative) const {
    V8 result = V8::Zero();
    for (int row=0; row < 6-derivative; ++row) {
      double factor=1.;
      for (int d=0; d<derivative; ++d) factor *= 5-row-d;
      result = (result*u + coeff_[i].row(row).transpose()*factor).eval();
    }
    return result / std::pow(times_[i+1]-times_[i], derivative);
  }
  void record(const franka::RobotState& state, double time) {
    if (time-last_record_s_ < .005-1e-9) return;
    const size_t n=telemetry_count_.load(std::memory_order_relaxed);
    if (n>=telemetry_.size()) fail(15,"Preallocated telemetry is full");
    auto& row=telemetry_[n];
    size_t j=0;
    row[j++]=time;
    for (double x:state.q) row[j++]=x;
    for (double x:state.dq) row[j++]=x;
    for (double x:state.O_T_EE) row[j++]=x;
    const V8 target=evaluate(time);
    for (int k=0;k<8;++k) row[j++]=target[k];
    row[j++]=width_.load();
    row[j++]=state.control_command_success_rate;
    for (double x:state.tau_ext_hat_filtered) row[j++]=x;
    last_record_s_=time;
    telemetry_count_.store(n+1,std::memory_order_release);
  }
  void updateSensor() {
    // Never block or spin in the control callback if the producer is writing.
    const auto first = sample_version_.load(std::memory_order_acquire);
    if (first & 1) return;
    const double width = width_.load(std::memory_order_relaxed), stamp = stamp_.load(std::memory_order_relaxed);
    if (first == sample_version_.load(std::memory_order_acquire)) {
      sensor_width_ = width; sensor_stamp_ = stamp;
    }
  }
  static bool poseClose(const M4& a, const M4& b, double xyz, double angle) {
    const double cosine = std::clamp(((a.topLeftCorner<3,3>().transpose()*b.topLeftCorner<3,3>()).trace()-1.)/2., -1., 1.);
    return (a.topRightCorner<3,1>()-b.topRightCorner<3,1>()).norm() <= xyz && std::acos(cosine) <= angle;
  }
  [[noreturn]] void fail(int code, const char* reason) {
    int zero=0; fault_.compare_exchange_strong(zero, code);
    // Panda::_runController catches this type and terminates control; it does
    // not recover, weaken robot protections, resume or pretend completion.
    throw franka::Exception(reason);
  }
};
