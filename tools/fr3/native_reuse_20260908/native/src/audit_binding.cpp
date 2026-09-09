#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "motion/time_optimal/trajectory.h"
#include <cmath>

namespace py = pybind11;
using motion::time_optimal::Path;
using motion::time_optimal::Trajectory;
PYBIND11_MODULE(_native_plan_audit, m) {
  m.doc() = "panda-py native planner audit ONLY: no Robot, FCI, BOX or execution API";
  py::class_<Path>(m, "Path")
    .def(py::init([](const std::vector<Eigen::VectorXd>& q, double deviation) {
      if(q.size()<2 || !std::isfinite(deviation) || deviation<0 || deviation>.02)
        throw std::invalid_argument("Invalid audit path");
      for(size_t i=0;i<q.size();++i)
        if(q[i].size()!=7 || !q[i].allFinite() || (i && (q[i]-q[i-1]).norm()<1e-9))
          throw std::invalid_argument("Duplicate/nonfinite waypoint: explicit dwell handling required");
      return new Path(std::list<Eigen::VectorXd>(q.begin(),q.end()), deviation);
    }));
  py::class_<Trajectory>(m, "Trajectory")
    .def(py::init<const Path&,const Eigen::VectorXd&,const Eigen::VectorXd&,double>())
    .def("valid", &Trajectory::isValid)
    .def("duration", &Trajectory::getDuration)
    .def("state", &Trajectory::getAuditState)
    .def("waypoint_path_positions", &Trajectory::getWaypointPathPositions);
}
