"""ROS 2 node: contact force at the gripper from the Piper's joint efforts.

The platform has no wrist force/torque sensor; this node stands in for one
(dynamics/effort.py has the model and its limits).

Inputs
  joint_state_topic (/joint_states)  sensor_msgs/JointState with effort (joint_names);
                                     piper_joint_state_relay passes the driver's efforts on
  calibration_file                   EffortModel JSON from ``calibrate_effort`` ("" =
                                     uncalibrated defaults: gain 1, no friction; a warning)
Outputs
  wrench_topic (/ft_sensor/raw)      geometry_msgs/WrenchStamped, force on the TCP in
                                     frame_id (base_link); torque 0 (not estimated)
  ~/status                           std_msgs/String JSON at 2 Hz: calibrated, samples, force,
                                     per-axis 1-σ noise at the current pose and its 3-σ
                                     magnitude (the smallest force a threshold can trust),
                                     joint speed (the model is quasi-static), message age
  ~/tare                             std_srvs/Trigger: zero the current residual (nothing
                                     touching the gripper; stem_grasp calls it when servoing
                                     starts, see force_tare_service)

    ros2 run scout_piper_whole_body_mpc effort_force_node --ros-args \\
        -p calibration_file:=effort_calibration.json
"""

from __future__ import annotations

import json
import os
import signal

import numpy as np
import rclpy
from geometry_msgs.msg import WrenchStamped
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .dynamics.effort import ContactForceEstimator, EffortModel
from .dynamics.piper import PiperKinematics


class EffortForceNode(Node):
    def __init__(self) -> None:
        super().__init__("effort_force_estimator")
        self.declare_parameters("", [
            ("joint_state_topic", "/joint_states"),
            ("joint_names", [f"piper_joint{i}" for i in range(1, 7)]),
            ("calibration_file", ""),
            ("tcp_offset_m", 0.14),
            ("wrench_topic", "/ft_sensor/raw"),
            ("frame_id", "base_link"),
            ("filter_s", 0.1),                  # low-pass time constant on the joint residual
            ("max_joint_speed", 0.5),           # rad/s; faster motion: inertia not modelled
        ])
        p = lambda k: self.get_parameter(k).value  # noqa: E731
        self.joint_names = list(p("joint_names"))
        self.frame_id = str(p("frame_id"))
        self.max_speed = float(p("max_joint_speed"))
        path = os.path.expanduser(str(p("calibration_file")))
        if path:
            model = EffortModel.load(path)
            self.get_logger().info(f"effort model {path}: {model.samples} samples, "
                                   f"σ = {np.round(model.sigma, 3).tolist()} N·m")
        else:
            model = EffortModel()
            self.get_logger().warn("no calibration_file: uncalibrated effort model (gain 1, no friction); "
                                   "run calibrate_effort before trusting the force")
        self.est = ContactForceEstimator(PiperKinematics(tcp_offset_m=float(p("tcp_offset_m"))), model,
                                         filter_s=float(p("filter_s")))
        self.pub = self.create_publisher(WrenchStamped, str(p("wrench_topic")), 20)
        self.pub_status = self.create_publisher(String, "~/status", 10)
        self.create_subscription(JointState, str(p("joint_state_topic")), self._on_joint_state, 50)
        self.create_service(Trigger, "~/tare", self._on_tare)
        self.create_timer(0.5, self._publish_status)
        self.q = None
        self.qd = np.zeros(6)
        self.last = None
        self.t_prev = None
        self.t_msg = None
        self.tared = False
        self.warned_effort = False
        self.get_logger().info(f"{p('joint_state_topic')} efforts -> {p('wrench_topic')} ({self.frame_id})")

    def _on_joint_state(self, msg: JointState) -> None:
        idx = {n: i for i, n in enumerate(msg.name)}
        if not all(n in idx for n in self.joint_names):
            return
        if len(msg.effort) != len(msg.name):
            if not self.warned_effort:
                self.get_logger().warn("joint states carry no effort: no force estimate "
                                       "(driver efforts must reach /joint_states via the relay)")
                self.warned_effort = True
            return
        sel = [idx[n] for n in self.joint_names]
        q = np.array([msg.position[i] for i in sel])
        qd = np.array([msg.velocity[i] for i in sel]) if len(msg.velocity) == len(msg.name) else np.zeros(6)
        tau = np.array([msg.effort[i] for i in sel])
        t = self.get_clock().now().nanoseconds * 1e-9
        dt = 0.0 if self.t_prev is None else t - self.t_prev
        self.t_prev = self.t_msg = t
        self.q, self.qd = q, qd
        self.last = self.est.update(q, qd, tau, dt)
        out = WrenchStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.frame_id
        out.wrench.force.x, out.wrench.force.y, out.wrench.force.z = (float(v) for v in self.last.force)
        self.pub.publish(out)

    def _on_tare(self, request, response):
        if self.last is None:
            response.success, response.message = False, "no joint efforts received yet"
            return response
        self.est.tare()
        self.tared = True
        response.success = True
        response.message = f"tared at |F| = {self.last.magnitude:.2f} N"
        self.get_logger().info(response.message)
        return response

    def _publish_status(self) -> None:
        m = self.est.model
        st = {"calibrated": m.samples > 0, "samples": m.samples, "tared": self.tared,
              "efforts": self.last is not None}
        if self.last is not None:
            noise = self.est.force_noise(self.q)
            speed = float(np.max(np.abs(self.qd)))
            st.update({
                "force_n": np.round(self.last.force, 3).tolist(),
                "magnitude_n": round(self.last.magnitude, 3),
                "noise_n": np.round(noise, 3).tolist(),
                "threshold_3sigma_n": round(3.0 * float(np.linalg.norm(noise)), 3),
                "joint_speed": round(speed, 3),
                "quasi_static": speed <= self.max_speed,
                "age_s": round(self.get_clock().now().nanoseconds * 1e-9 - self.t_msg, 3),
            })
        self.pub_status.publish(String(data=json.dumps(st)))


def _interrupt(signum, frame) -> None:
    raise KeyboardInterrupt


def main(args=None) -> None:
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, _interrupt)
    signal.signal(signal.SIGTERM, _interrupt)
    node = EffortForceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
