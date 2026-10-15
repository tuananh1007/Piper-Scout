"""Software stop shared by the instruction node and the operator GUI (P5.3.6).

The same stop as stem_grasp's hot-key ('x' in ``hotkey_stop_and_zero.py``)
plus the base and the planners: zero twists to moveit_servo (Cartesian and
joint jog, the MPPI servo's output), a zero ``/cmd_vel`` to the Scout, a
cancel to the whole-body MPC (which then sends its own stop), and the
piper_servo_bridge disabled so the Piper holds its measured pose and ignores
further servo commands until it is re-enabled. The service call is
asynchronous; ``bridge_confirmed`` turns true when the bridge answers.

It is a software stop: it does not cut power. The physical stops stay within
reach (INSTALL.md Step 9).
"""

from __future__ import annotations

from geometry_msgs.msg import Twist, TwistStamped
from std_msgs.msg import Empty
from std_srvs.srv import SetBool

try:
    from control_msgs.msg import JointJog
except ImportError:                                   # control_msgs missing: no joint-jog zero
    JointJog = None


class SafetyStop:
    def __init__(self, node, servo_twist_topic="/servo_node/delta_twist_cmds",
                 servo_jog_topic="/servo_node/delta_joint_cmds", cmd_vel_topic="/cmd_vel",
                 mpc_cancel_topic="/whole_body_mpc/cancel", bridge_enable_service="/piper_servo_bridge/enable"):
        self.node = node
        self.pub_twist = node.create_publisher(TwistStamped, servo_twist_topic, 5)
        self.pub_jog = node.create_publisher(JointJog, servo_jog_topic, 5) if JointJog is not None else None
        self.pub_base = node.create_publisher(Twist, cmd_vel_topic, 5)
        self.pub_cancel = node.create_publisher(Empty, mpc_cancel_topic, 5)
        self.bridge = node.create_client(SetBool, bridge_enable_service)
        self.bridge_confirmed = False
        self.count = 0

    def trigger(self, reason: str = "operator") -> dict:
        self.count += 1
        self.bridge_confirmed = False
        now = self.node.get_clock().now().to_msg()
        self.pub_cancel.publish(Empty())
        for _ in range(5):
            t = TwistStamped()
            t.header.stamp = now
            self.pub_twist.publish(t)
            if self.pub_jog is not None:
                j = JointJog()
                j.header.stamp = now
                self.pub_jog.publish(j)
            self.pub_base.publish(Twist())
        bridge = "not running"
        if self.bridge.service_is_ready():
            fut = self.bridge.call_async(SetBool.Request(data=False))
            fut.add_done_callback(self._done)
            bridge = "disabling"
        self.node.get_logger().warn(f"STOP ({reason}): zero commands sent, MPC cancelled, servo bridge {bridge}")
        return {"stopped": True, "reason": reason, "bridge": bridge, "count": self.count}

    def _done(self, fut) -> None:
        res = fut.result()
        self.bridge_confirmed = bool(res is not None and res.success)
