"""Small always-on-top window: FREEZE / UNFREEZE / ZERO for the Piper.

Started by quickstart.launch.py next to RViz. Keyboard (while the window has
focus): Space or Esc = freeze, U = unfreeze, Z = zero (asks to confirm).
Each button calls the matching std_srvs/Trigger service (/piper/freeze,
/piper/unfreeze, /piper/zero), which also work from a terminal:

    ros2 service call /piper/freeze std_srvs/srv/Trigger
"""

from __future__ import annotations

import queue
import sys
import threading
import tkinter as tk
from tkinter import messagebox

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool
from std_srvs.srv import Trigger


class PanelNode(Node):
    def __init__(self, events: "queue.Queue") -> None:
        super().__init__("safety_panel")
        self._events = events
        self.clients_ = {name: self.create_client(Trigger, f"/piper/{name}") for name in ("freeze", "unfreeze", "zero")}
        self.create_subscription(
            Bool, "/piper/frozen", lambda m: events.put(("frozen", m.data)),
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE))

    def call(self, name: str) -> None:
        client = self.clients_[name]
        if not client.service_is_ready():
            self._events.put(("result", name, False, f"/piper/{name} is not available"))
            return
        fut = client.call_async(Trigger.Request())
        fut.add_done_callback(lambda f: self._events.put(
            ("result", name, bool(f.result() and f.result().success),
             f.result().message if f.result() else "no response")))


class Panel:
    def __init__(self, node: PanelNode, events: "queue.Queue") -> None:
        self.node, self.events = node, events
        self.root = tk.Tk()
        self.root.title("Piper safety")
        self.root.attributes("-topmost", True)
        self.root.resizable(False, False)
        big = ("Helvetica", 18, "bold")
        self.freeze_btn = tk.Button(self.root, text="FREEZE  (Space / Esc)", font=big, bg="#c62828", fg="white",
                                    activebackground="#e53935", activeforeground="white", height=2, width=22,
                                    command=lambda: self.node.call("freeze"))
        self.freeze_btn.pack(padx=10, pady=(10, 6), fill="x")
        row = tk.Frame(self.root)
        row.pack(padx=10, pady=4, fill="x")
        tk.Button(row, text="Unfreeze (U)", font=("Helvetica", 12), width=12,
                  command=lambda: self.node.call("unfreeze")).pack(side="left", expand=True, fill="x", padx=(0, 4))
        tk.Button(row, text="Zero arm (Z)", font=("Helvetica", 12), width=12,
                  command=self.confirm_zero).pack(side="left", expand=True, fill="x", padx=(4, 0))
        self.state = tk.Label(self.root, text="state: unknown", font=("Helvetica", 12, "bold"))
        self.state.pack(padx=10, pady=(6, 0))
        self.last = tk.Label(self.root, text="", font=("Helvetica", 10), wraplength=320, justify="left")
        self.last.pack(padx=10, pady=(2, 10))
        for key in ("<space>", "<Escape>"):
            self.root.bind(key, lambda _e: self.node.call("freeze"))
        for key in ("u", "U"):
            self.root.bind(key, lambda _e: self.node.call("unfreeze"))
        for key in ("z", "Z"):
            self.root.bind(key, lambda _e: self.confirm_zero())
        self.root.after(50, self.poll)

    def confirm_zero(self) -> None:
        if messagebox.askokcancel("Zero arm", "Move the arm to its zero pose now?\n"
                                  "MoveIt plans a collision-checked path at reduced speed.", parent=self.root):
            self.last.config(text="zero: moving...", fg="black")
            self.node.call("zero")

    def poll(self) -> None:
        while True:
            try:
                ev = self.events.get_nowait()
            except queue.Empty:
                break
            if ev[0] == "frozen":
                self.state.config(text="FROZEN" if ev[1] else "running (not frozen)",
                                  fg="#c62828" if ev[1] else "#2e7d32")
            else:
                _, name, ok, msg = ev
                self.last.config(text=f"{name}: {msg}", fg="black" if ok else "#c62828")
        self.root.after(50, self.poll)


def main(args=None) -> int:
    rclpy.init(args=args)
    events: "queue.Queue" = queue.Queue()
    node = PanelNode(events)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    try:
        Panel(node, events).root.mainloop()
    except tk.TclError as e:
        print(f"safety_panel: cannot open a window ({e}); use `ros2 service call /piper/freeze std_srvs/srv/Trigger`",
              file=sys.stderr)
        return 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
