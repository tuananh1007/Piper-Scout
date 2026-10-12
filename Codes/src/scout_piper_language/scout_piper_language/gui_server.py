"""Operator GUI web server (P5.3): static page, JSON API, server-sent events, MJPEG.

ROS-free: a ``backend`` object provides the robot side (``operator_gui_node``
bridges it to ROS; ``FakeBackend`` serves the page without a robot, for the
browser test and for trying the interface). The page (``web/``) runs in any
browser on the operator's laptop or tablet; a Tauri shell can wrap it later
(P5.3.1) without changing the server.

Routes
  GET  /                    the page (index.html, app.js, style.css)
  GET  /api/state           snapshot JSON (pipeline, MPC, uncertainty, session, detections)
  GET  /api/events          server-sent events: say, proposal, state, stop
  GET  /api/stream.mjpg     camera image with the detections drawn (multipart JPEG)
  POST /api/instruction     {"text"}           -> the instruction node
  POST /api/confirm | /api/cancel              -> the pending proposal
  POST /api/abort           software stop (also on Escape in the page); always allowed
  POST /api/drive           {"v", "w"} base, or {"arm": [vx, vy, vz, wx, wy, wz]} TCP jog; each
                            command holds for ``hold_s`` (0.3 s), then zero: hold-to-move
  POST /api/drive_enable    {"enable"} arm the drive tab (servo started, bridge enabled)
  POST /api/voice           WAV body -> transcript (whisper-cpp, when configured) -> instruction
  GET  /api/tts?text=       WAV of the text (Piper TTS, when configured); 501 otherwise, and
                            the page falls back to the browser's speech synthesis

Binds to 127.0.0.1 by default; ``--host 0.0.0.0`` for a tablet on the robot's
network (no authentication: keep it on a closed network).
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png"}


class EventHub:
    """Fan-out of events to every connected SSE client."""

    def __init__(self) -> None:
        self._subs: List[queue.Queue] = []
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=200)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def publish(self, kind: str, data: Dict) -> None:
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait((kind, data))
            except queue.Full:
                pass


class FakeBackend:
    """Robot stand-in: answers instructions with a proposal and a countdown,
    records stops and drive commands (browser test, demo without a robot)."""

    def __init__(self, hub: EventHub) -> None:
        self.hub = hub
        self.log: List[Dict] = []
        self.pending: Optional[Dict] = None
        self.drive_enabled = False
        self.stops = 0
        self.dets = [{"index": 0, "class": "flower", "colour": "red", "bbox": [60, 50, 120, 110]},
                     {"index": 1, "class": "flower", "colour": "yellow", "bbox": [200, 60, 250, 115]},
                     {"index": 2, "class": "stem", "colour": "green", "bbox": [140, 80, 160, 230]}]
        self.t0 = time.time()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            time.sleep(0.2)
            if self.pending and time.time() >= self.pending["deadline"]:
                p, self.pending = self.pending, None
                self.hub.publish("proposal", {"type": "commit", **p})
                self.hub.publish("say", {"text": "Grasping."})
            self.hub.publish("state", self.snapshot())

    def _rec(self, kind, **kw):
        e = {"kind": kind, "t": round(time.time() - self.t0, 3), **kw}
        self.log.append(e)
        return e

    def snapshot(self) -> Dict:
        return {"pipeline": "SCANNING" if self.pending is None else "IDLE", "mpc": {"mode": "idle"},
                "uncertainty": {"u_jepa": 0.2, "u_geometry": 0.6, "u_twin": 0.0, "needs_view": True,
                                "unmet": ["corridor"]},
                "twin": {"confidence": 0.8, "valid": True}, "force_n": 0.05,
                "session": {"state": "countdown" if self.pending else "idle",
                            "countdown_remaining_s": max(0.0, self.pending["deadline"] - time.time()) if self.pending else 0.0},
                "detections": self.dets, "drive_enabled": self.drive_enabled, "stops": self.stops,
                "connected": True}

    def instruction(self, text: str) -> Dict:
        self._rec("instruction", text=text)
        t = text.lower()
        if "stop" in t:
            return self.abort()
        if "flower" in t and "red" not in t and "yellow" not in t:
            self.hub.publish("proposal", {"type": "question", "text": "I see 2 that match the flower. Which one: "
                                          "the left one or the right one?", "options": ["the left one", "the right one"]})
            return {"ok": True}
        det = self.dets[1] if "yellow" in t or "right" in t else self.dets[0]
        self.pending = {"id": len(self.log), "action": "grasp", "target": {"index": det["index"],
                        "label": f"#{det['index'] + 1} {det['colour']} {det['class']}", "bbox": det["bbox"]},
                        "message": f"The {det['colour']} flower: #{det['index'] + 1}.", "countdown_s": 2.0,
                        "advisories": [], "deadline": time.time() + 2.0}
        self.hub.publish("proposal", {"type": "proposal", **self.pending})
        self.hub.publish("say", {"text": self.pending["message"] + " Grasping in 2 s unless you cancel."})
        return {"ok": True}

    def confirm(self) -> Dict:
        self._rec("confirm")
        if self.pending:
            self.pending["deadline"] = 0.0
        return {"ok": True}

    def cancel(self) -> Dict:
        self._rec("cancel")
        if self.pending:
            self.pending = None
            self.hub.publish("proposal", {"type": "cancelled", "reason": "cancelled"})
            self.hub.publish("say", {"text": "Cancelled."})
        return {"ok": True}

    def abort(self) -> Dict:
        self.stops += 1
        self.pending = None
        self.drive_enabled = False
        self._rec("abort")
        self.hub.publish("stop", {"stopped": True, "count": self.stops})
        self.hub.publish("say", {"text": "Stopped."})
        return {"stopped": True, "count": self.stops}

    def drive(self, cmd: Dict) -> Dict:
        if not self.drive_enabled:
            return {"ok": False, "reason": "drive not enabled"}
        self._rec("drive", **cmd)
        return {"ok": True}

    def drive_enable(self, enable: bool) -> Dict:
        self.drive_enabled = bool(enable)
        self._rec("drive_enable", enable=self.drive_enabled)
        return {"ok": True, "enabled": self.drive_enabled}

    def jpeg(self) -> Optional[bytes]:
        try:
            import cv2  # noqa: PLC0415
            import numpy as np  # noqa: PLC0415
        except ImportError:
            return None
        img = np.full((240, 320, 3), (60, 110, 60), np.uint8)
        k = int((time.time() * 10) % 40)
        cv2.circle(img, (90 + k // 4, 80), 22, (40, 40, 220), -1)
        cv2.circle(img, (225, 88), 20, (30, 220, 230), -1)
        cv2.line(img, (150, 230), (150, 90), (40, 140, 40), 6)
        ok, buf = cv2.imencode(".jpg", img)
        return buf.tobytes() if ok else None

    def transcribe(self, wav: bytes) -> Optional[str]:
        return None

    def tts(self, text: str) -> Optional[bytes]:
        return None


class Handler(BaseHTTPRequestHandler):
    server_version = "PiperScoutGUI/1"

    def log_message(self, fmt, *args):            # quiet
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length", "0") or 0)
        return self.rfile.read(min(n, 20_000_000)) if n > 0 else b""

    def do_GET(self):                              # noqa: N802
        b = self.server.backend
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            return self._static("index.html")
        if u.path in ("/app.js", "/style.css", "/icon.svg"):
            return self._static(u.path[1:])
        if u.path == "/api/state":
            return self._json(b.snapshot())
        if u.path == "/api/debug_log" and isinstance(b, FakeBackend):
            return self._json(b.log)
        if u.path == "/api/events":
            return self._sse()
        if u.path == "/api/stream.mjpg":
            return self._mjpeg()
        if u.path == "/api/tts":
            text = parse_qs(u.query).get("text", [""])[0][:500]
            wav = b.tts(text) if text else None
            if wav is None:
                return self._json({"error": "no TTS configured"}, 501)
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(wav)))
            self.end_headers()
            self.wfile.write(wav)
            return None
        return self._json({"error": "not found"}, 404)

    def do_POST(self):                             # noqa: N802
        b = self.server.backend
        path = urlparse(self.path).path
        if path == "/api/abort":                   # first, and without parsing anything
            return self._json(b.abort())
        raw = self._body()
        if path == "/api/voice":
            text = b.transcribe(raw)
            if text is None:
                return self._json({"error": "no speech recognition configured"}, 501)
            res = b.instruction(text) if text.strip() else {"ok": False}
            return self._json({"transcript": text, **res})
        try:
            data = json.loads(raw.decode() or "{}")
        except ValueError:
            return self._json({"error": "bad JSON"}, 400)
        if path == "/api/instruction":
            text = str(data.get("text", "")).strip()[:500]
            return self._json(b.instruction(text) if text else {"ok": False, "reason": "empty"})
        if path == "/api/confirm":
            return self._json(b.confirm())
        if path == "/api/cancel":
            return self._json(b.cancel())
        if path == "/api/drive":
            return self._json(b.drive(data))
        if path == "/api/drive_enable":
            return self._json(b.drive_enable(bool(data.get("enable"))))
        return self._json({"error": "not found"}, 404)

    def _static(self, name):
        p = os.path.join(WEB, name)
        if not os.path.isfile(p):
            return self._json({"error": "missing"}, 404)
        with open(p, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", TYPES.get(os.path.splitext(name)[1], "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _sse(self):
        hub = self.server.hub
        q = hub.subscribe()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            self.wfile.write(b"retry: 1000\n\n")
            self.wfile.flush()
            while not self.server.stopping:
                try:
                    kind, data = q.get(timeout=5.0)
                    msg = f"event: {kind}\ndata: {json.dumps(data)}\n\n"
                except queue.Empty:
                    msg = ": keep-alive\n\n"
                self.wfile.write(msg.encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            hub.unsubscribe(q)

    def _mjpeg(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            while not self.server.stopping:
                jpg = self.server.backend.jpeg()
                if jpg:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    self.wfile.flush()
                time.sleep(1.0 / self.server.fps)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


class GuiServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, backend, hub: EventHub, fps: float = 10.0):
        super().__init__(addr, Handler)
        self.backend, self.hub, self.fps = backend, hub, fps
        self.stopping = False

    def start(self) -> threading.Thread:
        t = threading.Thread(target=self.serve_forever, daemon=True)
        t.start()
        return t

    def stop(self) -> None:
        self.stopping = True
        self.shutdown()
        self.server_close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Operator GUI without a robot (fake backend)")
    ap.add_argument("--fake", action="store_true", help="serve with the fake backend (no ROS)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    hub = EventHub()
    srv = GuiServer((a.host, a.port), FakeBackend(hub), hub)
    print(f"operator GUI (fake backend) on http://{a.host}:{srv.server_address[1]}/", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.stopping = True
        srv.server_close()


if __name__ == "__main__":
    main()
