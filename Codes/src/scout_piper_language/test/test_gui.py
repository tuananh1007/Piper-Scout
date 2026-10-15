"""Operator GUI server: JSON API and server-sent events with the fake backend,
then the page in headless Chromium (Playwright, Node) when available."""

import json
import os
import shutil
import subprocess
import threading
import time
import urllib.request

import pytest

from scout_piper_language.gui_server import EventHub, FakeBackend, GuiServer

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture()
def server():
    hub = EventHub()
    srv = GuiServer(("127.0.0.1", 0), FakeBackend(hub), hub)
    srv.start()
    yield srv, f"http://127.0.0.1:{srv.server_address[1]}/"
    srv.stop()


def _post(url, body=None):
    req = urllib.request.Request(url, data=json.dumps(body or {}).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def test_api_and_events(server):
    srv, url = server
    with urllib.request.urlopen(url, timeout=5) as r:
        assert b"Piper-Scout" in r.read()
    with urllib.request.urlopen(url + "api/state", timeout=5) as r:
        assert json.loads(r.read())["connected"]
    got = []

    def listen():
        with urllib.request.urlopen(url + "api/events", timeout=10) as r:
            for raw in r:
                line = raw.decode().strip()
                if line.startswith("event:"):
                    got.append(line.split(":", 1)[1].strip())
                if "proposal" in got and "say" in got:
                    return
    th = threading.Thread(target=listen, daemon=True)
    th.start()
    time.sleep(0.3)
    assert _post(url + "api/instruction", {"text": "grasp the red flower"})["ok"]
    th.join(5)
    assert "proposal" in got and "say" in got
    assert _post(url + "api/drive", {"v": 0.1})["ok"] is False          # not enabled
    assert _post(url + "api/drive_enable", {"enable": True})["enabled"]
    assert _post(url + "api/drive", {"v": 0.1})["ok"]
    assert _post(url + "api/abort")["stopped"]
    assert srv.backend.drive_enabled is False and srv.backend.stops == 1
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(url + "api/voice")
    assert e.value.code == 501                                           # no speech recognition configured


def _node():
    node = shutil.which("node")
    pw = "/opt/node22/lib/node_modules/playwright"
    return node if node and (os.path.isdir(pw) or subprocess.run([node, "-e", "require('playwright')"],
                                                                 capture_output=True).returncode == 0) else None


@pytest.mark.skipif(_node() is None, reason="node + playwright not available")
def test_page_in_headless_chromium(server):
    _, url = server
    env = dict(os.environ)
    if os.path.exists("/opt/pw-browsers/chromium"):
        env.setdefault("PW_CHROMIUM", "/opt/pw-browsers/chromium")
    shots = os.environ.get("GUI_SCREENSHOTS", "")
    out = subprocess.run([_node(), os.path.join(HERE, "gui_e2e.mjs"), url, shots], capture_output=True, text=True,
                         timeout=120, env=env)
    print(out.stdout, out.stderr)
    assert out.returncode == 0, out.stdout + out.stderr
