"""The physics<->SITL servo exchange, tested against a real UDP peer.

WHY THIS FILE EXISTS. The plugin reads ONE servo packet per physics step and
never drains its socket. Every packet that arrives late -- a 10 ms timeout while
SITL was descheduled, a duplicate during the boot handshake -- therefore sits in
the queue forever, and each one is 1.25 ms of control delay for the rest of the
session. Measured 2026-09-16/17: ~17 packets (21 ms) in a clean session, ~55
(70 ms) in the two that flew with a 1.5 Hz yaw limit cycle. No log line ever
said so. These tests pin the three properties that stop that:

  1. a backlog is drained to the NEWEST packet in one step;
  2. a transient timeout HOLDS the last motor command instead of zeroing it;
  3. the plugin can say how deep its queue is, and counts what it dropped.

Stdlib only, real sockets, no Isaac: runs anywhere with pytest.
"""
import pathlib
import socket
import struct
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "extensions/pegasus.simulator/pegasus/simulator/logic/backends/tools"))
from ArduPilotPlugin import ArduPilotPlugin  # noqa: E402

MAGIC = 18458


def servo(frame_count, pwm=1500):
    """A SITL servo packet: the '<HHI16H' layout the plugin unpacks."""
    return struct.pack("<HHI16H", MAGIC, 800, frame_count, *([pwm] * 16))


class FakeSITL:
    """The other end of the plugin's socket. Sends servo packets, receives state."""

    def __init__(self, plugin_port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 0))
        self.plugin = ("127.0.0.1", plugin_port)

    def send(self, frame_count, pwm=1500):
        self.sock.sendto(servo(frame_count, pwm), self.plugin)

    def close(self):
        self.sock.close()


def online_plugin():
    """A plugin that has completed the handshake: one packet received, SITL online."""
    p = ArduPilotPlugin(fdm_port_in=0)  # 0: an ephemeral port, so tests never collide
    port = p.motor_control_sock.getsockname()[1]
    sitl = FakeSITL(port)
    sitl.send(1, pwm=1100)
    time.sleep(0.02)
    received, pwm = p.pre_update(sim_time=0.001)
    assert received and p.arduPilotOnline and tuple(pwm) == (1100,) * 16
    return p, sitl


def test_backlog_is_drained_to_the_newest_packet_in_one_step():
    p, sitl = online_plugin()
    try:
        # SITL got ahead by 40 frames (a stall on our side). Frame N carries pwm
        # 1000+N so the test can tell WHICH packet came back.
        for fc in range(2, 42):
            sitl.send(fc, pwm=1000 + fc)
        time.sleep(0.05)
        received, pwm = p.pre_update(sim_time=0.002)
        assert received
        assert tuple(pwm) == (1041,) * 16, "must return the NEWEST frame, not the oldest"
        st = p.stats()
        assert st["drained"] == 39, "39 stale packets were skipped to reach the newest"
        assert st["max_backlog"] >= 39
        # And the queue is now empty: nothing left to leak into the next step.
        assert p.rx_queue_bytes() == 0
    finally:
        sitl.close()


def test_transient_timeout_holds_the_last_command_instead_of_zeroing():
    p, sitl = online_plugin()
    try:
        p.isLockStep = True
        p.lockstep_wait_s = 0.02  # keep the test fast; the default is longer
        received, pwm = p.pre_update(sim_time=0.002)  # nothing sent: a miss
        assert not received
        # The backend zeroes the rotors when it gets (), which on a one-step
        # miss is a 1.25 ms thrust cut -- a disturbance of its own. Hold instead.
        assert tuple(pwm) == (1100,) * 16
        assert p.stats()["timeouts"] == 1
    finally:
        sitl.close()


def test_offline_plugin_still_returns_nothing_on_timeout():
    # Before the handshake there IS no last command; zero is the right answer.
    p = ArduPilotPlugin(fdm_port_in=0)
    received, pwm = p.pre_update(sim_time=0.001)
    assert not received and pwm == ()


def test_lockstep_waits_for_a_late_reply_rather_than_counting_a_miss():
    p, sitl = online_plugin()
    try:
        p.isLockStep = True
        p.lockstep_wait_s = 0.5
        threading.Timer(0.1, lambda: sitl.send(2, pwm=1200)).start()
        t0 = time.monotonic()
        received, pwm = p.pre_update(sim_time=0.002)
        dt = time.monotonic() - t0
        assert received and tuple(pwm) == (1200,) * 16
        assert dt >= 0.09, "the step waited for the reply (it did not time out at 10 ms)"
        assert p.stats()["timeouts"] == 0
    finally:
        sitl.close()


def test_duplicate_frame_holds_the_last_command():
    p, sitl = online_plugin()
    try:
        sitl.send(1, pwm=1100)  # SITL re-sent frame 1 (the boot handshake does this)
        time.sleep(0.02)
        received, pwm = p.pre_update(sim_time=0.002)
        assert not received
        # The old code returned [] here, which the backend does not treat as ().
        assert tuple(pwm) == (1100,) * 16
    finally:
        sitl.close()


def test_queue_depth_is_observable():
    p, sitl = online_plugin()
    try:
        assert p.rx_queue_bytes() == 0
        for fc in range(2, 7):
            sitl.send(fc)
        time.sleep(0.05)
        assert p.rx_queue_bytes() > 0, "five unread packets must show as a non-zero queue"
        p.pre_update(sim_time=0.002)
        assert p.rx_queue_bytes() == 0
        st = p.stats()
        assert set(st) >= {"steps", "timeouts", "drained", "max_backlog", "queue_bytes"}
    finally:
        sitl.close()
