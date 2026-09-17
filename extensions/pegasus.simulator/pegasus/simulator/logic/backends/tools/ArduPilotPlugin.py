"""
| File: ArduPilotPlugin.py
| Author: Tomer Tip (tomerT1212@gmail.com)
| Description: File that implements the Mavlink Backend for communication/control with/of the vehicle simulation
| License: BSD-3-Clause. Copyright (c) 2024, Tomer Tip. All rights reserved.
"""
import time
import socket
import struct
import json
import math
from dataclasses import dataclass
import random

class ArduPilotPlugin:
    SERVO_PACKET_SIZE = 40
    SERVO_PACKET_MAGIC = 18458
    
    def __init__(self, fdm_port_in=9002):

        # The address for the flight dynamics model (i.e. this plugin)
        self.fdm_address = '127.0.0.1'
        # The port for the flight dynamics model
        self.fdm_port_in = fdm_port_in

        # FCU address and port are auto detected from receving UDP packet from connected client
        # The address for the SITL flight controller
        self.fcu_address = None
        # The port for the SITL flight controller
        self.fcu_port_out = None

        # Last received frame rate from the ArduPilot controller
        self.fcu_frame_rate = 0
        # Last received frame count from the ArduPilot controller
        self.fcu_frame_count = -1

        # Set to false when Gazebo starts to prevent blocking, true when
        # the ArduPilot controller is detected and online, and false if the
        # connection to the ArduPilot controller times out.
        self.arduPilotOnline = False
        
        # Number of consecutive missed ArduPilot controller messages
        self.connectionTimeoutCount = 0
        #  Max number of consecutive missed ArduPilot controller messages before timeout
        self.connectionTimeoutMaxCount = 5  # Example value
        
        # Set true to enforce lock-step simulation
        self.isLockStep = False
       
        # Last sent JSON string, so we can resend if needed.
        self.json_str: str = ""
        
        # Keep track of controller update sim-time.
        self.last_controller_update_time = 0
        # Keep track of the time the last servo packet was received.
        self.last_servo_packet_recv_time = 0

        # ---- lockstep bookkeeping (2026-09-17) -----------------------------------
        # This plugin used to read ONE servo packet per physics step, wait at most
        # 10 ms of wall time for it, and never drain the socket. Every reply that
        # arrived late -- SITL descheduled past the window, or a duplicate during
        # the boot handshake -- then sat in the queue for the rest of the session,
        # and each one was 1.25 ms of permanent control delay that no log line
        # ever reported. Measured on 2026-09-16/17: ~17 queued packets (21 ms) in
        # a clean session, ~55 (70 ms) in the two whose airframe flew a 1.5 Hz yaw
        # limit cycle. The fix is three things: drain to the newest packet every
        # step, hold the last command on a transient miss instead of zeroing the
        # rotors, and SAY what happened. See tests/test_ardupilot_plugin_lockstep.py.
        # Wall seconds to wait for SITL's reply when online and in lockstep. A
        # stall now costs wall time, never sim-time correctness.
        self.lockstep_wait_s = 0.25
        # Held on a transient miss. The backend zeroes the rotors on (), which on
        # a single missed step is a 1.25 ms thrust cut -- a disturbance of its own.
        self.last_pwm = ()
        self.stat_steps = 0
        self.stat_timeouts = 0
        self.stat_drained = 0
        self.stat_max_backlog = 0
        self.stats_every_s = 10.0
        self._last_stats_wall = time.monotonic()
        self._announced = set()
        
        # Sockets
        self.motor_control_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.motor_control_sock.setblocking(True)
        self.motor_control_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            self.motor_control_sock.bind((self.fdm_address, self.fdm_port_in))
            print(f"Flight dynamics model @ {self.fdm_address}:{self.fdm_port_in}")

        except socket.error as e:
            print(f"Failed to bind with {self.fdm_address}:{self.fdm_port_in}. Aborting plugin.")

        
    def create_state_json(self, sensor_data, sim_time):
        state = {
            "timestamp": sim_time,
            "imu": {
                "gyro": [
                    sensor_data.xgyro,
                    sensor_data.ygyro,
                    sensor_data.zgyro
                ],
                "accel_body": [
                    sensor_data.xacc,
                    sensor_data.yacc,
                    sensor_data.zacc,
                ]
            },  
            "position": [
               sensor_data.sim_position[0], # X
               sensor_data.sim_position[1], # Y
               sensor_data.sim_position[2], # Z
            ],
            "quaternion": [
               sensor_data.sim_attitude[0], # W
               sensor_data.sim_attitude[1], # X
               sensor_data.sim_attitude[2], # Y
               sensor_data.sim_attitude[3], # Z
            ],
            "velocity": [
              sensor_data.sim_velocity_inertial[0], # X
              sensor_data.sim_velocity_inertial[1], # Y
              sensor_data.sim_velocity_inertial[2]  # Z
            ]
        }

        # --- diagnostics/guard: never feed SITL non-finite values. arducopter
        # runs with FP exceptions enabled, so a single NaN/inf in this JSON
        # aborts it with SIGFPE ("fell down suddenly" + core dump). Log the
        # first occurrences (crash-vs-starvation differentiator) and resend
        # the last good state instead of the poisoned one.
        flat = (state["imu"]["gyro"] + state["imu"]["accel_body"] + state["position"]
                + state["quaternion"] + state["velocity"] + [state["timestamp"]])
        if not all(math.isfinite(v) for v in flat):
            self._nonfinite_count = getattr(self, "_nonfinite_count", 0) + 1
            if self._nonfinite_count <= 5:
                print(f"[ArduPilotPlugin] NON-FINITE FDM state #{self._nonfinite_count} at sim_time={sim_time}: {state}")
            if self.json_str:
                return self.json_str
        else:
            # Finite but extreme values (e.g. PhysX contact impulse) pass the
            # isfinite check yet overflow float math inside SITL, which runs
            # with FE_OVERFLOW|FE_DIVBYZERO enabled -> SIGFPE. Observe-only:
            # log the first few so the crash stack (gdb) can be correlated.
            _g = state["imu"]["gyro"]; _a = state["imu"]["accel_body"]; _vel = state["velocity"]
            if (max(abs(x) for x in _g) > 50.0 or max(abs(x) for x in _a) > 500.0
                    or max(abs(x) for x in _vel) > 200.0):
                self._extreme_count = getattr(self, "_extreme_count", 0) + 1
                if self._extreme_count <= 5:
                    print(f"[ArduPilotPlugin] EXTREME FDM values #{self._extreme_count} at sim_time={sim_time}: {state}")

        json_str = json.dumps(state, separators=(',', ':'))
        json_str = "\n" + json_str + "\n"
        json_str = json_str.encode('utf-8')

        self.json_str = json_str
        
        return self.json_str 
        

    def send_state(self):
        # --- diagnostics: a wall-clock gap between sends longer than SITL's
        # ~1 s JSON watchdog is what precedes "No JSON sensor message
        # received" + SIGFPE. Make stalls visible in the autosim log.
        _now = time.monotonic()
        _last = getattr(self, "_last_send_wall", None)
        if _last is not None and _now - _last > 0.7:
            print(f"[ArduPilotPlugin] FDM send stalled {_now - _last:.2f}s wall (sim hitch)")
        self._last_send_wall = _now
        if self.motor_control_sock:
            bytes_sent = self.motor_control_sock.sendto(
                self.json_str,
                (self.fcu_address, self.fcu_port_out)
            )

    def unpack_servo_packet(self, data):
        # Ensure the data length is valid
        if len(data) != self.SERVO_PACKET_SIZE:
            raise ValueError(f"Data length must be {self.SERVO_PACKET_SIZE} bytes, got {len(data)} bytes")
        
        # Define the format string according to the structure for little-endian
        # '<HHI16H' specifies:
        #   - < (little-endian byte order)
        #   - HH (2 uint16_t fields)
        #   - I (1 uint32_t field)
        #   - 16H (16 uint16_t fields for pwm values)
        format_string = '<HHI16H'
        
        try:
            # Unpack the data
            unpacked_data = struct.unpack(format_string, data)
            pkt_magic = unpacked_data[0]
            pkt_frame_rate = unpacked_data[1]
            pkt_frame_count = unpacked_data[2]
            pkt_pwm = unpacked_data[3:]
            
            if pkt_magic != self.SERVO_PACKET_MAGIC:
                print(f"Incorrect protocol magic {pkt_magic}, should be {self.SERVO_PACKET_MAGIC}")
                return None, None, None, None
            
            return pkt_magic, pkt_frame_rate, pkt_frame_count, pkt_pwm
        
        except struct.error as e:
            print(f"Unpacking error: {e}")
            return None, None, None, None

    def drain_unread_packets(self):
        """
        Drains all unread packets from the UDP socket.
        """

        # Set socket to non-blocking mode
        self.motor_control_sock.setblocking(False)
        while True:
            try:
                # Attempt to receive data
                data, (client_addr, client_out) = self.motor_control_sock.recvfrom(self.SERVO_PACKET_SIZE)
                if (self.fcu_address is None) or (self.fcu_port_out is None):
                    self.fcu_address = client_addr
                    self.fcu_port_out = client_out

            except BlockingIOError:
                # No more data to read, exit the loop
                break
            except Exception as e:
                # Handle unexpected exceptions
                print(f"An error occurred while draining packets: {e}")
                break
        
        # Reset the socket to blocking mode
        self.motor_control_sock.setblocking(True)
        print("Drained all packets.")


    def rx_queue_bytes(self):
        """Bytes waiting unread on this plugin's socket, from /proc/net/udp.

        The kernel counts buffer truesize, about 960 B per 40 B servo packet on
        loopback, so divide by ~960 for a packet count. -1 where unreadable.
        This is the number that was 16320 in a session everyone thought was
        healthy.
        """
        try:
            port = self.motor_control_sock.getsockname()[1]
            with open("/proc/net/udp") as f:
                next(f)
                for line in f:
                    fld = line.split()
                    if int(fld[1].split(":")[1], 16) == port:
                        return int(fld[4].split(":")[1], 16)
        except Exception:
            return -1
        return -1

    def stats(self):
        return {
            "steps": self.stat_steps,
            "timeouts": self.stat_timeouts,
            "drained": self.stat_drained,
            "max_backlog": self.stat_max_backlog,
            "queue_bytes": self.rx_queue_bytes(),
        }

    def _maybe_print_stats(self, force=False):
        now = time.monotonic()
        if force or now - self._last_stats_wall >= self.stats_every_s:
            self._last_stats_wall = now
            s = self.stats()
            print("[ArduPilotPlugin] lockstep: steps=%d timeouts=%d drained=%d max_backlog=%d queue_bytes=%d"
                  % (s["steps"], s["timeouts"], s["drained"], s["max_backlog"], s["queue_bytes"]))

    def _announce(self, key, msg):
        if key not in self._announced:
            self._announced.add(key)
            print("[ArduPilotPlugin] " + msg)

    def receive_servo_packet(self):
        self.stat_steps += 1
        # Online and in lockstep: wait for the reply. Otherwise the old short
        # waits, so a dead SITL cannot stall a session that never had one.
        if self.arduPilotOnline and self.isLockStep:
            wait_sec = self.lockstep_wait_s
        else:
            wait_sec = 0.010 if self.arduPilotOnline else 0.001

        try:
            self.motor_control_sock.settimeout(wait_sec)
            data, (client_addr, client_out) = self.motor_control_sock.recvfrom(self.SERVO_PACKET_SIZE)
        except socket.timeout:
            self.stat_timeouts += 1
            if self.arduPilotOnline:
                self._announce("timeout", "first servo timeout after %.0f ms (step %d); holding the last PWM"
                               % (wait_sec * 1000.0, self.stat_steps))
                self.connectionTimeoutCount += 1
                if self.connectionTimeoutCount > self.connectionTimeoutMaxCount:
                    self.connectionTimeoutCount = 0
                    if self.isLockStep:
                        print("Send State from receive_servo_packet")
                        self.send_state()
                    else:
                        print("Socket timeout")
                        self.arduPilotOnline = False
                        print(f"Broken ArduPilot connection, resetting motor control.")
                        return False, ()
            self._maybe_print_stats()
            return False, (self.last_pwm if self.arduPilotOnline else ())

        # DRAIN TO THE NEWEST. SITL's frame_count increments once per state it
        # received, so a queue deeper than one packet means we are behind; the
        # newest packet answers the most recent state SITL has seen and the
        # rest are stale by exactly one physics step each.
        backlog = 0
        self.motor_control_sock.setblocking(False)
        while True:
            try:
                data2, addr2 = self.motor_control_sock.recvfrom(self.SERVO_PACKET_SIZE)
            except (BlockingIOError, socket.timeout):
                break
            except OSError:
                break
            backlog += 1
            data, (client_addr, client_out) = data2, addr2
        if backlog:
            self.stat_drained += backlog
            self.stat_max_backlog = max(self.stat_max_backlog, backlog)
            self._announce("drained", "drained %d stale servo packets in one step (step %d): the queue had built up"
                           % (backlog, self.stat_steps))

        # Track the FCU (ArduPilot SITL) return address on every packet.
        # Latching it only once breaks SITL restarts: replies keep going to
        # the dead ephemeral port of the previous instance (lockstep mode
        # never marks ArduPilot offline, so the stale address is never
        # cleared) and the new arducopter loops on "No JSON sensor message
        # received".
        if (client_addr, client_out) != (self.fcu_address, self.fcu_port_out):
            if self.fcu_address is not None:
                print(f"ArduPilot endpoint changed to {client_addr}:{client_out}")
            self.fcu_address = client_addr
            self.fcu_port_out = client_out

        pkt_magic, pkt_frame_rate, pkt_frame_count, *pkt_pwm = self.unpack_servo_packet(data)
        pkt_pwm = pkt_pwm[0]
        if pkt_magic is None:
            return False, ()

        if not self.arduPilotOnline:
            print(f"Connected to ArduPilot controller @ {self.fcu_address}:{self.fcu_port_out}")
            self.arduPilotOnline = True

        self.fcu_frame_rate = pkt_frame_rate

        if pkt_frame_count < self.fcu_frame_count:
            print("ArduPilot controller has reset")
        elif pkt_frame_count == self.fcu_frame_count:
            # The boot handshake: SITL re-sends its servos once a second until it
            # sees a state. Answer it, and hold the command we already have.
            print("Duplicate input frame")
            if self.isLockStep:
                self.send_state()
            return False, self.last_pwm
        elif pkt_frame_count != self.fcu_frame_count + 1 and self.arduPilotOnline and backlog == 0:
            # A gap with NOTHING drained is a packet lost in flight, which is
            # news; a gap after draining is just the stale packets we skipped.
            print(f"Missed {pkt_frame_count - self.fcu_frame_count} input frames")

        self.fcu_frame_count = pkt_frame_count
        self.connectionTimeoutCount = 0
        self.last_pwm = tuple(pkt_pwm)
        self._maybe_print_stats()
        return True, pkt_pwm

    def pre_update(self, sim_time):
        # Update the control surfaces
        recieved, pwms = self.receive_servo_packet()
        if recieved:
            self.last_servo_packet_recv_time = sim_time

        return recieved, pwms

    def post_update(self, sensor_data, sim_time):
        if sim_time > self.last_controller_update_time and self.arduPilotOnline:
            self.last_controller_update_time = sim_time
            self.create_state_json(sim_time=self.last_controller_update_time, sensor_data=sensor_data)
            self.send_state()

    # Mock Sensor Data
    @dataclass
    class SensorData:
        sim_position = [
            -6.874587183616472e-12 + random.gauss(0, 0.01),
            -1.6699334495870352e-12 + random.gauss(0, 0.01),
            -0.1949994162458527 + random.gauss(0, 0.01)
        ]
        sim_attitude = [
            1,
            -4.281847537232883e-12 + random.gauss(0, 0.001),
            1.7627199589076353e-11 + random.gauss(0, 0.001),
            -4.281847537232883e-12 + random.gauss(0, 0.001)
        ]
        sim_velocity_inertial = [
            4.4028248386528135e-12 + random.gauss(0, 0.01),
            5.46182226507895e-12 + random.gauss(0, 0.01),
            5.454422342398644e-18 + random.gauss(0, 0.01)
        ]
        xgyro: float = 4.4028248386528135e-12
        ygyro: float = 5.46182226507895e-12
        zgyro: float = 5.454422342398644e-18
        xacc: float = -1.521417774123255e-9
        yacc: float = 1.976595291657851e-9
        zacc: float = -9.80000000000001


if __name__ == '__main__':
    ap = ArduPilotPlugin()
    ap.drain_unread_packets()

    while True:
        ap.pre_update()
        ap.post_update(sensor_data=ap.SensorData())
        time.sleep(0.01)

