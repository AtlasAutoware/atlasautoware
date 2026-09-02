#!/usr/bin/env python3
"""Direct VESC test over USB (no ROS): FW version, telemetry, servo sweep, 2 s of SET_RPM."""
import serial, struct, time, sys

PORT = "/dev/sensors/vesc"
COMM_FW_VERSION, COMM_GET_VALUES, COMM_SET_CURRENT, COMM_SET_RPM, COMM_SET_SERVO_POS = 0, 4, 6, 8, 12

def crc16(data):
    crc = 0
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if crc & 0x8000 else (crc << 1)
            crc &= 0xFFFF
    return crc

def packet(payload):
    return bytes([2, len(payload)]) + payload + struct.pack(">H", crc16(payload)) + b"\x03"

def read_packet(ser, want_id, timeout=0.5):
    t0 = time.time(); buf = b""
    while time.time() - t0 < timeout:
        buf += ser.read(256)
        i = buf.find(b"\x02")
        while i != -1 and len(buf) >= i + 2:
            n = buf[i + 1]
            if len(buf) < i + 2 + n + 3: break
            payload = buf[i + 2:i + 2 + n]
            if payload and payload[0] == want_id: return payload
            buf = buf[i + 1:]; i = buf.find(b"\x02")
    return None

def get_values(ser):
    ser.reset_input_buffer(); ser.write(packet(bytes([COMM_GET_VALUES])))
    p = read_packet(ser, COMM_GET_VALUES)
    if not p or len(p) < 54: return None
    temp_fet = struct.unpack(">h", p[1:3])[0] / 10
    i_motor = struct.unpack(">i", p[5:9])[0] / 100
    i_in = struct.unpack(">i", p[9:13])[0] / 100
    duty = struct.unpack(">h", p[21:23])[0] / 1000
    rpm = struct.unpack(">i", p[23:27])[0]
    v_in = struct.unpack(">h", p[27:29])[0] / 10
    fault = p[53]
    return dict(temp_fet=temp_fet, i_motor=i_motor, i_in=i_in, duty=duty, rpm=rpm, v_in=v_in, fault=fault)
