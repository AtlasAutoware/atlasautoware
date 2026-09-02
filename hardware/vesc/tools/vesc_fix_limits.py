#!/usr/bin/env python3
"""Set l_current_max/min, l_abs_current_max, l_slow_abs_current in mcconf via COMM_SET_MCCONF (13). Saves original."""
import importlib.util, struct, time
spec = importlib.util.spec_from_file_location("vd", "/tmp/vesc_direct_lib.py"); vd = importlib.util.module_from_spec(spec); spec.loader.exec_module(vd)
src = open("/tmp/vesc_mcconf3.py").read()
exec(src.split("ser = vd.serial.Serial")[0])   # S (layout), read_any_packet, COMM_GET_MCCONF
COMM_SET_MCCONF = 13
def packet_long(payload):
    n = len(payload); return bytes([3, n >> 8, n & 0xFF]) + payload + struct.pack(">H", vd.crc16(payload)) + b"\x03"
# compute byte offsets of every field
off = {}; o = 1
for name, t, sc in S:
    off[name] = o; o += {"u8": 1, "u16": 2, "f16": 2, "u32": 4, "i32": 4, "f32": 4}[t]
ser = vd.serial.Serial(vd.PORT, 115200, timeout=0.05)
ser.reset_input_buffer(); ser.write(vd.packet(bytes([COMM_GET_MCCONF]))); p = read_any_packet(ser, COMM_GET_MCCONF)
assert p and p[0] == COMM_GET_MCCONF
open("/tmp/mcconf_original.bin", "wb").write(p)
f = lambda n: struct.unpack(">f", p[off[n]:off[n]+4])[0]
print(f"before: l_current_max={f('l_current_max'):.1f} l_current_min={f('l_current_min'):.1f} l_abs_current_max={f('l_abs_current_max'):.1f} l_slow_abs_current={p[off['l_slow_abs_current']]}")
assert 95 < f('l_current_max') < 105 and 145 < f('l_abs_current_max') < 155, "unexpected values; not writing"
b = bytearray(p)
b[off['l_current_max']:off['l_current_max']+4] = struct.pack(">f", 60.0)
b[off['l_current_min']:off['l_current_min']+4] = struct.pack(">f", -60.0)
b[off['l_abs_current_max']:off['l_abs_current_max']+4] = struct.pack(">f", 120.0)
b[off['l_slow_abs_current']] = 1
ser.reset_input_buffer(); ser.write(packet_long(bytes([COMM_SET_MCCONF]) + bytes(b[1:])))
ack = read_any_packet(ser, COMM_SET_MCCONF, timeout=4.0); print("write ack:", "OK" if ack else "no ack")
time.sleep(1.5)
ser.reset_input_buffer(); ser.write(vd.packet(bytes([COMM_GET_MCCONF]))); p = read_any_packet(ser, COMM_GET_MCCONF); ser.close()
print(f"after:  l_current_max={f('l_current_max'):.1f} l_current_min={f('l_current_min'):.1f} l_abs_current_max={f('l_abs_current_max'):.1f} l_slow_abs_current={p[off['l_slow_abs_current']]}")
