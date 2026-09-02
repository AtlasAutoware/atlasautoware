#!/usr/bin/env python3
"""Read appconf, set servo_out_enable=1 and app_to_use=UART(3), write back with COMM_SET_APPCONF (16), verify."""
import importlib.util, struct, time
spec = importlib.util.spec_from_file_location("vd", "/tmp/vesc_direct_lib.py"); vd = importlib.util.module_from_spec(spec); spec.loader.exec_module(vd)
exec(open("/tmp/vesc_mcconf3.py").read().split("# (name, type, scale)")[0].split("COMM_GET_MCCONF = 14")[1])  # read_any_packet
COMM_SET_APPCONF, COMM_GET_APPCONF = 16, 17
ser = vd.serial.Serial(vd.PORT, 115200, timeout=0.05)
ser.reset_input_buffer(); ser.write(vd.packet(bytes([COMM_GET_APPCONF]))); p = read_any_packet(ser, COMM_GET_APPCONF)
assert p and p[0] == COMM_GET_APPCONF, "no appconf"
open("/tmp/appconf_original.bin", "wb").write(p)
# offsets (after id byte): sig 1-4, controller_id 5, timeout 6-9, brake 10-13, can_rate 14-17, 9 u8 18-26, uavcan_rpm 27-30,
# uavcan_cur_mode 31, servo_out_enable 32, kill_sw_mode 33, app_to_use 34
i_servo, i_app = 32, 34
print(f"before: servo_out_enable={p[i_servo]} app_to_use={p[i_app]} kill_sw={p[33]} (expect 0, 4, 0)")
assert p[i_servo] in (0, 1) and p[i_app] == 4 and p[33] == 0, "layout sanity check failed, not writing"
blob = bytearray(p[1:]); blob[i_servo - 1] = 1; blob[i_app - 1] = 3
def packet_long(payload):
    n = len(payload)
    return bytes([3, n >> 8, n & 0xFF]) + payload + struct.pack(">H", vd.crc16(payload)) + b"\x03"
ser.reset_input_buffer(); ser.write(packet_long(bytes([COMM_SET_APPCONF]) + bytes(blob)))
ack = read_any_packet(ser, COMM_SET_APPCONF, timeout=3.0)
print("write ack:", "OK" if ack else "no ack (may still have applied)")
time.sleep(1.5)
ser.reset_input_buffer(); ser.write(vd.packet(bytes([COMM_GET_APPCONF]))); q = read_any_packet(ser, COMM_GET_APPCONF)
print(f"after:  servo_out_enable={q[i_servo]} app_to_use={q[i_app]}  (want 1, 3=UART)")
ser.close()
