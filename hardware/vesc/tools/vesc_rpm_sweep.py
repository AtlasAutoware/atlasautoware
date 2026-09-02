#!/usr/bin/env python3
"""SET_RPM at several setpoints, 1.2 s each, report duty/current/rpm. Wheels off the ground."""
import importlib.util, struct, time
spec = importlib.util.spec_from_file_location("vd", "/tmp/vesc_direct_lib.py"); vd = importlib.util.module_from_spec(spec); spec.loader.exec_module(vd)
ser = vd.serial.Serial(vd.PORT, 115200, timeout=0.05)
print("idle:", vd.get_values(ser))
for target in [1500, 3500, 6000, 9000]:
    t0 = time.time(); last = None
    while time.time() - t0 < 1.2:
        ser.write(vd.packet(bytes([vd.COMM_SET_RPM]) + struct.pack(">i", target)))
        last = vd.get_values(ser); time.sleep(0.1)
    print(f"SET_RPM {target:5d} -> duty={last['duty']:+.3f} i_motor={last['i_motor']:+.1f}A rpm={last['rpm']:6d} fault={last['fault']}" if last else f"SET_RPM {target}: no reply", flush=True)
ser.write(vd.packet(bytes([vd.COMM_SET_CURRENT]) + struct.pack(">i", 0))); time.sleep(0.5)
print("stopped:", vd.get_values(ser)); ser.close()
