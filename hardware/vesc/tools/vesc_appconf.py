#!/usr/bin/env python3
"""COMM_GET_APPCONF (17) head decode, FW 6.06 layout."""
import importlib.util, struct, time
spec = importlib.util.spec_from_file_location("vd", "/tmp/vesc_direct_lib.py"); vd = importlib.util.module_from_spec(spec); spec.loader.exec_module(vd)
exec(open("/tmp/vesc_mcconf3.py").read().split("# (name, type, scale)")[0].split("COMM_GET_MCCONF = 14")[1])  # read_any_packet
COMM_GET_APPCONF = 17
ser = vd.serial.Serial(vd.PORT, 115200, timeout=0.05)
ser.reset_input_buffer(); ser.write(vd.packet(bytes([COMM_GET_APPCONF]))); p = read_any_packet(ser, COMM_GET_APPCONF); ser.close()
if not p: print("no appconf reply"); raise SystemExit(1)
o = 1
sig = struct.unpack(">I", p[o:o+4])[0]; o += 4
controller_id = p[o]; o += 1
timeout_msec = struct.unpack(">I", p[o:o+4])[0]; o += 4
timeout_brake = struct.unpack(">f", p[o:o+4])[0]; o += 4
o += 4  # can_status_rate_1/2
can_msgs_r1, can_msgs_r2, can_baud, pairing_done, perm_uart, shutdown_mode, can_mode, uavcan_idx, uavcan_raw = p[o:o+9]; o += 9
o += 4  # uavcan_raw_rpm_max
uavcan_cur_mode, servo_out_enable, kill_sw_mode, app_to_use, ppm_ctrl_type = p[o:o+5]; o += 5
ppm_pid_max_erpm = struct.unpack(">f", p[o:o+4])[0]; o += 4
apps = {0: "NONE", 1: "PPM", 2: "ADC", 3: "UART", 4: "PPM_UART", 5: "ADC_UART", 6: "NUNCHUK", 7: "NRF", 8: "CUSTOM", 9: "BALANCE", 10: "PAS", 11: "ADC_PAS"}
kill = {0: "DISABLED", 1: "PPM_LOW", 2: "PPM_HIGH", 3: "ADC2_LOW", 4: "ADC2_HIGH"}
shut = {0: "ALWAYS_OFF", 1: "ALWAYS_ON", 2: "TOGGLE_BUTTON_ONLY", 3: "OFF_AFTER_10S", 4: "OFF_AFTER_1M", 5: "OFF_AFTER_5M", 6: "OFF_AFTER_10M", 7: "OFF_AFTER_30M", 8: "OFF_AFTER_1H", 9: "OFF_AFTER_5H"}
ppm_ct = {0: "NONE", 1: "CURRENT", 2: "CURRENT_NOREV", 3: "CURRENT_NOREV_BRAKE", 4: "DUTY", 5: "DUTY_NOREV", 6: "PID", 7: "PID_NOREV", 8: "CURRENT_HYST_NOREV_BRAKE", 9: "CURRENT_SMART_REV", 10: "PID_POSITION_180", 11: "PID_POSITION_360"}
print(f"appconf bytes={len(p)} signature={sig} controller_id={controller_id}")
print(f"  timeout_msec            = {timeout_msec}   (default 1000)")
print(f"  timeout_brake_current   = {timeout_brake:.2f} A")
print(f"  shutdown_mode           = {shut.get(shutdown_mode, shutdown_mode)}")
print(f"  servo_out_enable        = {servo_out_enable}")
print(f"  kill_sw_mode            = {kill.get(kill_sw_mode, kill_sw_mode)}")
print(f"  app_to_use              = {apps.get(app_to_use, app_to_use)}")
print(f"  ppm ctrl_type           = {ppm_ct.get(ppm_ctrl_type, ppm_ctrl_type)}   ppm pid_max_erpm={ppm_pid_max_erpm:.0f}")
print(f"  permanent_uart_enabled  = {perm_uart}   can_mode={can_mode}   pairing_done={pairing_done}")
