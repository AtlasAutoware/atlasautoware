#!/usr/bin/env python3
"""Exact FW 6.06 mcconf decoder (layout from bldc release_6_06 confgenerator.c)."""
import importlib.util, struct, time
spec = importlib.util.spec_from_file_location("vd", "/tmp/vesc_direct_lib.py"); vd = importlib.util.module_from_spec(spec); spec.loader.exec_module(vd)
COMM_GET_MCCONF = 14
def read_any_packet(ser, want_id, timeout=1.5):
    t0 = time.time(); buf = b""
    while time.time() - t0 < timeout:
        buf += ser.read(4096)
        for start in (2, 3):
            i = buf.find(bytes([start]))
            while i != -1:
                if start == 2 and len(buf) >= i + 2: n = buf[i+1]; hdr = 2
                elif start == 3 and len(buf) >= i + 3: n = (buf[i+1] << 8) | buf[i+2]; hdr = 3
                else: break
                if len(buf) < i + hdr + n + 3: break
                p = buf[i+hdr:i+hdr+n]
                if p and p[0] == want_id: return p
                i = buf.find(bytes([start]), i+1)
    return None

# (name, type, scale) ; types: u32 u8 u16 i32 f32 f16
S = []
def add(t, names, scale=1):
    for n in names.split(): S.append((n, t, scale))
add("u32", "signature"); add("u8", "pwm_mode comm_mode motor_type sensor_mode")
add("f32", "l_current_max l_current_min l_in_current_max l_in_current_min")
add("f16", "l_in_current_map_start l_in_current_map_filter", 10000)
add("f32", "l_abs_current_max l_min_erpm l_max_erpm"); add("f16", "l_erpm_start", 10000)
add("f32", "l_max_erpm_fbrake l_max_erpm_fbrake_cc")
add("f16", "l_min_vin l_max_vin l_battery_cut_start l_battery_cut_end l_battery_regen_cut_start l_battery_regen_cut_end", 10)
add("u8", "l_slow_abs_current l_temp_fet_start l_temp_fet_end l_temp_motor_start l_temp_motor_end")
add("f16", "l_temp_accel_dec l_min_duty l_max_duty", 10000); add("f32", "l_watt_max l_watt_min")
add("f16", "l_current_max_scale l_current_min_scale l_duty_start", 10000)
add("f32", "sl_min_erpm sl_min_erpm_cycle_int_limit sl_max_fullbreak_current_dir_change")
add("f16", "sl_cycle_int_limit", 10); add("f16", "sl_phase_advance_at_br", 10000)
add("f32", "sl_cycle_int_rpm_br sl_bemf_coupling_k")
add("u8", " ".join(f"hall_table{i}" for i in range(8)))
add("f32", "hall_sl_erpm foc_current_kp foc_current_ki foc_f_zv foc_dt_us"); add("u8", "foc_encoder_inverted")
add("f32", "foc_encoder_offset foc_encoder_ratio"); add("u8", "foc_sensor_mode")
add("f32", "foc_pll_kp foc_pll_ki foc_motor_l foc_motor_ld_lq_diff foc_motor_r foc_motor_flux_linkage foc_observer_gain foc_observer_gain_slow")
add("f16", "foc_observer_offset", 1000); add("f32", "foc_duty_dowmramp_kp foc_duty_dowmramp_ki")
add("f16", "foc_start_curr_dec", 10000); add("f32", "foc_start_curr_dec_rpm foc_openloop_rpm")
add("f16", "foc_openloop_rpm_low foc_d_gain_scale_start foc_d_gain_scale_max_mod", 1000)
add("f16", "foc_sl_openloop_hyst foc_sl_openloop_time_lock foc_sl_openloop_time_ramp foc_sl_openloop_time foc_sl_openloop_boost_q foc_sl_openloop_max_q", 100)
add("u8", " ".join(f"foc_hall_table{i}" for i in range(8)))
add("f32", "foc_hall_interp_erpm foc_sl_erpm_start foc_sl_erpm")
add("u8", "foc_control_sample_mode foc_current_sample_mode foc_sat_comp_mode"); add("f16", "foc_sat_comp", 1000)
add("u8", "foc_temp_comp"); add("f16", "foc_temp_comp_base_temp", 100); add("f16", "foc_current_filter_const", 10000)
add("u8", "foc_cc_decoupling foc_observer_type foc_hfi_amb_mode"); add("f16", "foc_hfi_amb_current", 10); add("u8", "foc_hfi_amb_tres")
add("f16", "foc_hfi_voltage_start foc_hfi_voltage_run foc_hfi_voltage_max", 10); add("f16", "foc_hfi_gain foc_hfi_max_err", 1000)
add("f16", "foc_hfi_hyst", 100); add("f32", "foc_sl_erpm_hfi"); add("u16", "foc_hfi_start_samples"); add("f32", "foc_hfi_obs_ovr_sec")
add("u8", "foc_hfi_samples foc_offsets_cal_mode"); add("f32", "foc_offsets_current0 foc_offsets_current1 foc_offsets_current2")
add("f16", "foc_offsets_voltage0 foc_offsets_voltage1 foc_offsets_voltage2 foc_offsets_voltage_undriven0 foc_offsets_voltage_undriven1 foc_offsets_voltage_undriven2", 10000)
add("u8", "foc_phase_filter_enable foc_phase_filter_disable_fault"); add("f32", "foc_phase_filter_max_erpm"); add("u8", "foc_mtpa_mode")
add("f32", "foc_fw_current_max"); add("f16", "foc_fw_duty_start", 10000); add("f16", "foc_fw_ramp_time", 1000); add("f16", "foc_fw_q_current_factor", 10000)
add("u8", "foc_speed_soure foc_short_ls_on_zero_duty"); add("f16", "foc_overmod_factor", 10000); add("u8", "sp_pid_loop_rate")
add("f32", "s_pid_kp s_pid_ki s_pid_kd"); add("f16", "s_pid_kd_filter", 10000); add("f32", "s_pid_min_erpm"); add("u8", "s_pid_allow_braking")
add("f32", "s_pid_ramp_erpms_s"); add("u8", "s_pid_speed_source")
add("f32", "p_pid_kp p_pid_ki p_pid_kd p_pid_kd_proc"); add("f16", "p_pid_kd_filter", 10000); add("f32", "p_pid_ang_div"); add("f16", "p_pid_gain_dec_angle", 10); add("f32", "p_pid_offset")
add("f16", "cc_startup_boost_duty", 10000); add("f32", "cc_min_current cc_gain"); add("f16", "cc_ramp_step_max", 10000)
add("i32", "m_fault_stop_time_ms"); add("f16", "m_duty_ramp_step", 10000); add("f32", "m_current_backoff_gain"); add("u32", "m_encoder_counts")
add("f16", "m_encoder_sin_amp m_encoder_cos_amp m_encoder_sin_offset m_encoder_cos_offset m_encoder_sincos_filter_constant m_encoder_sincos_phase_correction", 1000)
add("u8", "m_sensor_port_mode m_invert_direction m_drv8301_oc_mode m_drv8301_oc_adj")
add("f32", "m_bldc_f_sw_min m_bldc_f_sw_max m_dc_f_sw m_ntc_motor_beta"); add("u8", "m_out_aux_mode m_motor_temp_sens_type")
add("f32", "m_ptc_motor_coeff"); add("f16", "m_ntcx_ptcx_res", 0.1); add("f16", "m_ntcx_ptcx_temp_base", 10)
add("u8", "m_hall_extra_samples m_batt_filter_const si_motor_poles"); add("f32", "si_gear_ratio si_wheel_diameter")
add("u8", "si_battery_type si_battery_cells"); add("f32", "si_battery_ah si_motor_nl_current")

ser = vd.serial.Serial(vd.PORT, 115200, timeout=0.05)
ser.reset_input_buffer(); ser.write(vd.packet(bytes([COMM_GET_MCCONF]))); p = read_any_packet(ser, COMM_GET_MCCONF); ser.close()
o = 1; V = {}
for name, t, sc in S:
    if t == "u8": V[name] = p[o]; o += 1
    elif t == "u16": V[name] = struct.unpack(">H", p[o:o+2])[0]; o += 2
    elif t == "u32": V[name] = struct.unpack(">I", p[o:o+4])[0]; o += 4
    elif t == "i32": V[name] = struct.unpack(">i", p[o:o+4])[0]; o += 4
    elif t == "f32": V[name] = struct.unpack(">f", p[o:o+4])[0]; o += 4
    elif t == "f16": V[name] = struct.unpack(">h", p[o:o+2])[0] / sc; o += 2
print(f"decoded {o} of {len(p)} bytes")
show = """motor_type sensor_mode foc_sensor_mode l_current_max l_current_min l_in_current_max l_in_current_min l_abs_current_max
l_slow_abs_current l_min_erpm l_max_erpm l_min_duty l_max_duty l_temp_fet_start l_temp_fet_end l_temp_motor_start l_temp_motor_end l_min_vin l_max_vin
foc_motor_l foc_motor_ld_lq_diff foc_motor_r foc_motor_flux_linkage foc_observer_gain foc_current_kp foc_current_ki foc_sl_erpm foc_sl_erpm_start foc_hall_interp_erpm
foc_hall_table0 foc_hall_table1 foc_hall_table2 foc_hall_table3 foc_hall_table4 foc_hall_table5 foc_hall_table6 foc_hall_table7
sp_pid_loop_rate s_pid_kp s_pid_ki s_pid_kd s_pid_kd_filter s_pid_min_erpm s_pid_allow_braking s_pid_ramp_erpms_s s_pid_speed_source
p_pid_kp m_invert_direction m_sensor_port_mode m_motor_temp_sens_type m_ntc_motor_beta si_motor_poles si_gear_ratio si_wheel_diameter si_battery_cells""".split()
for n in show: print(f"  {n:28s} {V[n]}")
