#!/usr/bin/env bash
# bake_trim.sh — make the steering trim permanent.
#
# Trim is a per-command steering offset applied by web_pilot. It works, but it costs throw
# on one side (a right command plus a left trim never reaches full right lock) and it lives
# outside the config, so autonomy and a bare manual bring-up do not get it. This converts
# whatever trim you dialled in with Q/E into the servo centre in vesc.yaml, then zeroes the
# trim -- same straight-line behaviour, full throw restored, and every mode gets it.
#
#   ./bake_trim.sh          apply
#   ./bake_trim.sh --dry    show the numbers only
#
# servo = gain * steering_angle + offset, so adding trim t (rad) to every command is the
# same as shifting the centre by gain*t. Hence: offset_new = offset_old + gain*t.
set -e
V=$HOME/f1tenth_ws/src/f1tenth_system/f1tenth_stack/config/vesc.yaml
J=$HOME/f1tenth_ws/src/f1tenth_system/f1tenth_stack/config/joy_teleop_f310.yaml
T=$HOME/.atlascar_trim.json
DRY=0; [ "${1:-}" = "--dry" ] && DRY=1

read NEW_OFFSET REPORT < <(python3 - "$V" "$J" "$T" <<'PY'
import json, re, sys
vesc, joy, trimf = sys.argv[1:4]
v = open(vesc).read()
gain   = float(re.search(r'steering_angle_to_servo_gain:\s*(-?[\d.]+)', v).group(1))
offset = float(re.search(r'steering_angle_to_servo_offset:\s*(-?[\d.]+)', v).group(1))
smin   = float(re.search(r'servo_min:\s*([\d.]+)', v).group(1))
smax   = float(re.search(r'servo_max:\s*([\d.]+)', v).group(1))
m = re.search(r'drive-steering_angle:.*?scale:\s*([\d.]+)', open(joy).read(), re.S)
scale = float(m.group(1)) if m else 0.34
try:    trim = float(json.load(open(trimf))['steer_trim'])
except Exception: trim = 0.0
t_rad = trim * scale                      # trim in axis units -> radians of steering
new   = offset + gain * t_rad
left  = (new - smin) / abs(gain)          # achievable throw each way, radians
right = (smax - new) / abs(gain)
print('%.4f' % new,
      'trim=%.3f(%.4f rad) gain=%.4f offset %.4f -> %.4f | throw left %.3f rad right %.3f rad'
      % (trim, t_rad, gain, offset, new, left, right))
PY
)
echo "$REPORT"
[ "$DRY" = 1 ] && exit 0
[ "$NEW_OFFSET" = "" ] && { echo "could not compute an offset"; exit 1; }

cp "$V" "$V.bak.$(date +%s)"
sed -i "s/^\( *steering_angle_to_servo_offset:\) *[-0-9.]*/\1 $NEW_OFFSET/" "$V"
echo '{"steer_trim": 0.0}' > "$T"
grep -E "steering_angle_to_servo_offset" "$V"
cd "$HOME/f1tenth_ws" && source /opt/ros/humble/setup.bash && \
    colcon build --packages-select f1tenth_stack 2>&1 | grep -E "Finished|failed"
bash "$HOME/restart_remote.sh"
echo "done — drive it: it should track straight with trim at 0. If it still pulls, re-trim with Q/E and run this again."
