#!/usr/bin/env bash
# carnet.sh — pick how the car gets on a network (run on the Jetson, needs sudo).
#
#   carnet.sh status                      what we are on, our IPs, signal
#   carnet.sh hotspot                     be the AP: SSID AtlasCar, 5 GHz ch 36, car = 10.42.0.1   (short range, fast)
#   carnet.sh hotspot24                   be the AP on 2.4 GHz ch 6 (about 2x the range of 5 GHz, less bandwidth)
#   carnet.sh client <SSID> [password]    join an existing WiFi / mesh (eero, Orbi, Deco, ...) as a station; the car
#                                         roams between mesh nodes, so range = the mesh's footprint. Then find the car
#                                         with `carnet.sh status` or http://ubuntu.local:8080/ (mDNS).
#   carnet.sh home <SSID> <password> [ip]   permanent home router for FPV range: fixed IP (default 192.168.0.250),
#                                         autoconnect above the hotspot; falls back to being the AP out of range
#   carnet.sh tether                      use a phone plugged in over USB as the uplink (phone: enable USB tethering),
#                                         for the cellular/Tailscale path (see docs/REMOTE.md)
#
# One radio: hotspot and client are mutually exclusive. Profiles are created once and reused.
set -e
IF=wlP1p1s0
cmd="${1:-status}"

need_root() { [ "$(id -u)" = 0 ] || exec sudo -E "$0" "$@"; }

mk_hotspot() {   # name band channel
    if ! nmcli -t -f NAME connection show | grep -qx "$1"; then
        nmcli connection add type wifi ifname "$IF" con-name "$1" ssid AtlasCar mode ap \
            802-11-wireless.band "$2" 802-11-wireless.channel "$3" ipv4.method shared ipv6.method disabled \
            wifi-sec.key-mgmt wpa-psk wifi-sec.psk "${ATLASCAR_PSK:?set ATLASCAR_PSK=<password> the first time}" \
            connection.autoconnect yes connection.autoconnect-priority 10 >/dev/null
    fi
}

case "$cmd" in
  status)
    nmcli -t -f DEVICE,STATE,CONNECTION device | grep "^$IF"
    nmcli -t -f ACTIVE,SSID,SIGNAL,CHAN,FREQ device wifi list ifname "$IF" 2>/dev/null | grep '^yes' || true
    echo "IPs: $(hostname -I)"
    ip route show default | head -1 || true
    ;;
  hotspot)
    need_root "$@"
    mk_hotspot AtlasCar a 36
    nmcli connection modify AtlasCar connection.autoconnect-priority 10
    nmcli connection modify AtlasCar24 connection.autoconnect-priority 0 2>/dev/null || true
    nmcli connection up AtlasCar && echo "hotspot AtlasCar (5 GHz) up: car = 10.42.0.1"
    ;;
  hotspot24)
    need_root "$@"
    mk_hotspot AtlasCar24 bg 6
    nmcli connection modify AtlasCar24 connection.autoconnect-priority 10
    nmcli connection modify AtlasCar connection.autoconnect-priority 0 2>/dev/null || true
    nmcli connection up AtlasCar24 && echo "hotspot AtlasCar (2.4 GHz) up: car = 10.42.0.1"
    ;;
  client)
    need_root "$@"
    ssid="${2:?usage: carnet.sh client <SSID> [password]}"
    if [ -n "${3:-}" ]; then
        nmcli device wifi connect "$ssid" password "$3" ifname "$IF" >/dev/null
    else
        nmcli connection up "$ssid" >/dev/null || nmcli device wifi connect "$ssid" ifname "$IF" >/dev/null
    fi
    # low latency for teleop: no WiFi power save, roam aggressively between mesh nodes
    nmcli connection modify "$ssid" 802-11-wireless.powersave 2 connection.autoconnect-priority 20 2>/dev/null || true
    iw dev "$IF" set power_save off 2>/dev/null || true
    echo "joined $ssid: car IP $(hostname -I | awk '{print $1}')  (also http://ubuntu.local:8080/ via mDNS)"
    ;;
  home)
    # Permanent home network for FPV range: a dedicated router (e.g. AtlasNet). Fixed IP so the
    # pilot page has a stable URL, autoconnect priority above the hotspot (20 > 10), so the car
    # joins this whenever it is in range and falls back to being its own AP when it is not.
    #   carnet.sh home <SSID> <password> [ip]        default ip 192.168.0.250 (gateway/dns = x.x.x.1)
    need_root "$@"
    ssid="${2:?usage: carnet.sh home <SSID> <password> [ip]}"; pw="${3:?password}"; ip="${4:-192.168.0.250}"
    gw="${ip%.*}.1"
    nmcli connection delete "$ssid" >/dev/null 2>&1 || true
    join() {   # $1 = key-mgmt (sae for WPA3-Personal, wpa-psk for WPA2)
        nmcli connection delete "$ssid" >/dev/null 2>&1 || true
        nmcli connection add type wifi ifname "$IF" con-name "$ssid" ssid "$ssid" \
            wifi-sec.key-mgmt "$1" wifi-sec.psk "$pw" \
            connection.autoconnect yes connection.autoconnect-priority 20 \
            802-11-wireless.powersave 2 \
            ipv4.method manual ipv4.addresses "$ip/24" ipv4.gateway "$gw" ipv4.dns "$gw" >/dev/null
        nmcli connection up "$ssid" >/dev/null 2>&1
    }
    if join sae; then echo "joined $ssid (WPA3)"
    elif join wpa-psk; then echo "joined $ssid (WPA2)"
    else echo "could not join $ssid: check the password, or set the router to WPA2/WPA3 mixed"; exit 1; fi
    iw dev "$IF" set power_save off 2>/dev/null || true
    echo "car = $ip  (autoconnect priority 20; AtlasCar hotspot is the fallback when out of range)"
    echo "pilot page: http://$ip:8080/"
    ;;
  tether)
    need_root "$@"
    dev=$(nmcli -t -f DEVICE,TYPE device | awk -F: '$2=="ethernet" && $1 ~ /^(usb|enx|enP.*u)/ {print $1; exit}')
    [ -n "$dev" ] || { echo "no USB-tethered phone found (enable USB tethering on the phone first)"; exit 1; }
    nmcli device connect "$dev" >/dev/null && echo "tethered via $dev: $(ip -4 addr show "$dev" | grep -oE 'inet [0-9.]+')"
    ;;
  *) sed -n '2,16p' "$0"; exit 1 ;;
esac
