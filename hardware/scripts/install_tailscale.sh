#!/usr/bin/env bash
# install_tailscale.sh — put the car on a Tailscale (WireGuard) VPN so the web pilot works from
# anywhere with internet: over a USB-tethered phone, an LTE modem, or any WiFi. Carrier NAT is
# not a problem; Tailscale punches through or relays. Run on the Jetson while it has internet
# (carnet.sh tether, or the phone hotspot). Needs sudo once. Then authenticate in a browser.
set -e
[ "$(id -u)" = 0 ] || exec sudo -E "$0" "$@"
. /etc/os-release
curl -fsSL "https://pkgs.tailscale.com/stable/ubuntu/${VERSION_CODENAME}.noarmor.gpg" -o /usr/share/keyrings/tailscale-archive-keyring.gpg
curl -fsSL "https://pkgs.tailscale.com/stable/ubuntu/${VERSION_CODENAME}.tailscale-keyring.list" -o /etc/apt/sources.list.d/tailscale.list
apt-get update -qq && apt-get install -y -qq tailscale
systemctl enable --now tailscaled
echo
echo "Now run:   sudo tailscale up --hostname atlascar"
echo "and open the login URL it prints on your laptop (same Tailscale account there)."
echo "Afterwards the car is reachable from anywhere as  http://atlascar:8080/  (MagicDNS)"
echo "or  http://\$(tailscale ip -4):8080/ . Use the page's 'video: low' setting on cellular."
