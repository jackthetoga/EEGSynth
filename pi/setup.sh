#!/usr/bin/env bash
# One-time Raspberry Pi setup. Run from the repo root on the Pi:  ./pi/setup.sh
set -euo pipefail

cd "$(dirname "$0")/.."

sudo apt-get update
sudo apt-get install -y python3-venv python3-dev

# Dialout group gives access to /dev/ttyUSB* without sudo.
sudo usermod -aG dialout "$USER"

# The Cyton dongle is an FTDI chip whose default 16 ms latency timer makes
# samples arrive in bursts. Drop it to 1 ms whenever the dongle is plugged in.
sudo tee /etc/udev/rules.d/99-openbci.rules >/dev/null <<'EOF'
ACTION=="add", SUBSYSTEM=="usb-serial", DRIVER=="ftdi_sio", ATTR{latency_timer}="1"
EOF
sudo udevadm control --reload-rules
sudo udevadm trigger

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e .

echo
echo "Done. Log out and back in (or reboot) for serial port permissions."
echo "Then test with:  .venv/bin/eegsynth check"
