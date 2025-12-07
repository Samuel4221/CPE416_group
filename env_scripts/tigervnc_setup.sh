#!/bin/bash

# === Configuration ===
WIFI_SSID="CP-IoT-Secure"
WIFI_PASSWORD="$1"  # Accept as argument
VNC_DISPLAY=":1"
USER_HOME="/home/$(logname)"
DEB_DIR="$USER_HOME/vnc"
VNC_PASS="cpe1234"

if [ -z "$WIFI_PASSWORD" ]; then
    echo "❌ Usage: sudo $0 <wifi_password>"
    exit 1
fi

echo "✅ Starting Jetson Orin Nano local VNC setup..."

### 1. Connect to Wi-Fi
echo "📶 Connecting to Wi-Fi: $WIFI_SSID"
nmcli device wifi connect "$WIFI_SSID" password "$WIFI_PASSWORD"

### 2. Set Display Resolution (optional)
echo "🖥️ Setting display resolution to 1920x1080"
export DISPLAY=$VNC_DISPLAY
xrandr --output DP-0 --mode 1920x1080 || echo "⚠️ Could not set resolution (headless?)"

### 3. Remove apport nag packages
### echo "🧹 Removing apport..."
### apt purge -y apport apport-gtk apport-symptoms

### 4. Install and configure TigerVNC
echo "🐯 Installing TigerVNC..."
apt update && apt install -y tigervnc-standalone-server tigervnc-common

echo "🔐 Setting VNC password..."
mkdir -p ~/.vnc
echo "$VNC_PASS" | vncpasswd -f > ~/.vnc/passwd
chmod 600 ~/.vnc/passwd

echo "📝 Creating xstartup script..."
cat <<EOF > ~/.vnc/xstartup
#!/bin/sh
xrdb \$HOME/.Xresources
startxfce4 &
EOF
chmod +x ~/.vnc/xstartup

echo "🛠️ Enabling VNC service..."
cat <<EOF > /etc/systemd/system/tigervncserver@.service
[Unit]
Description=TigerVNC server at %i
After=syslog.target network.target

[Service]
Type=forking
User=$(logname)
PAMName=login
PIDFile=/home/$(logname)/.vnc/%H:%i.pid
ExecStartPre=-/usr/bin/vncserver -kill :%i > /dev/null 2>&1
ExecStart=/usr/bin/vncserver :%i
ExecStop=/usr/bin/vncserver -kill :%i

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable tigervncserver@1
systemctl start tigervncserver@1

### 5. Install .deb files if available
if [ -d "$DEB_DIR" ]; then
    echo "📦 Installing .deb files from $DEB_DIR..."
    dpkg -i $DEB_DIR/*.deb || apt --fix-broken install -y
    mv $DEB_DIR/vncserver* /etc/tigervnc/ 2>/dev/null
fi

### 6. Fix PolicyKit for Wi-Fi scans
echo "🔓 Allowing Wi-Fi scans without prompts..."
cat <<EOF | tee /etc/polkit-1/localauthority/50-local.d/47-allow-wifi-scan.pkla
[Allow Wifi Scan]
Identity=unix-user:*
Action=org.freedesktop.NetworkManager.wifi.scan;org.freedesktop.NetworkManager.enable-disable-wifi;org.freedesktop.NetworkManager.settings.modify.own;org.freedesktop.NetworkManager.settings.modify.system;org.freedesktop.NetworkManager.network-control
ResultAny=yes
ResultInactive=yes
ResultActive=yes
EOF

### 7. Fix PolicyKit for colord prompt
echo "🔕 Disabling colord auth prompts..."
cat <<EOF | tee /etc/polkit-1/localauthority/50-local.d/45-allow-colord.pkla
[Allow Colord all Users]
Identity=unix-user:*
Action=org.freedesktop.color-manager.create-device;org.freedesktop.color-manager.create-profile;org.freedesktop.color-manager.delete-device;org.freedesktop.color-manager.delete-profile;org.freedesktop.color-manager.modify-device;org.freedesktop.color-manager.modify-profile
ResultAny=no
ResultInactive=no
ResultActive=yes
EOF

### 8. Install Firefox and Tilix
echo "🌐 Installing Firefox and Tilix..."
wget -q https://packages.mozilla.org/apt/repo-signing-key.gpg -O- | tee /etc/apt/keyrings/packages.mozilla.org.asc > /dev/null

echo "deb [signed-by=/etc/apt/keyrings/packages.mozilla.org.asc] https://packages.mozilla.org/apt mozilla main" | tee -a /etc/apt/sources.list.d/mozilla.list > /dev/null

cat <<EOF | tee /etc/apt/preferences.d/mozilla
Package: *
Pin: origin packages.mozilla.org
Pin-Priority: 1000
EOF

apt update
apt install -y firefox tilix

echo "✅ Setup complete. Reboot recommended."
