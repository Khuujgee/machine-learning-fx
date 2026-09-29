#!/usr/bin/env bash
# Run the scanner as a macOS background service (launchd LaunchAgent):
# starts at login, restarts if it crashes, independent of any Terminal window.
#
#   ./service.sh install     install + start (also starts at every login)
#   ./service.sh uninstall   stop + remove
#   ./service.sh restart     restart (e.g. after changing code or .env)
#   ./service.sh status      is it loaded / running?
#   ./service.sh logs        follow the log (Ctrl-C to stop following)
set -euo pipefail
cd "$(dirname "$0")"
PROJECT="$(pwd)"
LABEL="com.forexmlpapertrader.scanner"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG="$PROJECT/logs/scanner.log"
DOMAIN="gui/$(id -u)"

write_plist() {
  mkdir -p "$PROJECT/logs" "$HOME/Library/LaunchAgents"
  cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$PROJECT/.venv/bin/python</string>
    <string>-W</string><string>ignore</string>
    <string>-m</string><string>src.scanner</string>
  </array>
  <key>WorkingDirectory</key><string>$PROJECT</string>
  <key>EnvironmentVariables</key>
  <dict><key>PYTHONUNBUFFERED</key><string>1</string></dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>60</integer>
  <key>ProcessType</key><string>Background</string>
  <key>StandardOutPath</key><string>$LOG</string>
  <key>StandardErrorPath</key><string>$LOG</string>
</dict>
</plist>
EOF
}

case "${1:-status}" in
  install)
    [[ -x .venv/bin/python ]] || { echo "Run ./setup.sh first."; exit 1; }
    [[ -f models/xgb_direction.joblib ]] || { echo "Train a model first: .venv/bin/python -m src.train"; exit 1; }
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
    write_plist
    launchctl bootstrap "$DOMAIN" "$PLIST"
    echo "Installed and started. Log: $LOG"
    ;;
  uninstall)
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
    rm -f "$PLIST"
    echo "Stopped and removed."
    ;;
  restart)
    launchctl kickstart -k "$DOMAIN/$LABEL" && echo "Restarted."
    ;;
  status)
    if launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
      launchctl print "$DOMAIN/$LABEL" | grep -E "^\s+(state|pid|last exit code) =" | head -3 | sed -E "s/^[[:space:]]+/  /"
    else
      echo "  not installed (run ./service.sh install)"
    fi
    ;;
  logs)
    tail -n 30 -f "$LOG"
    ;;
  *)
    echo "usage: $0 install|uninstall|restart|status|logs"; exit 1 ;;
esac
