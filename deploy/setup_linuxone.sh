#!/usr/bin/env bash
# One-shot setup for PAVAN SEWA Command on an IBM LinuxONE (s390x) VM.
# Works on RHEL / Rocky / Alma, Ubuntu and SLES (LinuxONE Community Cloud images).
#
# The server only needs Flask + cryptography. The trained model ships as JSON,
# so nothing has to be compiled on s390x (no numpy / pandas / scikit-learn).
#
#   cd pavan-sewa-command && bash deploy/setup_linuxone.sh && python3 app.py
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== host: $(uname -m) / $(. /etc/os-release && echo "$PRETTY_NAME") / $(python3 --version)"
[ "$(uname -m)" = "s390x" ] && echo "== running on IBM Z / LinuxONE" || echo "== WARNING: not s390x (fine for development)"

# cryptography comes pre-built from the distro (building it with pip needs Rust)
if command -v dnf >/dev/null; then
  sudo dnf install -y python3-pip python3-cryptography
elif command -v apt-get >/dev/null; then
  sudo apt-get update -y && sudo apt-get install -y python3-pip python3-flask python3-cryptography
elif command -v zypper >/dev/null; then
  sudo zypper --non-interactive install python3-pip python3-cryptography
fi

# Flask is pure Python - always installs cleanly
python3 -c "import flask" 2>/dev/null || python3 -m pip install --user "flask>=2.0,<3.1"
python3 -c "import flask, cryptography; print('== flask', flask.__version__, '| cryptography', cryptography.__version__)"

# Re-run the evaluation and the latency benchmark ON IBM Z (stdlib only)
python3 scripts/evaluate_dispatch.py > /dev/null && echo "== dispatch evaluation re-run on $(uname -m)"
python3 scripts/benchmark.py | tee "model/benchmark_$(uname -m).txt"

# Optional: retrain on IBM Z too, only if scikit-learn happens to be available
if python3 -c "import sklearn, pandas, numpy" 2>/dev/null; then
  python3 data/generate.py && python3 model/train.py
else
  echo "== (skipping retrain: scikit-learn not installed - the shipped model/triage_model.json is used)"
fi

# open the port if a firewall is active
if command -v firewall-cmd >/dev/null && sudo firewall-cmd --state >/dev/null 2>&1; then
  sudo firewall-cmd --add-port=8080/tcp --permanent && sudo firewall-cmd --reload
elif command -v ufw >/dev/null && sudo ufw status | grep -q active; then
  sudo ufw allow 8080/tcp
fi

cat <<EOF

== Done. Start the server:   python3 app.py
   From your laptop:          ssh -i <key>.pem -L 8080:localhost:8080 linux1@<VM-IP>
   then open                  http://localhost:8080
EOF
