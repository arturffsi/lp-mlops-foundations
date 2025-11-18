#!/bin/bash
set -ex

echo "=== Bootstrap: Environment Check ==="
pwd
ls -la /opt/ml/processing/input/ || true
ls -la /opt/ml/processing/input/scripts/ || true
which python3
python3 --version
which pip

echo ""
echo "=== Bootstrap: Installing dependencies ==="
python3 -m pip install --upgrade pip
python3 -m pip install redshift_connector

echo ""
echo "=== Running export script ==="
python3 /opt/ml/processing/input/scripts/export_to_s3.py "$@"

echo ""
echo "=== Bootstrap: Complete ==="
