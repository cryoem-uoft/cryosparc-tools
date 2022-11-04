#!/usr/bin/env bash
set -e

PYTHON_VERSION="3.9.15"
PYTHON="python3.9"

# Update system deps, install sqlite
yum update -y
yum install bzip2-devel libffi-devel openssl-devel sqlite-devel -y

# Extract and compile python
curl -L https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz -o Python-${PYTHON_VERSION}.tgz
tar -xzf Python-${PYTHON_VERSION}.tgz
cd Python-${PYTHON_VERSION}
./configure
make
make altinstall
cd ..

# Create virtual environment and install python dependencies
$PYTHON --version
$PYTHON -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -e ".[build]"
