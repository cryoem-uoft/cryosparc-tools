#!/usr/bin/env bash
set -e

PYTHON_VERSION="3.10.8"
PYTHON="python3.10"
PIP="pip3.10"

# Update system deps, install sqlite
yum update -y
yum install sqlite-devel -y

# Extract and compile python
curl -L https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz -o Python-${PYTHON_VERSION}.tgz
tar -xzf Python-${PYTHON_VERSION}.tgz
cd Python-${PYTHON_VERSION}
./configure
make
make altinstall
cd ..

$PYTHON --version
$PIP install --upgrade pip wheel
$PIP install -e ".[build]"
