#!/bin/bash

# Install latest CMake from Kitware repository
set -e

echo "Installing latest CMake from Kitware repository..."

# Add Kitware's repository and key
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | \
sudo gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg

echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" | \
sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null

# Update and install
sudo apt update
sudo apt install -y cmake

# Verify installation
echo "CMake version: $(cmake --version | head -n1)"

