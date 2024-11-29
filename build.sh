#!/bin/bash

# Install Rust
curl --proto 'https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

# Verify Rust installation
rustc --version

# Upgrade pip and install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
