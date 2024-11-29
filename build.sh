#!/bin/bash

# Install Rust
curl --proto 'https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

# Install Python dependencies
pip install -r requirements.txt
