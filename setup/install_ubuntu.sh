#!/bin/bash
# Production installation script for Ubuntu 22.04 LTS

set -e  # Exit on error

echo "========================================="
echo "Sovereign AI Suite - Production Install"
echo "========================================="

# Update system
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
echo "Installing system dependencies..."
sudo apt install -y \
    build-essential \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    htop \
    mongodb-org \
    nginx \
    supervisor

# Create application directory
echo "Creating application directory..."
sudo mkdir -p /opt/sovereign-ai-suite
sudo chown $USER:$USER /opt/sovereign-ai-suite
cd /opt/sovereign-ai-suite

# Create Python virtual environment
echo "Setting up Python environment..."
python3.10 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install Python requirements
echo "Installing Python packages..."
pip install -r setup/requirements.txt

# Install CUDA (if NVIDIA GPU present)
if lspci | grep -i nvidia > /dev/null; then
    echo "NVIDIA GPU detected. Installing CUDA..."
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt update
    sudo apt install -y cuda-12-3
fi

# Setup MongoDB
echo "Configuring MongoDB..."
sudo systemctl start mongod
sudo systemctl enable mongod

# Create directories
echo "Creating required directories..."
mkdir -p models data logs uploads config

# Download models
echo "Downloading AI models (this will take time)..."
python scripts/download_models.py

# Setup Nginx
echo "Configuring Nginx..."
sudo cp config/nginx.conf /etc/nginx/sites-available/sovereign-ai
sudo ln -s /etc/nginx/sites-available/sovereign-ai /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# Setup Supervisor
echo "Configuring Supervisor..."
sudo cp config/supervisor.conf /etc/supervisor/conf.d/sovereign-ai.conf
sudo supervisorctl reread
sudo supervisorctl update

echo "========================================="
echo "Installation complete!"
echo "========================================="
echo "To start services:"
echo "sudo supervisorctl start all"
echo "========================================="