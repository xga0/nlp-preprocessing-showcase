#!/bin/bash

# Twitter Sentiment Analysis Installation Script
# This script sets up the environment and installs all required dependencies

echo "=========================================="
echo "Twitter Sentiment Analysis Setup"
echo "=========================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "FAILED: Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "SUCCESS: Python 3 found: $(python3 --version)"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "FAILED: pip3 is not installed. Please install pip first."
    exit 1
fi

echo "SUCCESS: pip3 found: $(pip3 --version)"

# Create virtual environment (optional)
read -p "Do you want to create a virtual environment? (y/n): " create_venv
if [[ $create_venv == "y" || $create_venv == "Y" ]]; then
    echo "Creating virtual environment..."
    python3 -m venv twitter_sentiment_env
    source twitter_sentiment_env/bin/activate
    echo "SUCCESS: Virtual environment created and activated"
fi

# Upgrade pip
echo "Upgrading pip..."
pip3 install --upgrade pip

# Install required packages
echo "Installing required packages..."
pip3 install -r requirements.txt

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo "SUCCESS: All packages installed successfully!"
else
    echo "FAILED: Some packages failed to install. Please check the error messages above."
    exit 1
fi

# Test the setup
echo "Testing setup..."
python3 test_setup.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "SUCCESS: SETUP COMPLETED SUCCESSFULLY!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "1. Run 'python demo.py' to see the packages in action"
    echo "2. Run 'python train.py' to train the sentiment analysis model"
    echo "3. Run 'python inference.py' to make predictions"
    echo ""
    echo "For more information, check the README.md file."
else
    echo ""
    echo "=========================================="
    echo "FAILED: SETUP FAILED!"
    echo "=========================================="
    echo "Please check the error messages above and try again."
    exit 1
fi
