#!/bin/bash
# PDF Content Extractor Server Start Script

if [ ! -f "./venv/bin/python" ]; then
    echo " --------------------------------------------------------"
    echo "| ❌ Python executable not found at: ./venv/bin/python    |"
    echo "| 👉 Read the documentation to setup the environment.     |"
    echo " --------------------------------------------------------"
    exit 1
fi

# Kill any existing servers on port 8000-8010
for port in {8000..8010}; do
    lsof -ti:$port | xargs kill -9 2>/dev/null || true
done

sleep 1                     # Wait a moment for cleanup
./venv/bin/python main.py   # Start the server using the virtual environment directly