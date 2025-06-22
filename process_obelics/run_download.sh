#!/bin/bash

PYTHON_SCRIPT="download_data.py"

echo "Starting dataset download and loading process..."
echo "This script will repeatedly run $PYTHON_SCRIPT until the dataset is successfully loaded."

while true; do
    echo "--- Attempting to load dataset at $(date) ---"
    
    # Run the Python script and capture its output
    # '2>&1' redirects stderr to stdout so we can capture all output
    # `tee` will print to console and also save to a log file if needed
    OUTPUT=$(python3 "$PYTHON_SCRIPT" 2>&1)
    
    # Print the output from the last run for debugging/monitoring
    echo "$OUTPUT"

    # Check if the desired success message is in the output
    if echo "$OUTPUT" | grep -q "loaded!!"; then
        echo "--- Dataset successfully loaded! Exiting. ---"
        break # Exit the loop if "loaded!!" is found
    else
        echo "--- Dataset not fully loaded. Retrying in 10 seconds... ---"
        sleep 10 # Wait for 10 seconds before retrying
    fi
done