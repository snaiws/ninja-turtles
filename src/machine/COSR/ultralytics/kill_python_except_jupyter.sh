#!/bin/bash

# List all processes containing 'python' in their name and loop through each process ID
for pid in $(pgrep -f python); do
    # Check if the process command line does not include 'jupyter'
    if ! ps -p $pid -o cmd | grep -q jupyter; then
        # Kill the process
        kill -9 $pid
    fi
done
