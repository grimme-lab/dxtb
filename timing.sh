#!/bin/bash

# Set the number of iterations
iterations=10

# Array of molecules
molecules=("mb16_43_01" "c60" "vancoh2")
molecules=("vancoh2")

# Loop over the molecules
for molecule in "${molecules[@]}"
do
    echo "Running for molecule: $molecule"

    # Array to store the timings
    timings=()

    # Run the program multiple times
    for ((i=1; i<=$iterations; i++))
    do
        # Run the program and capture the output
        output=$(dxtb "../xtbml-tools/test-molecules/$molecule/coord" --dtype dp --int-driver libcint --xtol 5e-2 --ftol 5e-4)

        # Extract the timing from the output using awk
        timing=$(echo "$output" | awk '/SCF/{print $2F}')

        # Print the timing for the current run
        echo "Timing for run $i: $timing sec"

        # Add the timing to the array
        timings+=("$timing")
    done

    # Calculate the average timing
    total=0
    for timing in "${timings[@]}"
    do
        total=$(awk "BEGIN {print $total + $timing}")
    done
    average=$(awk "BEGIN {print $total / $iterations}")

    # Print the average timing
    echo "Average timing for $molecule: $average sec"
    echo
done
