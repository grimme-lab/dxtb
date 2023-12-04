#!/bin/bash

# Set the number of iterations
iterations=10

# Array of molecules
molecules=("mb16_43_01" "c60" "vancoh2")
#molecules=("mb16_43_01")

# Loop over the molecules
for molecule in "${molecules[@]}"; do
    echo "Running for molecule: $molecule"

    # Array to store the timings
    timings=()
    timings2=()

    # Run the program multiple times
    for ((i=1; i<=$iterations; i++)); do
        # Run the program and capture the output
        output=$(dxtb "../xtbml-tools/test-molecules/$molecule/coord" --dtype dp --grad --xtol 5e-4 --ftol 5e-5)
        # --xtol 5e-2 --ftol 5e-4

        #output2=$(xtb "../xtbml-tools/test-molecules/$molecule/coord" --gfn 1 --grad --ceasefiles)
        cd $HOME/Dokumente/tblite-my-fork
        output2=$(fpm run --profile release -- "../xtbml-tools/test-molecules/$molecule/coord" --method gfn1 --grad)
        cd - &> /dev/null

        # Extract the timing from the output using awk
        timing=$(echo "$output" | awk '/total/{print $2F}')
        timing2=$(echo "$output2" | awk '/total:/{print $2F}')

        # Print the timing for the current run
        echo "Timing for run $i: $timing vs $timing2"

        # Add the timing to the array
        timings+=("$timing")
        timings2+=("$timing2")
    done

    # Calculate the average timing
    total=0
    for timing in "${timings[@]}"; do
        total=$(awk "BEGIN {print $total + $timing}")
    done
    total2=0
    for timing2 in "${timings2[@]}"; do
        total2=$(awk "BEGIN {print $total2 + $timing2}")
    done
    average=$(awk "BEGIN {print $total / $iterations}")
    average2=$(awk "BEGIN {print $total2 / $iterations}")

    # Print the average timing
    echo "Average timing for $molecule: $average vs $average2"
    echo
done
