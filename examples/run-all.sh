#!/bin/bash

set -euo pipefail

HERE=$(dirname "$(readlink -f "$0")")
cd "$HERE"

# search recursively for all python files
for example in $(find . -name "*.py"); do
    title="Running $example"

    # match the length of the title
    line=$(printf '=%.0s' $(seq 1 ${#title}))

    echo "$line"
    echo "$title"
    echo "$line"

    python3 "$example"

    printf "\n\n"
done
