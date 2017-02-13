#!/bin/bash

# A collection of scripts for automatically cleaning up common issues in the repo.

# Remove trailing whitespace from source files
find . -type f -name '*.cpp' -exec sed --in-place 's/[[:space:]]\+$//' {} \+
find . -type f -name '*.h' -exec sed --in-place 's/[[:space:]]\+$//' {} \+
find . -type f -name '*.hpp' -exec sed --in-place 's/[[:space:]]\+$//' {} \+
find . -type f -name '*.sh' -exec sed --in-place 's/[[:space:]]\+$//' {} \+
find . -type f -name '*.py' -exec sed --in-place 's/[[:space:]]\+$//' {} \+
find . -type f -name '*.toml' -exec sed --in-place 's/[[:space:]]\+$//' {} \+
find . -type f -name '*.yml' -exec sed --in-place 's/[[:space:]]\+$//' {} \+
find . -type f -name '*Lists.txt' -exec sed --in-place 's/[[:space:]]\+$//' {} \+

