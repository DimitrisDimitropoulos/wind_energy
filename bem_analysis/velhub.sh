#!/bin/env bash

INPUT_FILE="aeroin.inp"
BACKUP_FILE="${INPUT_FILE}.bak"

# Create a backup of the original file
cp "$INPUT_FILE" "$BACKUP_FILE"

for VELHUB in {5..30}; do
	# Use awk to replace the first number in the file followed by "! VELHUB"
	awk -v new_vel="$VELHUB" '
        /! VELHUB/ { $1 = new_vel } 1
    ' "$BACKUP_FILE" >"$INPUT_FILE"

	echo "Running ./raft with VELHUB = $VELHUB"
	./raft
done

# Restore original file
mv "$BACKUP_FILE" "$INPUT_FILE"
