#!/bin/bash
backup_dir=$1
dest_dir=$2
input=$3

while IFS= read -r filename
do
    [ -z "$filename" ] && continue
    dest=$(find "$dest_dir" -name "$filename" -print -quit)

    if [ -z "$dest_dir" ]; then
        echo "No match found for: $filename in $dest_dir"
        continue
    fi
    src="$backup_dir$filename"

    if [ ! -f "$src" ]; then
        echo "No source file in $backup_dir: $filename"
        continue
    fi

    echo "mv \"$src\" \"$dest\""
    # mv "$src" "$dest"
done < "$input"

echo "Done"