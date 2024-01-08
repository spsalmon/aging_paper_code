#!/bin/bash

# Function to recursively search for files
search_files() {
  local current_folder="$1"

  # Iterate over files in the current folder
  for file in "$current_folder"/*; do
    if [ -f "$file" ]; then
      # Extract the file name without extension
      file_name=$(basename "$file")
      file_name_without_ext="${file_name%.*}"

      # Extract the country code from the input file name
      country_code=$(echo "$file_name_without_ext" | awk -F "." '{print $1}')

      # Output file name with CSV extension
      output_file="$current_folder/$country_code.csv"

      # Extract the lines starting from "Year" until the end
      extracted_lines=$(awk '/^  Year/{flag=1;print;next}/^[[:space:]]*$/{flag=0}flag' "$file")

      # Replace "110+" with "110"
      modified_lines=$(echo "$extracted_lines" | sed 's/110+/110/g')

      # Remove lines containing isolated dots
      modified_lines=$(echo "$modified_lines" | grep -Pv '(\s|^)\.(\s|$)')

      # Replace multiple spaces with a single comma
      csv_data=$(echo "$modified_lines" | tr -s ' ' ',' | sed 's/^,//')

      # Save the CSV data to the output file
      echo "$csv_data" > "$output_file"

      # Delete the input file if the output file name is different
      if [ "$file" != "$output_file" ]; then
        rm "$file"
        echo "Input file has been deleted: $file"
      fi

      echo "CSV file has been created: $output_file"
    elif [ -d "$file" ]; then
      # Recursively search for files in subfolders
      search_files "$file"
    fi
  done
}

# Input folder name
input_folder="$1"

# Check if the input folder is provided
if [ -z "$input_folder" ]; then
  echo "Please provide the input folder as an argument."
  exit 1
fi

# Call the search_files function with the input folder
search_files "$input_folder"
