# #!/bin/bash

# # Define the source directory containing the .zip files and the destination directory
# SOURCE_DIR="Improve_prGOFmax/PDB_test_set/structure_file"
# DEST_DIR="Improve_prGOFmax/PDB_test_set/structure_file/unzipped"

# # Create the destination directory if it doesn't exist
# mkdir -p "$DEST_DIR"

# # Loop through all .zip files in the source directory
# for zip_file in "$SOURCE_DIR"/*.zip; do
#   # Unzip the file into the destination directory
#   unzip -q "$zip_file" -d "$DEST_DIR"
# done

# echo "All .zip files have been unzipped to $DEST_DIR."


#!/bin/bash

# Define the directory containing the .gz files
TARGET_DIR="Improve_prGOFmax/PDB_test_set/structure_file/unzipped"

# Loop through all .gz files in the target directory
for gz_file in "$TARGET_DIR"/*.gz; do
  if gunzip -f "$gz_file"; then
    echo "Successfully unzipped: $gz_file"
  else
    echo "Failed to unzip: $gz_file. Removing..."
    rm -f "$gz_file"
  fi
done

echo "Process completed."
