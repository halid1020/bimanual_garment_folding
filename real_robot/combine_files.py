import os
import argparse

def combine_files_recursively(directory_path, output_filename):
    """
    Recursively finds all .py, .yaml, .yml, and .json files in a directory 
    and combines them into a single text file.
    """
    # Define the file extensions we want to capture
    valid_extensions = ('.py', '.yaml', '.yml', '.json')
    
    matched_files = []
    
    # Walk through the directory recursively
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(valid_extensions):
                # Create the full path to the file
                matched_files.append(os.path.join(root, file))

    # Check if any files were found
    if not matched_files:
        print(f"No matching files found in '{directory_path}'.")
        return

    try:
        # Open the output file in write mode
        with open(output_filename, 'w', encoding='utf-8') as outfile:
            for file_path in matched_files:
                # Extract the relative path for clear identification in the output
                relative_path = os.path.relpath(file_path, directory_path)
                
                # Write a clear visual header for each file
                outfile.write(f"{'='*50}\n")
                outfile.write(f"FILE: {relative_path}\n")
                outfile.write(f"{'='*50}\n\n")
                
                # Read the contents of the current file and write it
                try:
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        outfile.write(infile.read())
                except Exception as e:
                    print(f"Error reading {relative_path}: {e}")
                    outfile.write(f"# Error reading {relative_path}: {e}\n")
                
                # Add some blank space before the next file begins
                outfile.write("\n\n\n")
                
        print(f"Success! Combined {len(matched_files)} files into '{output_filename}'.")
    
    except Exception as e:
        print(f"An error occurred while writing the output file: {e}")

# --- Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursively combine Python, YAML, and JSON files into a single text file.")
    
    parser.add_argument(
        "--target_directory", 
        type=str, 
        default=".", 
        help="The directory to recursively scan. Defaults to the current directory."
    )
    
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="all_my_code.txt", 
        help="The name of the generated text file."
    )
    
    args = parser.parse_args()
    
    combine_files_recursively(args.target_directory, args.output_file)