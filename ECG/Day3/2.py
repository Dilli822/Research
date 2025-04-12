import os

longterm_directory = "CYBHi/data/long-term"
shortterm_directory = "CYBHi/data/short-term"

# Count files in each directory
longterm_files = [name for name in os.listdir(longterm_directory) if os.path.isfile(os.path.join(longterm_directory, name))]
shortterm_files = [name for name in os.listdir(shortterm_directory) if os.path.isfile(os.path.join(shortterm_directory, name))]

# Total count
file_count = len(longterm_files) + len(shortterm_files)

print(f"Number of files in long-term: {len(longterm_files)}")
print(f"Number of files in short-term: {len(shortterm_files)}")
print(f"Total number of files: {file_count}")
