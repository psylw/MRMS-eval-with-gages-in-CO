#%%
import os
import glob
import shutil
# List of replacements in order
replacements = [
    ("S23", "S3"),
    ("S3", "S4"),
    ("S4", "S5"),
    ("S8", "S6"),
    ("S11", "S7"),
    ("S7", "S8"),
    ("S5", "S9"),
    ("S6", "S10"),
    ("S22", "S11"),
    ("S17", "S12"),
    ("S21", "S13"),
    ("S18", "S14"),
    ("S10", "S15"),
    ("S20", "S16"),
    ("S12", "S17"),
    ("S16", "S18"),
    ("S13", "S19"),
    ("S14", "S20"),
    ("S15", "S21"),
    ("S9", "S22"),
    ("S19", "S23")

]
files = glob.glob('*.pdf')

directory = os.getcwd()
temp_directory = os.path.join(directory, "temp_renamed_files")
os.makedirs(temp_directory, exist_ok=True)
# Apply the renaming based on the provided mapping
for old_name, new_name in replacements:
    # Check if a file with the old name exists in the directory
    for file in files:
        if old_name in file:
            # Get the file extension
            file_extension = os.path.splitext(file)[1]
            
            # Construct old file name and new file name with extension
            old_file = os.path.join(directory, file)
            new_file = os.path.join(temp_directory, new_name + file_extension)
            
            # Copy the file to the temporary directory with the new name
            shutil.copy(old_file, new_file)
            print(f"Copied and renamed: {old_file} -> {new_file}")

print(f"Renaming completed! All renamed files are in: {temp_directory}")

# %%
