import os
import shutil

base_target = os.path.join("data", "1D")
base_source = os.path.join("DT", "1D")

sizes = [25, 50, 100]

for ii in sizes:
    folder_name = f"syst_L={ii}"
    
    current_source_dir = os.path.join(base_source, folder_name)
    current_target_dir = os.path.join(base_target, folder_name)

    if os.path.exists(current_source_dir) and os.path.exists(current_target_dir):
        for filename in os.listdir(current_source_dir):
            source_file = os.path.join(current_source_dir, filename)
            target_file = os.path.join(current_target_dir, filename)

            if os.path.isfile(source_file) and not os.path.exists(target_file):
                shutil.copy2(source_file, target_file)
                print(f"Added: {filename}")