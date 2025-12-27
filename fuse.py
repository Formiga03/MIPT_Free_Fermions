import os

base_target = os.path.join("data", "2D")
base_source = os.path.join("DT", "2D")

sizes = [12, 16]

for ii in sizes:
    folder_name = f"syst_L={ii}x{ii}"
    
    current_source_dir = os.path.join(base_source, folder_name)
    current_target_dir = os.path.join(base_target, folder_name)

    if os.path.exists(current_source_dir) and os.path.exists(current_target_dir):
        for filename in os.listdir(current_source_dir):
            source_file = os.path.join(current_source_dir, filename)
            target_file = os.path.join(current_target_dir, filename)

            if os.path.isfile(source_file) and os.path.exists(target_file):
                with open(source_file, "r") as f_src:
                    content = f_src.read()

                if content:
                    with open(target_file, "r+") as f_dst:
                        existing_content = f_dst.read()
                        if existing_content and not existing_content.endswith("\n"):
                            f_dst.write("\n")
                        
                        f_dst.write(content)
                        print(f"Fused: {filename}")
                        