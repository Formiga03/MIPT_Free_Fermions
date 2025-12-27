import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

# Configuration
source_root = os.path.join("data", "1D")
dest_root = os.path.join("data", "filtered") # Change this to your desired path
search_pattern = "(500,1,last(500))"
sizes = [25, 50, 100, 250, 500, 1000]

for ii in sizes:
    folder_name = f"syst_L={ii}"
    
    current_source = os.path.join(source_root, folder_name)
    current_dest = os.path.join(dest_root, folder_name)

    if os.path.exists(current_source):
        # Create the destination sub-folder if it doesn't exist
        if not os.path.exists(current_dest):
            os.makedirs(current_dest)

        for filename in os.listdir(current_source):
            # Check if the pattern exists in the filename
            if search_pattern in filename:
                source_file = os.path.join(current_source, filename)
                dest_file = os.path.join(current_dest, filename)
                
                shutil.move(source_file, dest_file)
                print(f"Moved: {filename}")

"""
for ii in [1, 200, 400]:
    for jj in [0.4, 0.8]:
        file_path1 = f"IPR_24x24_p={jj}_t={ii}.txt"
        file_path2 = f"IPR_28x28_p={jj}_t={ii}.txt"

        data1 = np.loadtxt(file_path1)
        data2 = np.loadtxt(file_path2)

        x1 = np.linspace(0, 1, len(data1))
        x2 = np.linspace(0, 1, len(data2))

        plt.plot(x1, data1, label='24x24')
        plt.plot(x2, data2, label='28x28')

        plt.xlabel('Normalized Position')
        plt.ylabel('EigenVals')
        plt.legend()

        plt.savefig(f"p={jj}t={ii}.png")
        plt.clf()
        plt.close()
"""