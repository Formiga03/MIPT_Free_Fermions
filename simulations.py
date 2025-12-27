import subprocess
import sys

scripts = [
    "monitored_systs_2D.py", 
    "monitored_systs_2D.py",
    "monitored_systs_2D.py",
    "monitored_systs_2D.py"
]

args_list = [
    (["EEnt", "IPR"], True, 800, 0.25, 120, f"last({120})", [ii*0.05 for ii in range(11)], [12, 16], "1|4", True),
    (["EEnt", "IPR"], True, 400, 0.50,  60,  f"last({60})", [ii*0.05 for ii in range(11)], [12, 16], "1|4", True),
    (["EEnt", "IPR"], True, 267, 0.75,  40,  f"last({40})", [ii*0.05 for ii in range(11)], [12, 16], "1|4", True),
    (["EEnt", "IPR"], True, 200, 1.00,  30,  f"last({30})", [ii*0.05 for ii in range(11)], [12, 16], "1|4", True),
]

python_executable = sys.executable
for i in range(len(scripts)):
    current_script = scripts[i]
    current_params = args_list[i]
    
    cmd_args = [str(p) for p in current_params]
    
    subprocess.run(["python3", current_script, *cmd_args], check=True)