import kagglehub
import shutil
import os
from py_script_exec import py_run

# Download latest version
path = kagglehub.dataset_download("utkarshsaxenadn/fruits-classification")

# move to the right directory
if os.path.exists("./Fruits Classification"):
    os.rmdir("./Fruits Classification")
shutil.move(path, "./")
shutil.move("1/Fruits Classification", "./Fruits_Classification")
os.rmdir("1")

# run resplit_data.py
exitcode, std_out, std_error = py_run("resplit_data.py")

if exitcode != 0:
    print("Error in resplit_data.py")
    print(std_error)
else:
    print("Data fetched successfully")
    print(std_out)

print("Path to dataset files: ./Fruits_Classification")
