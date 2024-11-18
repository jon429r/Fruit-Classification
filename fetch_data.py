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
py_run("resplit_data.py")

print("Path to dataset files: ./Fruits_Classification")
