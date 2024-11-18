import kagglehub
import shutil
import os

# Download latest version
path = kagglehub.dataset_download("utkarshsaxenadn/fruits-classification")

# move to the right directory
if os.path.exists("./Fruits Classification"):
    os.rmdir("./Fruits Classification")
shutil.move(path, "./")
shutil.move("1/Fruits Classification", "./Fruits_Classification")
os.rmdir("1")

# run resplit_data.py
os.system("python3 resplit_data.py")

print("Path to dataset files: ./Fruits_Classification")
