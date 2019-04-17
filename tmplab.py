import os
from shutil import copyfile
path = "/home/agajan/mylab/fmridata/KKI/"
folders = os.listdir(path)

for folder in folders:
    if os.path.isdir(path + folder):
        for file in os.listdir(path + folder):
            if file.startswith("sfnwmrda") and file.endswith(".nii.gz"):
                print(file)
                copyfile(path + folder + "/" + file, "/home/agajan/data/" + file)