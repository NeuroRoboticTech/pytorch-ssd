import os
from pathlib import Path

dir_a = "../../gun_dataset/detector/other/train/Error/"
dir_b = "../../gun_dataset/detector/other/train/Error/JPEGImages/"

files_a = [os.path.basename(x) for x in Path(dir_a).glob('*.jpg')]
files_b = [os.path.basename(x) for x in Path(dir_b).glob('*.jpg')]

# Look for duplicates and delete from second directory.
dup_count = 0
for img_file in files_a:
    if img_file in files_b:
        del_file = os.path.abspath(dir_b + img_file)
        os.remove(del_file)
        dup_count += 1
        print("deleting: " + img_file)

print("deleted file total: " + str(dup_count))




