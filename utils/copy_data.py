import glob
import os
import shutil
import sys
import logging
import random

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)

file_handler = logging.FileHandler('copy_data.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

partition_group = 0

# img_dir = '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/gun/train/images/'
# labels_dir = '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/gun/train/labels/'

# img_dir = '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/other/train/group_8/images/'
# labels_dir = '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/other/train/group_8/labels/'

# img_dir = '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/gun/train/images/'
# labels_dir = '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/gun/train/labels/'

# img_dir = '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/other/train/group_2/images/'
# labels_dir = '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/other/train/group_2/labels/'

img_dir = '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/gun/test/images/'
labels_dir = '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/gun/test/labels/'

# img_dir = '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/other/test/images/'
# labels_dir = '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/other/test/labels/'

# output_img_dir = '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/ssd/train/images/'
# output_label_dir = '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/ssd/train/labels/'

# output_img_dir = '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/ssd/validation/images/'
# output_label_dir = '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/ssd/validation/labels/'

output_img_dir = '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/ssd/test/images/'
output_label_dir = '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/ssd/test/labels/'

if not os.path.exists(output_img_dir):
    os.makedirs(output_img_dir)

if not os.path.exists(output_label_dir):
    os.makedirs(output_label_dir)

files = glob.glob(os.path.join(img_dir, '*.jpg'))
files.sort()
total_files = len(files)
group_size = 1000

start_idx = group_size * partition_group
end_idx = group_size * partition_group + group_size
group_files = random.sample(files, group_size)
# group_files = files[start_idx:end_idx]
total_files = len(group_files)
done = False
idx = 1
for img_file in group_files:
    basename = os.path.basename(img_file)[:-4]
    logging.info("processing %d: %s", idx, basename)

    label_file = labels_dir + basename + ".txt"

    copy_img_file = output_img_dir + basename + ".jpg"
    copy_label_file = output_label_dir + basename + ".txt"

    img_file_exists = os.path.exists(img_file)
    label_file_exists = os.path.exists(label_file)
    if img_file_exists and label_file_exists:
        shutil.copy(img_file, copy_img_file)
        shutil.copy(label_file, copy_label_file)
    else:
        logging.info("file: %s is missing label of image.", basename)
    idx += 1

logging.info("finished processing.")
