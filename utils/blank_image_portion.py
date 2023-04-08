import glob
import os
import shutil
import sys
import logging

import cv2


logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)

file_handler = logging.FileHandler('partition_data.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

partition_group = 13

img_dir = '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/other/train/images/'

output_img_dir = '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/other/train/group_' + \
                 str(partition_group + 1) + '/images/'
orig_img_dir = '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/other/train/group_' + \
                   str(partition_group + 1) + '/images_orig/'

if not os.path.exists(img_dir):
    raise RuntimeError("images dir not found.")

if not os.path.exists(output_img_dir):
    raise RuntimeError("original images not found.")

if not os.path.exists(orig_img_dir):
    os.makedirs(orig_img_dir)

files = glob.glob(os.path.join(img_dir, '*.jpg'))
files.sort()
total_files = len(files)
group_size = 5000

start_idx = group_size * partition_group
end_idx = group_size * partition_group + group_size
group_files = files[start_idx:end_idx]
total_files = len(group_files)

done = False
idx = 0
start_img_idx = 0
end_img_idx = 1230  # 4979
top_left = (156, 400)
bottom_right = (440, 497)
# for idx in range(0, total_files):
#     img_file = group_files[idx]
#     basename = os.path.basename(img_file)[:-4]
#     logging.info("idx: %d, file: %s", idx, basename)

for idx in range(start_img_idx, end_img_idx):
    img_file = group_files[idx]
    basename = os.path.basename(img_file)[:-4]
    logging.info("processing %d: %s", idx, basename)

    save_img_file = output_img_dir + basename + ".jpg"
    copy_img_file = orig_img_dir + basename + ".jpg"

    # Copy the original image file to orig_img files
    img_file_exists = os.path.exists(img_file)
    shutil.copy(img_file, copy_img_file)

    # Now fix the original image.
    img = cv2.imread(img_file)
    img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = (0, 0, 0)

    # cv2.imshow("Updated", img)
    # cv2.waitKey(0)

    cv2.imwrite(save_img_file, img)

logging.info("finished processing.")
