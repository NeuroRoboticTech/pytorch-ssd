from typing import List
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

file_handler = logging.FileHandler('create_training_test_sets.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

train_output_img_dir = '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/ssd/train/images/'
train_output_label_dir = '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/ssd/train/labels/'
test_output_img_dir = '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/ssd/test/images/'
test_output_label_dir = '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/ssd/test/labels/'
val_output_img_dir = '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/ssd/validation/images/'
val_output_label_dir = '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/ssd/validation/labels/'

input_folders = [
    '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/other/train/group_1',
    '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/other/train/group_2',
    '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/other/train/group_3',
    '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/other/train/group_4',
    '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/other/train/group_5',
    '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/other/train/group_6',
    '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/other/train/group_7',
    '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/other/train/group_8',
    '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/other/train/group_9',
    '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/other/train/group_10',
    '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/other/train/group_11',
    '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/other/train/group_12',
    '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/other/train/group_13',
    '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/other/train/group_14',
    '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/other/train/group_15',
    '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/other/test',
    '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/other/train/group_2',
    '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/gun/train',
    '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/gun/test']

if not os.path.exists(train_output_img_dir):
    os.makedirs(train_output_img_dir)
if not os.path.exists(train_output_label_dir):
    os.makedirs(train_output_label_dir)
if not os.path.exists(test_output_img_dir):
    os.makedirs(test_output_img_dir)
if not os.path.exists(test_output_label_dir):
    os.makedirs(test_output_label_dir)
if not os.path.exists(val_output_img_dir):
    os.makedirs(val_output_img_dir)
if not os.path.exists(val_output_label_dir):
    os.makedirs(val_output_label_dir)

# Get the total list of images we are working with.
logging.info('getting total counts for all folders.')
image_files = []
for input_dir in input_folders:
    images_dir = input_dir + '/images'
    if not os.path.exists(images_dir):
        raise RuntimeError('images dir not found: {}'.format(images_dir))
    dir_files = glob.glob(os.path.join(images_dir, '*.jpg'))
    image_files.extend(dir_files)

random.shuffle(image_files)

val_perc: float = 0.05
test_perc: float = 0.05
total_images_count = len(image_files)
total_val_count = int(total_images_count * val_perc)
total_test_count = int(total_images_count * test_perc)
total_train_count = total_images_count - total_val_count - total_test_count

logging.info('total image files: %d', total_images_count)
logging.info('total train files: %d', total_train_count)
logging.info('total test files: %d', total_test_count)
logging.info('total validation files: %d', total_val_count)


def copy_group(output_img_dir: str, output_label_dir: str, images: List[str], start_idx: int, end_idx: int):

    for idx in range(start_idx, end_idx, 1):
        img_file = images[idx]
        basename = os.path.basename(img_file)[:-4]
        logging.info("processing %d: %s", idx, basename)

        label_file = img_file.replace('/images/', '/labels/').replace('.jpg', '.txt')

        copy_img_file = output_img_dir + basename + ".jpg"
        copy_label_file = output_label_dir + basename + ".txt"

        img_file_exists = os.path.exists(img_file)
        label_file_exists = os.path.exists(label_file)
        if img_file_exists and label_file_exists:
            shutil.copy(img_file, copy_img_file)
            shutil.copy(label_file, copy_label_file)
        else:
            logging.info("file: %s is missing label of image.", basename)


# start out by copying test images.
logging.info('copying test images.')
start_idx = 0
end_idx = start_idx + total_test_count
copy_group(test_output_img_dir, test_output_label_dir, image_files, start_idx, end_idx)

# then copy validation images.
logging.info('copying validation images.')
start_idx = end_idx
end_idx = start_idx + total_val_count
copy_group(val_output_img_dir, val_output_label_dir, image_files, start_idx, end_idx)

# then copy training images.
logging.info('copying training images.')
start_idx = end_idx
end_idx = start_idx + total_train_count
copy_group(train_output_img_dir, train_output_label_dir, image_files, start_idx, end_idx)

logging.info('finished creating dataset.')



