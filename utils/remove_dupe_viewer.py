import glob
import os
import shutil
import logging

import numpy as np
import cv2

import logging
import sys

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)

file_handler = logging.FileHandler('remove_dupes.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

img_dir = '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/other/train/images/'
labels_dir = '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/other/train/labels/'
labels_combined_dir = \
    '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/other/train/labels_combined/'
labels_person_dir = '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/other/train/labels_person/'

dupe_img_dir = '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/other/train/duplicates/images/'
dupe_labels_dir = '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/other/train/duplicates/labels/'
dupe_labels_combined_dir = \
    '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/other/train/duplicates/labels_combined/'
dupe_labels_person_dir = \
    '/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/other/train/duplicates/labels_person/'


def draw_labels(img, labels_file):
    lines = []
    with open(labels_file) as f:
        lines = [line.rstrip() for line in f]

    for line in lines:
        box_vals = line.split(' ')
        class_type = int(box_vals[0])
        x = int(float(box_vals[1]) * img.shape[1])
        y = int(float(box_vals[2]) * img.shape[0])
        half_width = int((float(box_vals[3]) * img.shape[1])/2.0)
        half_height = int((float(box_vals[4]) * img.shape[0])/2.0)

        top_left_x = int(x - half_width)
        top_left_y = int(y - half_height)
        bottom_right_x = int(x + half_width)
        bottom_right_y = int(y + half_height)

        # top_left_x = 100
        # top_left_y = 200
        # bottom_right_x = 200
        # bottom_right_y = 400

        if class_type == 0:
            box_color = (0, 255, 0)
        else:
            box_color = (0, 0, 255)

        cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), box_color, 2)


def move_image(basename, img_file):
    # Move the image file
    dupe_img_file = dupe_img_dir + basename + '.jpg'
    shutil.move(img_file, dupe_img_file)

    # now check labels.
    labels_file = labels_dir + basename + '.txt'
    dupe_labels_file = dupe_labels_dir + basename + '.txt'
    if os.path.exists(labels_file):
        shutil.move(labels_file, dupe_labels_file)

    # now check labels_combined.
    labels_file = labels_combined_dir + basename + '.txt'
    dupe_labels_file = dupe_labels_combined_dir + basename + '.txt'
    if os.path.exists(labels_file):
        shutil.move(labels_file, dupe_labels_file)

    # now check labels_person.
    labels_file = labels_person_dir + basename + '.txt'
    dupe_labels_file = dupe_labels_person_dir + basename + '.txt'
    if os.path.exists(labels_file):
        shutil.move(labels_file, dupe_labels_file)

    logging.info("moved %s", basename)


def undo_move(basename):
    # Move the image file
    img_file = img_dir + basename + '.jpg'
    dupe_img_file = dupe_img_dir + basename + '.jpg'
    if os.path.exists(dupe_img_file):
        shutil.move(dupe_img_file, img_file)

    # now check labels.
    labels_file = labels_dir + basename + '.txt'
    dupe_labels_file = dupe_labels_dir + basename + '.txt'
    if os.path.exists(dupe_labels_file):
        shutil.move(dupe_labels_file, labels_file)

    # now check labels_combined.
    labels_file = labels_combined_dir + basename + '.txt'
    dupe_labels_file = dupe_labels_combined_dir + basename + '.txt'
    if os.path.exists(dupe_labels_file):
        shutil.move(dupe_labels_file, labels_file)

    # now check labels_person.
    labels_file = labels_person_dir + basename + '.txt'
    dupe_labels_file = dupe_labels_person_dir + basename + '.txt'
    if os.path.exists(dupe_labels_file):
        shutil.move(dupe_labels_file, labels_file)

    logging.info("undo move %s", basename)


files = glob.glob(os.path.join(img_dir, '*.jpg'))
files.sort()
total_files = len(files)
start_file_idx = 66247

done = False
undo = False
total_moved = 0
idx = start_file_idx
while not done and 0 <= idx < total_files:
    img_file = files[idx]
    basename = os.path.basename(img_file)[:-4]
    if undo:
        undo_move(basename)
        undo = False
        total_moved -= 1

    if os.path.exists(img_file):
        img = cv2.imread(img_file, 1)
    else:
        img = np.zeros((800, 800, 3), np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        image = cv2.putText(img, 'Moved', (400, 380), font,
                            1, (255, 0, 0), 2, cv2.LINE_AA)

    labels_file = labels_combined_dir + basename + '.txt'
    if os.path.exists(labels_file):
        draw_labels(img, labels_file)

    if img.shape[0] > 768 or img.shape[1] > 1024:
        if img.shape[0] > 768 and img.shape[0] > 0:
            scale_percent = 768 / img.shape[0]
        elif img.shape[1] > 0:
            scale_percent = 1024 / img.shape[1]
        else:
            scale_percent = 0.7

        width = int(img.shape[1] * scale_percent)
        height = int(img.shape[0] * scale_percent)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.imshow('image', np.array(img, dtype=np.uint8))
    cv2.imshow('image', img)
    key = cv2.waitKey(0)

    # logging.info("key: %d", key)
    if key == ord('d'):
        move_image(basename, img_file)
        total_moved += 1
        idx += 1
    elif key == ord(' '):
        # If we hit the space bar move to next image
        idx += 1
    elif key == ord('u'):
        # If we hit the u key then undo moving the last image.
        idx -= 1
        undo = True
    elif key == ord('q'):
        # If we hit q then quite
        done = True
    elif key == 83:
        # If we hit the left arrow move to next image
        idx += 1
    elif key == 81:
        # If we hit the right arrow move back
        idx -= 1

    if idx >= total_files:
        idx = total_files
    if idx < 0:
        idx = 0
    logging.info("current img: %d", (idx - total_moved))


cv2.destroyAllWindows()
