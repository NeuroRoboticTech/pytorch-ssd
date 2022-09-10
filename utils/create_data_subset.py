import os
from pathlib import Path
import random
from filecmp import cmp

# # Test gun images
# images_input_dir = "../../gun_dataset/detector/gun/test/images"
# labels_input_dir = "../../gun_dataset/detector/gun/test/labels"
# images_output_dir = "../../gun_dataset/detector/gun/test_subset/images"
# labels_output_dir = "../../gun_dataset/detector/gun/test_subset/labels"

# Train gun images
# images_input_dir = "../../gun_dataset/detector/gun/train/images"
# labels_input_dir = "../../gun_dataset/detector/gun/train/labels"
# images_output_dir = "../../gun_dataset/detector/gun/train_subset/images"
# labels_output_dir = "../../gun_dataset/detector/gun/train_subset/labels"

# Test other images
# images_input_dir = "../../gun_dataset/detector/other/test/"
# labels_input_dir = ""
# images_output_dir = "../../gun_dataset/detector/other/test_subset/"
# labels_output_dir = ""

# Train other images
# images_input_dir = "../../gun_dataset/detector/other/train/"
# labels_input_dir = ""
# images_output_dir = "../../gun_dataset/detector/other/train_subset/"
# labels_input_dir = ""


# Test gun images
# images_input_dir = "../../gun_dataset/detector/gun/train/images"
# labels_input_dir = "../../gun_dataset/detector/gun/test/labels"
# images_output_dir = "../../gun_dataset/detector/combined/test/images"
# labels_output_dir = "../../gun_dataset/detector/combined/test/labels"

# Train gun images
# images_input_dir = "../../gun_dataset/detector/gun/train/images"
# labels_input_dir = "../../gun_dataset/detector/gun/train/labels"
# images_output_dir = "../../gun_dataset/detector/combined/train/images"
# labels_output_dir = "../../gun_dataset/detector/combined/train/labels"

# Test other images
# images_input_dir = "../../gun_dataset/detector/other/test/"
# labels_input_dir = ""
# images_output_dir = "../../gun_dataset/detector/combined/test/images"
# labels_output_dir = "../../gun_dataset/detector/combined/test/labels"

# Train other images
images_input_dir = "../../gun_dataset/detector/other/train/"
labels_input_dir = ""
images_output_dir = "../../gun_dataset/detector/combined/train/images"
labels_output_dir = "../../gun_dataset/detector/combined/train/labels"


# create the output folder if needed
if not os.path.exists(images_output_dir):
    os.makedirs(images_output_dir)

if labels_output_dir != "" and not os.path.exists(labels_output_dir):
    os.makedirs(labels_output_dir)

files = list(Path(images_input_dir).rglob('*.jpg'))
new_files = random.sample(files, 5000)

# now create symlinks.
for img_file in new_files:
    basename = os.path.basename(img_file)
    filename = os.path.splitext(basename)[0]
    out_img_file = os.path.join(images_output_dir, f"{filename}.jpg")

    full_img_file = os.path.abspath(img_file)
    full_out_img_file = os.path.abspath(out_img_file)
    os.symlink(full_img_file, full_out_img_file)

    label_file = os.path.join(labels_input_dir, f"{filename}.txt")
    out_label_file = os.path.join(labels_output_dir, f"{filename}.txt")
    if labels_input_dir != "":
        full_label_file = os.path.abspath(label_file)
        full_out_label_file = os.path.abspath(out_label_file)
        os.symlink(full_label_file, full_out_label_file)
    else:
        # Create a blank label file.
        full_out_label_file = os.path.abspath(out_label_file)
        with open(full_out_label_file, 'w') as fp:
            pass
