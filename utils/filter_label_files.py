import os
from pathlib import Path

labels_input_dir = '/media/dcofer/Ubuntu_Data/active_shooter_defense/yolov7/coco/labels/val2017'
labels_output_dir = '/media/dcofer/Ubuntu_Data/active_shooter_defense/yolov7/coco/labels/val2017_person'
classes_to_keep = ['0']

# create the output folder if needed
if not os.path.exists(labels_output_dir):
    os.makedirs(labels_output_dir)

files = list(Path(labels_input_dir).rglob('*.txt'))

for label_file in files:
    basename = os.path.basename(label_file)
    print("processing {}".format(basename))

    lines = []
    done = False
    with open(label_file, 'r') as in_file:
        while not done:
            line = in_file.readline()
            if not line:
                done = True
            else:
                lines.append(line)

    out_file = labels_output_dir + '/' + basename
    with open(out_file, 'w') as out_file:
        for line in lines:
            vals = line.split(' ')
            if vals[0] in classes_to_keep:
                out_file.writelines(line)

