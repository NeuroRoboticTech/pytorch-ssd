import glob
import os

classes = []
gun_label_dir = "/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/gun/train/labels/"
person_label_dir = "/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/gun/train/labels_person/"
output_label_dir = "/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/gun/train/labels_combined/"

# create the labels folder (output directory)
if not os.path.exists(output_label_dir):
    os.mkdir(output_label_dir)

# identify all the xml files in the annotations folder (input directory)
gun_files = glob.glob(os.path.join(gun_label_dir, '*.txt'))
# loop through each
for gun_file in gun_files:
    basename = os.path.basename(gun_file)
    print("processing {}".format(basename))

    gun_lines = []
    done = False
    with open(gun_file, 'r') as in_file:
        while not done:
            line = in_file.readline()
            if not line:
                done = True
            else:
                gun_lines.append(line)

    person_file = person_label_dir + basename
    person_lines = []
    if os.path.exists(person_file):
        done = False
        with open(person_file, 'r') as in_file:
            while not done:
                line = in_file.readline()
                if not line:
                    done = True
                else:
                    person_lines.append(line)

    # Now create the merged file
    out_file = output_label_dir + '/' + basename
    with open(out_file, 'w') as out_file:
        for line in person_lines:
            out_file.writelines(line)

        for line in gun_lines:
            line = '1' + line[1:]
            out_file.writelines(line)

print("finished processing")


