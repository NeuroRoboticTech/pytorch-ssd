import glob
import os
import csv

classes = ['person', 'gun']
input_dir = "/media/dcofer/Ubuntu_Data/active_shooter_defense/gun_dataset/detector/ssd/test"
output_file = "sub-test-annotations-bbox.csv"

image_dir = os.path.join(input_dir, 'images')
label_dir = os.path.join(input_dir, 'labels')
output_path = os.path.join(input_dir, output_file)

# identify all the xml files in the annotations folder (input directory)
image_files = glob.glob(os.path.join(image_dir, '*.jpg'))

# Open the csv file
csv_file = open(output_path, 'w')
writer = csv.writer(csv_file)
header = ['ImageID', 'Source', 'LabelName', 'Confidence',
          'XMin', 'XMax', 'YMin', 'YMax', 'IsOccluded',
          'IsTruncated', 'IsGroupOf', 'IsDepiction',
          'IsInside', 'id', 'ClassName']
writer.writerow(header)

# loop through each
for imag_file in image_files:
    basename = os.path.basename(imag_file)
    filename = os.path.splitext(basename)[0]
    img_file = os.path.join(image_dir, f"{filename}.jpg")
    label_file = os.path.join(label_dir, f"{filename}.txt")
    # check if the label contains the corresponding image file
    if not os.path.exists(img_file):
        print(f"{filename} image does not exist!")
        continue
    if not os.path.exists(label_file):
        print(f"{filename} label does not exist!")
        continue

    with open(label_file, mode='r') as csv_label_file:
        label_data = csv.DictReader(csv_label_file)
        for row in label_data:
            class_id = row[0]
            x_center = row[1]
            y_center = row[2]
            width = row[3]
            height = row[4]

            x_min = x_center - width/2.0
            x_max = x_center + width/2.0
            y_min = y_center - height/2.0
            y_max = y_center + height/2.0

            class_name = classes[class_id]
            label_name = '/m/' + class_name
            row_data = [filename, 'custom', label_name, 1,
                        x_min, x_max, y_min, y_max,
                        0, 0, 0, 0, 0, label_name, class_name]
            writer.writerow(row_data)

csv_file.close()
