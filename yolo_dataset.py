import numpy as np
import logging
import pathlib
import glob
import cv2
import os


class YoloDataset:

    def __init__(self, root, transform=None, target_transform=None, keep_difficult=False, skip_no_labels=False):
        """Dataset for YOLO data.
        Args:
            root: the root of the YOLO dataset, the directory contains the following sub-directories:
                images and labels.
        """
        self.root = pathlib.Path(root)
        self.images_dir = os.path.join(root, 'images')
        self.labels_dir = os.path.join(root, 'labels')
        self.transform = transform
        self.target_transform = target_transform
        self.keep_difficult = keep_difficult
        self.skip_no_labels = skip_no_labels

        if not os.path.exists(self.images_dir):
            raise RuntimeError("images directory not found. dir: {}".format(self.images_dir))

        if not os.path.exists(self.labels_dir):
            raise RuntimeError("labels directory not found. dir: {}".format(self.labels_dir))

        self.image_list = sorted(glob.glob('%s/*.jpg' % self.images_dir))

        self.ids = []
        for image_file in self.image_list:
            image_id = pathlib.Path(image_file).stem
            label_file = os.path.join(self.labels_dir, image_id + '.txt')
            label_file_size = os.path.getsize(label_file)

            if os.path.exists(image_file) and os.path.exists(label_file):
                if not self.skip_no_labels or (self.skip_no_labels and label_file_size > 0):
                    self.ids.append(image_id)
                else:
                    print("skipping file: {} because it has no label".format(image_id))
            else:
                print("skipping file: {}".format(image_id))

        # if the labels file exists, read in the class names
        label_file_name = self.root / "labels.txt"

        if os.path.isfile(label_file_name):
            classes = []

            # classes should be a line-separated list
            with open(label_file_name, 'r') as infile:
                for line in infile:
                    classes.append(line.rstrip())

            # prepend BACKGROUND as first class
            classes.insert(0, 'BACKGROUND')
            # classes  = [ elem.replace(" ", "") for elem in classes]
            self.class_names = tuple(classes)
            print("Labels read from file: " + str(self.class_names))
        else:
            raise RuntimeError('no labels.txt file found')

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        _, image, boxes, labels = self._getitem(index)
        return image, boxes, labels

    def _getitem(self, index):
        image_id = self.ids[index]
        boxes, labels, is_difficult = self._get_annotation(image_id)

        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]

        image = self._read_image(image_id)

        # Convert the normalized box values to concrete values for this image.
        if len(boxes):
            boxes[:, 0] *= image.shape[1]
            boxes[:, 1] *= image.shape[0]
            boxes[:, 2] *= image.shape[1]
            boxes[:, 3] *= image.shape[0]

        if logging.root.level is logging.DEBUG:
            logging.debug(
                f"yolo_dataset image_id={image_id}" + ' \n    boxes=' + str(boxes) + ' \n    labels=' + str(labels))

        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)

        return image_id, image, boxes, labels

    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        """To conform the eval_ssd implementation that is based on the VOC dataset."""
        image_id, image, boxes, labels = self._getitem(index)
        is_difficult = np.zeros(boxes.shape[0], dtype=np.uint8)
        return image_id, (boxes, labels, is_difficult)

    def __len__(self):
        return len(self.ids)

    def _get_annotation(self, image_id):
        label_file = os.path.join(self.labels_dir, image_id + '.txt')

        boxes = []
        labels = []
        is_difficult = []
        with open(label_file, 'r') as fp:
            for row in fp:
                row_data = row.split(' ')
                if len(row_data) == 5:
                    # Add 1 here because we added a background as first ID.
                    class_id = int(row_data[0]) + 1
                    x_center = float(row_data[1])
                    y_center = float(row_data[2])
                    width = float(row_data[3])
                    height = float(row_data[4])

                    x_min = x_center - width / 2.0
                    x_max = x_center + width / 2.0
                    y_min = y_center - height / 2.0
                    y_max = y_center + height / 2.0

                    # Ensure we do not go under 0 or above 1.
                    x_min = max(x_min, 0.0)
                    x_max = min(x_max, 1.0)
                    y_min = max(y_min, 0.0)
                    y_max = min(y_max, 1.0)

                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(class_id)
                    is_difficult.append(0)
                else:
                    print("Invalid row size for {}".format(image_id))

        # If there are no labels in this image then we need to add one for the background
        # so it does not mess up all the other calcs.
        if not len(boxes):
            boxes.append([0, 0, 0.01, 0.01])
            labels.append(0)
            is_difficult.append(0)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def _find_image(self, image_id):
        img_extensions = (
        '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF')

        for ext in img_extensions:
            image_file = os.path.join(self.images_dir, "{:s}{:s}".format(image_id, ext))

            if os.path.exists(image_file):
                return image_file

        return None

    def _read_image(self, image_id):
        image_file = self._find_image(image_id)

        if image_file is None:
            raise IOError('failed to load ' + image_file)

        image = cv2.imread(str(image_file))

        if image is None or image.size == 0:
            raise IOError('failed to load ' + str(image_file))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
