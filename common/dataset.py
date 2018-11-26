import os
import numpy as np
import pickle
from chainer.dataset import dataset_mixin
from PIL import Image

DATA_DIR = "./common/dataset/facades/base"
PROCESSED_DIR = "./common/dataset/facades"


class FacadeDataset(dataset_mixin.DatasetMixin):

    def __init__(self, datadir=DATA_DIR, processed_dir=PROCESSED_DIR, data_range=(1, 300)):
        print("load dataset start")
        print("    from: %s" % datadir)
        print("    range: [%d, %d)" % (data_range[0], data_range[1]))
        self.datadir = os.path.join(processed_dir, "train.dump")
        try:
            with open(self.datadir, "rb") as pkl:
                self.dataset = pickle.load(pkl)
        except FileNotFoundError:
            self.dataset = []
            for i in range(data_range[0], data_range[1]):
                image_path = os.path.join(datadir, "cmp_b%04d.jpg" % i)
                label_path = os.path.join(datadir, "cmp_b%04d.png" % i)
                img = Image.open(image_path)
                label = Image.open(label_path)

                w, h = img.size

                r = 286 / float(min(w, h))

                # resize images so that min(w, h) = 286
                img = img.resize((int(r * w), int(r * h)), Image.BILINEAR)
                label = label.resize((int(r * w), int(r * h)), Image.NEAREST)

                img = np.asarray(img).astype("f").transpose(2, 0, 1) / 128.0 - 1
                label_ = np.asarray(label) - 1
                # express condition as one-hot channel
                label = np.zeros((12, img.shape[1], img.shape[2])).astype("i")
                for j in range(12):
                    label[j] = (label_ == j)
                self.dataset.append((img, label))
            with open(self.datadir, "wb") as pkl:
                pickle.dump(self.dataset, pkl)
        print("load datset done")

    def __len__(self):
        return len(self.dataset)

    def get_example(self, i, cropped_width=256):
        _, h, w = self.dataset[i][0].shape
        x_l = np.random.randint(0, w - cropped_width)
        x_r = x_l + cropped_width
        y_l = np.random.randint(0, h - cropped_width)
        y_r = y_l + cropped_width

        # we want to convert image .png to .jpg
        return self.dataset[i][1][:, y_l:y_r, x_l:x_r], self.dataset[i][0][:, y_l:y_r, x_l:x_r]


class FacadeValidDataset(dataset_mixin.DatasetMixin):

    def __init__(self, datadir=DATA_DIR, processed_dir=PROCESSED_DIR, data_range=(300, 380)):
        print("load test dataset start")
        print("    from: %s" % datadir)
        print("    range: [%d, %d)" % (data_range[0], data_range[1]))
        self.datadir = os.path.join(processed_dir, "test.dump")
        try:
            with open(self.datadir, "rb") as pkl:
                self.dataset = pickle.load(pkl)
        except FileNotFoundError:
            self.dataset = []
            for i in range(data_range[0], data_range[1]):
                label_path = os.path.join(datadir, "cmp_b%04d.png" % i)
                label = Image.open(label_path)

                w, h = label.size

                r = 286 / float(min(w, h))
                # resize images so that min(w, h) = 286
                label = label.resize((int(r * w), int(r * h)), Image.NEAREST)

                label_ = np.asarray(label) - 1
                # express condition as one-hot channel
                label = np.zeros((12, label_.shape[0], label_.shape[1])).astype("i")
                for j in range(12):
                    label[j] = (label_ == j)
                self.dataset.append(label)
            with open(self.datadir, "wb") as pkl:
                pickle.dump(self.dataset, pkl)
        print("load datset done")

    def __len__(self):
        return len(self.dataset)

    def get_example(self, i, cropped_width=256):
        # we want to convert image .png to .jpg
        _, h, w = self.dataset[i].shape
        x_l = np.random.randint(0, w - cropped_width)
        x_r = x_l + cropped_width
        y_l = np.random.randint(0, h - cropped_width)
        y_r = y_l + cropped_width
        return self.dataset[i][:, y_l:y_r, x_l:x_r]
