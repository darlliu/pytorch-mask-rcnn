import os
import pydicom
import zipfile
import io
import numpy as np 
import matplotlib 
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy.random
import pandas as pd
from utils import Dataset
from sklearn.model_selection import train_test_split
TRAINING_DATA_PATH = "../stage1data/stage_1_train_images.zip"
TESTING_DATA_PATH = "../stage1data/stage_1_test_images.zip"
tdf = pd.read_csv("../stage1data/stage_1_train_labels.csv")
tdf.set_index('patientId', inplace=True)
total_dataset_size = 25000
imagesize = (1024,1024)

def generate_image_files (fname):
    with zipfile.ZipFile(fname) as zf:
        fl = zf.filelist
        total_dataset_size = len(fl)
        numpy.random.shuffle(fl)
        for fs in fl:
            f = io.BytesIO(zf.open(fs).read())
            ds = pydicom.read_file (f)
            yield ds.PatientName, ds.pixel_array

def generate_image_info (fname,skip_empty = True):
    with zipfile.ZipFile(fname) as zf:
        fl = zf.filelist
        total_dataset_size = len(fl)
        for f in fl:
            gid = f.filename.split(".")[0]
            if not skip_empty or gid not in tdf.index:
                yield gid
            else:
                rows = tdf.loc[gid]
                if type(rows) == pd.Series:
                    tgs = [rows.Target]
                else: 
                    tgs = [int(r.Target) for _,r in rows.iterrows()]
                if max(tgs) > 0:
                    yield gid

def get_image_data_from_name(imagename, fname= TRAINING_DATA_PATH):
    with zipfile.ZipFile(fname) as zf:
        f = io.BytesIO(zf.open(imagename+".dcm").read())
        ds = pydicom.read_file(f)
        return ds.pixel_array

def generate_train_validate_data():
    for ds, d in generate_image_files(TRAINING_DATA_PATH):
        row = tdf.loc[str(ds)]
        if type(row) == pd.Series:
            yield (row.x,row.y,row.width,row.height), row.Target, ds, d
        else:
            for _,r in row.iterrows():
                yield (r.x,r.y,r.width,r.height),r.Target, ds, d

def generate_train_validate_data_split(split=0.1):
    validate = []
    train = []
    val_size = split*total_dataset_size
    for idx, d in enumerate(generate_train_validate_data()):
        if len(validate) < val_size:
            validate.append(d)
        else:
            train.append(d)
        if idx % 500 == 0:
            print (idx)
    return validate, train

def generate_train_validate_data_split_names (split= 0.1):
    fnames = [i for i in generate_image_info(TRAINING_DATA_PATH)]
    train, test = train_test_split(fnames, test_size = split)
    return train, test
    

def generate_test_names():
    fnames = [i for i in generate_image_info(TESTING_DATA_PATH)]
    return fnames

class RSNADataset(Dataset):

    def initialize(self, names, sourcef = TRAINING_DATA_PATH):
        self.add_class ("RSNA", 1, "tumor")
        self.sourcef=sourcef
        for fname in names:
            self.add_image("RSNA", image_id = fname,
            path=None)
        return
    def load_image(self, image_id):
        info = self.image_info[image_id]
        image = get_image_data_from_name(info["id"], self.sourcef)
        #print (image.shape, info["id"])
        return np.stack((image,)*3, -1)

    def load_mask2(self, image_id):
        info = self.image_info[image_id]
        rows = tdf.loc[info["id"]]
        def random_range(shape, curr = None):
            xmin = np.random.randint(0, shape[0]-24)
            xmax = np.random.randint(xmin+1, shape[0])
            ymin = np.random.randint(0, shape[1]-24)
            ymax = np.random.randint(ymin+1, shape[1])
            if curr is not None:
                if np.sum(curr[xmin:xmax,ymin:ymax])>0:
                    return random_range(shape, curr)
            return xmin, xmax, ymin, ymax
        background_examples = np.random.randint(1, 4)
        targets = []
        if type(rows) == pd.Series:
            mask = np.zeros([imagesize[0], imagesize[1], 1+ background_examples],dtype=np.uint8)
            row = rows
            if int(row.Target) == 1:
                mask [int(row.x):int(row.x+row.width), int(row.y):int(row.y+row.height), 0] = 1
                targets.append(2)
            else: 
                xmin,xmax,ymin,ymax = random_range(imagesize)
                mask[xmin:xmax, ymin:ymax, 0] = 1
                targets.append(1)
            for i in range(background_examples):
                xmin,xmax,ymin,ymax = random_range(imagesize, mask[:,:,0])
                mask[xmin:xmax, ymin:ymax, i+1] = 1
                targets.append(1)
        else:
            mask = np.zeros([imagesize[0], imagesize[1], len(rows)+background_examples],dtype=np.uint8)
            for idx,(_,row) in enumerate(rows.iterrows()):
                if int(row.Target)==1:
                    mask [int(row.x):int(row.x+row.width), int(row.y):int(row.y+row.height), idx] = 1
                    targets.append(2)
                else:
                    xmin,xmax,ymin,ymax = random_range(imagesize)
                    mask[xmin:xmax, ymin:ymax, idx] = 1
                    targets.append(1)
            for i in range(background_examples):
                xmin,xmax,ymin,ymax = random_range(imagesize, mask[:,:,0])
                mask[xmin:xmax, ymin:ymax, i + len(rows)] = 1
                targets.append(1)
        return mask.astype(np.bool), np.array(targets, dtype=np.uint8)

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        rows = tdf.loc[info["id"]]
        targets = []
        mask = np.empty((0,0,0))
        if type(rows) == pd.Series:
            row = rows
            if int(row.Target) == 1:
                mask = np.zeros([imagesize[0], imagesize[1], 1],dtype=np.uint8)
                mask [int(row.x):int(row.x+row.width), int(row.y):int(row.y+row.height), 0] = 1
                targets.append(1)
        else:
            rows = rows[rows.Target==1]
            mask = np.zeros([imagesize[0], imagesize[1], len(rows)],dtype=np.uint8)
            for idx,(_,row) in enumerate(rows.iterrows()):
                mask [int(row.x):int(row.x+row.width), int(row.y):int(row.y+row.height), idx] = 1
                targets.append(1)
        return mask.astype(np.bool), np.array(targets, dtype=np.uint8)
