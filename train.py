from generate_data import generate_train_validate_data_split_names, get_image_data_from_name, generate_test_names, RSNADataset,TRAINING_DATA_PATH,TESTING_DATA_PATH
import matplotlib.pyplot as plt
import numpy as np
import model as modellib
import torch
from config import Config
from model import log
import visualize
import pandas as pd
from tqdm import tqdm
import pickle as pkl
from model import Mask, Classifier
import torch.nn as nn
from coco import CocoConfig
from itertools import chain

class RSNAConfig(Config):
    NAME="RSNA"
    GPU_COUNT = 1
    IMAGES_PER_GPU= 2
    NUM_CLASSES = 1 + 1
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 1024
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    TRAIN_ROIS_PER_IMAGE = 32
    STEPS_PER_EPOCH = 2000
    VALIDATION_STEPS = 50
    LEARNING_RATE = 0.001

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


class PretrainedMaskRCNN (modellib.MaskRCNN):
    def __init__(self,kind = "COCO", config = None, model_dir =None):
        self.config=config
        self.model_dir = model_dir
        self.previous_preloaded_model = self.find_last()
        if kind == "COCO":
            pretrained_model_path = "./mask_rcnn_coco.pth"
            cfg = CocoConfig()
            cfg.NAME = config.NAME
        else:
            raise NotImplementedError("Provide a valid pretrained model name")
        super().__init__(cfg, model_dir)
        self.load_weights(pretrained_model_path)
        # change classifier
        self.classifier = Classifier(256, config.POOL_SIZE, config.IMAGE_SHAPE, config.NUM_CLASSES)
        # change mask
        self.mask = Mask(256, config.MASK_POOL_SIZE, config.IMAGE_SHAPE, config.NUM_CLASSES)
        self.config = config
        for m in chain(self.classifier.modules(), self.mask.modules()):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_().cuda()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
        return

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    config = RSNAConfig()
    trains, validates = generate_train_validate_data_split_names(0.1)
    Train = RSNADataset()
    Train.initialize(trains)
    Train.prepare()
    Validate = RSNADataset()
    Validate.initialize(validates)
    Validate.prepare()
    print ("generated these two: {} {}".format(Train, Validate))



    if 1:
        model = PretrainedMaskRCNN(config = config, model_dir = "./model").cuda()
        print (model)
        model_path = model.previous_preloaded_model
        print("Loading last run weights from ", model_path)
        model.load_weights(model_path[1])
        # print ("Training the mask and classifier with large training steps")
        # model.train_model(Train, Validate, 
        #             learning_rate=config.LEARNING_RATE*2,
        #             epochs=20, 
        #             layers="tails")
        print ("Training the heads with normal training steps")
        model.train_model(Train, Validate, 
                    learning_rate=config.LEARNING_RATE,
                    epochs=20, 
                    layers="heads")
    if 0:
        print ("Training all networks with small training steps")
        model.train_model(Train, Validate, 
                    learning_rate=config.LEARNING_RATE/10,
                    epochs=100, 
                    layers="all")
                

    if 0:
        image_ids = np.random.choice([tid for tid in Train.image_ids if Train.load_mask(tid)[1][0]==2], 4)
        for image_id in image_ids:
            image = Train.load_image(image_id)
            mask, class_ids = Train.load_mask(image_id)
            print (mask.shape, class_ids, image.shape)
            visualize.display_top_masks(image, mask, class_ids, Train.class_names, limit = len(class_ids))

    class InferenceConfig(RSNAConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    inference_config = InferenceConfig()

    if 0:
        # Load trained weights
            # Recreate the model in inference mode
        model = modellib.MaskRCNN(mode="inference", 
                                config=inference_config,
                                model_dir="./model")

        # Get path to saved weights
        # Either set a specific path or find last trained weights
        # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
        # model_path = model.find_last()
        model.load_weights(model_path, by_name=True)
        iid = np.random.choice(Validate.image_ids)
        original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(Validate, inference_config, 
                            iid, use_mini_mask=False)

        results = model.detect([original_image], verbose=1)
        r = results[0]
        log("original_image", original_image)
        log("image_meta", image_meta)
        print (gt_class_id)
        if len(r["class_ids"]) == 0 or len(gt_class_id) == 0: pass
        visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                                Train.class_names, figsize=(8, 8))
        visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                                Train.class_names, r['scores'], ax=get_ax())


    if 0:
        model = modellib.MaskRCNN(mode="inference", 
                                config=inference_config,
                                model_dir="./model")
        model_path = model.find_last()
        model.load_weights(model_path, by_name=True)
        ims = []
        fns = generate_test_names()
        results=[]
        for fname in tqdm(fns,"testing data"):
            im = get_image_data_from_name(fname,TESTING_DATA_PATH)
            im3 = np.stack((im,)*3, -1)
            result = model.detect([im3])
            res = (fname, result[0]["rois"], result[0]["scores"])
            results.append(res)
        pkl.dump(results, open("./stage1test.npd","wb"))
        
    if 0:
        results = pkl.load(open("./stage1test.npd","rb"))
        ss_out = []
        for fn, rois, scores in results:
            ss=[]
            for roi, score in zip(rois, scores):
                if score > 0.9:
                    ss.append("{:.2f} {:d} {:d} {:d} {:d}".format(score, roi[0], roi[1],
                    roi[2]-roi[0], roi[3]-roi[1]))
            ss = " ".join(ss)
            ss_out.append(ss)
        df = pd.DataFrame()
        df["patientId"]=[i[0] for i in results]
        df["PredictionString"] = ss_out
        df.set_index("patientId", inplace=True)
        df.to_csv("prediction.csv")


            
