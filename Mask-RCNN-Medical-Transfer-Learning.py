#!/usr/bin/env python
# coding:utf-8
# Author: ASU --<andrei.suiu@gmail.com>
# Purpose: 
# Created: 7/12/2019
from os.path import abspath, join, basename

import keras
from pandas import DataFrame

__author__ = 'ASU'

import gc
import glob
import os
import random
import sys
import warnings

import cv2
import matplotlib.pyplot as plt
import mrcnn.model as modellib
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom
from imgaug import augmenters as iaa
from mrcnn import utils
from mrcnn import visualize
from mrcnn.config import Config
from skimage.morphology import label
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook

debug = False

warnings.filterwarnings("ignore")

RT = 'D:\work\Projects\ML\Pneumothorax'
to_root = lambda _: os.path.join(RT, _)
sys.path.insert(0, to_root('/kaggle/input/siim-acr-pneumothorax-segmentation'))
from mask_functions import rle2mask, mask2rle

# %%
DATA_DIR = to_root(r'D:\kaggle\input\siim-acr-pneumothorax-segmentation-data\pneumothorax')
DATA_DIR2 = to_root(r'D:\kaggle\input\data2\pneumothorax')

# Directory to save logs and trained model
ROOT_DIR = to_root('D:\kaggle\working')
# %%
# Import Mask RCNN
# sys.path.append(os.path.join(ROOT_DIR, 'Mask_RCNN'))  # To find local version of the library

### Load Pneumonia pre-trained weights
# %%
# get model with best validation score: https://www.kaggle.com/hmendonca/mask-rcnn-and-coco-transfer-learning-lb-0-155/
WEIGHTS_PATH = to_root("mask_rcnn_pneumonia.h5")
# The following parameters have been selected to reduce running time for demonstration purposes
# These are not optimal

IMAGE_DIM = 512


class MaskRCNNCached(modellib.MaskRCNN):
    FIRST_MODEL_FNAME = 'first_model.h5'

    def __init__(self, mode, config, model_dir, cache_file_name=FIRST_MODEL_FNAME):
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        if os.path.exists(cache_file_name):
            self.keras_model = keras.models.load_model(cache_file_name)
        else:
            self.keras_model = self.build(mode=mode, config=config)
            self.keras_model.save(cache_file_name)


class DetectorConfig(Config):
    # Give the configuration a recognizable name
    NAME = 'Pneumothorax'

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    BACKBONE = 'resnet50'

    NUM_CLASSES = 2  # background and pneumothorax classes

    IMAGE_MIN_DIM = IMAGE_DIM
    IMAGE_MAX_DIM = IMAGE_DIM
    RPN_ANCHOR_SCALES = (32, 64, 128, 256)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 12
    DETECTION_MAX_INSTANCES = 4
    DETECTION_MIN_CONFIDENCE = 0.90
    DETECTION_NMS_THRESHOLD = 0.1

    STEPS_PER_EPOCH = 20 if debug else 320
    VALIDATION_STEPS = 10 if debug else 100

    ## balance out losses
    LOSS_WEIGHTS = {
        "rpn_class_loss": 10.0,
        "rpn_bbox_loss": 0.6,
        "mrcnn_class_loss": 6.0,
        "mrcnn_bbox_loss": 1.0,
        "mrcnn_mask_loss": 2.4
    }


class DetectorDatasetDF(utils.Dataset):
    def __init__(self, image_annotations: DataFrame, orig_height, orig_width):
        super().__init__(self)

        # Add classes
        self.add_class('pneumothorax', 1, 'Pneumothorax')

        # add images
        for i, row in image_annotations.iterrows():
            image_id = row["ImageId"]
            path = row["fp"]
            # annotations = row['EncodedPixels']
            annotations = image_annotations.query(f"ImageId=='{image_id}'")['EncodedPixels']
            self.add_image('pneumothorax', image_id=image_id, path=path,
                           annotations=annotations, orig_height=orig_height, orig_width=orig_width)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

    def load_image(self, image_id):
        info = self.image_info[image_id]
        fp = info['path']
        ds = pydicom.read_file(fp)
        image = ds.pixel_array
        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
        return image

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        annotations = info['annotations']
        #         print(image_id, annotations)
        count = len(annotations)
        if count == 0 or (count == 1 and annotations.values[0] == ' -1'):  # empty annotation
            mask = np.zeros((info['orig_height'], info['orig_width'], 1), dtype=np.uint8)
            class_ids = np.zeros((1,), dtype=np.int32)
        else:
            mask = np.zeros((info['orig_height'], info['orig_width'], count), dtype=np.uint8)
            class_ids = np.zeros((count,), dtype=np.int32)
            for i, a in enumerate(annotations):
                mask[:, :, i] = rle2mask(a, info['orig_height'], info['orig_width']).T
                class_ids[i] = 1
        return mask.astype(np.bool), class_ids.astype(np.int32)


class DetectorDataset(utils.Dataset):
    """Dataset class for training our dataset.
    """

    def __init__(self, image_fps, image_annotations, orig_height, orig_width):
        super().__init__(self)

        # Add classes
        self.add_class('pneumothorax', 1, 'Pneumothorax')

        # add images
        for i, fp in enumerate(image_fps):
            image_id = fp.split(os.sep)[-1][:-4]
            annotations = image_annotations.query(f"ImageId=='{image_id}'")['EncodedPixels']
            self.add_image('pneumothorax', image_id=i, path=fp,
                           annotations=annotations, orig_height=orig_height, orig_width=orig_width)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

    def load_image(self, image_id):
        info = self.image_info[image_id]
        fp = info['path']
        ds = pydicom.read_file(fp)
        image = ds.pixel_array
        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
        return image

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        annotations = info['annotations']
        #         print(image_id, annotations)
        count = len(annotations)
        if count == 0 or (count == 1 and annotations.values[0] == ' -1'):  # empty annotation
            mask = np.zeros((info['orig_height'], info['orig_width'], 1), dtype=np.uint8)
            class_ids = np.zeros((1,), dtype=np.int32)
        else:
            mask = np.zeros((info['orig_height'], info['orig_width'], count), dtype=np.uint8)
            class_ids = np.zeros((count,), dtype=np.int32)
            for i, a in enumerate(annotations):
                mask[:, :, i] = rle2mask(a, info['orig_height'], info['orig_width']).T
                class_ids[i] = 1
        return mask.astype(np.bool), class_ids.astype(np.int32)


class InferenceConfig(DetectorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def multi_rle_encode(img, **kwargs):
    ''' Encode disconnected regions as separated masks
    '''
    labels = label(img)
    if img.ndim > 2:
        return [rle2mask(np.sum(labels == k, axis=2), **kwargs) for k in
                np.unique(labels[labels > 0])]
    else:
        return [rle2mask(labels == k, **kwargs) for k in np.unique(labels[labels > 0])]


def masks_as_image(rle_list, shape):
    # Take the individual masks and create a single mask array
    all_masks = np.zeros(shape, dtype=np.uint8)
    for mask in rle_list:
        if isinstance(mask, str) and mask != '-1':
            all_masks |= rle2mask(mask, shape[0], shape[1]).T.astype(bool)
    return all_masks


def masks_as_color(rle_list, shape):
    # Take the individual masks and create a color mask array
    all_masks = np.zeros(shape, dtype=np.float)
    scale = lambda x: (len(rle_list) + x + 1) / (
            len(rle_list) * 2)  ## scale the heatmap image to shift
    for i, mask in enumerate(rle_list):
        if isinstance(mask, str) and mask != '-1':
            all_masks[:, :] += scale(i) * rle2mask(mask, shape[0], shape[1]).T
    return all_masks


def predict(image_fps, config: Config, filepath='submission.csv'):
    min_conf = config.DETECTION_MIN_CONFIDENCE
    # assume square image
    resize_factor = ORIG_SIZE / config.IMAGE_SHAPE[0]
    with open(filepath, 'w') as file:
        file.write("ImageId,EncodedPixels\n")

        for fp in tqdm_notebook(image_fps):
            image_id = fp.split('/')[-1][:-4]
            maks_written = 0

            if image_id in positives.index:
                ds = pydicom.read_file(fp)
                image = ds.pixel_array
                # If grayscale. Convert to RGB for consistency.
                if len(image.shape) != 3 or image.shape[2] != 3:
                    image = np.stack((image,) * 3, -1)
                image, window, scale, padding, crop = utils.resize_image(
                    image,
                    min_dim=config.IMAGE_MIN_DIM,
                    min_scale=config.IMAGE_MIN_SCALE,
                    max_dim=config.IMAGE_MAX_DIM,
                    mode=config.IMAGE_RESIZE_MODE)

                results = model.detect([image])
                r = results[0]

                #                 assert( len(r['rois']) == len(r['class_ids']) == len(r['scores']) )
                n_positives = positives.loc[image_id].N
                num_instances = min(len(r['rois']), n_positives)

                for i in range(num_instances):
                    if r['scores'][i] > min_conf and np.sum(r['masks'][..., i]) > 1:
                        mask = r['masks'][..., i].T * 255
                        #                         print(len(r['rois']), r['scores'][i], r['rois'][i], r['masks'].shape)
                        #                         print(mask.shape, np.max(mask), np.stack((mask,) * 3, -1).shape)
                        mask, _, _, _, _ = utils.resize_image(
                            np.stack((mask,) * 3, -1),  # requires 3 channels
                            min_dim=ORIG_SIZE,
                            min_scale=config.IMAGE_MIN_SCALE,
                            max_dim=ORIG_SIZE,
                            mode=config.IMAGE_RESIZE_MODE)
                        mask = (mask[..., 0] > 0) * 255
                        #                         print(mask.shape)
                        #                         plt.imshow(mask, cmap=get_cmap('jet'))
                        file.write(image_id + "," + mask2rle(mask, ORIG_SIZE, ORIG_SIZE) + "\n")
                        maks_written += 1

                # fill up remaining masks
                for i in range(n_positives - maks_written):
                    padding = 88750
                    file.write(image_id + f",{padding} {ORIG_SIZE * ORIG_SIZE - padding * 2}\n")
                    maks_written += 1

            #                 assert n_positives == maks_written
            #                 print(image_id, n_positives, num_instances, maks_written)

            if maks_written == 0:
                file.write(image_id + ",-1\n")  ## no pneumothorax


def visualize_test():
    ids_with_mask = sub[sub.EncodedPixels != '-1'].ImageId.values
    fp = random.choice([fp for fp in test_names if fp.split('/')[-1][:-4] in ids_with_mask])
    #     import pdb; pdb.set_trace()

    # original image
    image_id = fp.split('/')[-1][:-4]
    ds = pydicom.read_file(fp)
    image = ds.pixel_array

    # If grayscale. Convert to RGB for consistency.
    if len(image.shape) != 3 or image.shape[2] != 3:
        image = np.stack((image,) * 3, -1)

    # assume square image
    resize_factor = 1  ## ORIG_SIZE / config.IMAGE_SHAPE[0]

    # Detect on full size test images (without resizing)
    results = model.detect([image])
    r = results[0]
    for bbox in r['rois']:
        #         print(bbox)
        x1 = int(bbox[1] * resize_factor)
        y1 = int(bbox[0] * resize_factor)
        x2 = int(bbox[3] * resize_factor)
        y2 = int(bbox[2] * resize_factor)
        cv2.rectangle(image, (x1, y1), (x2, y2), (77, 255, 9), 3, 1)
        width = x2 - x1
        height = y2 - y1
    #         print("x {} y {} h {} w {}".format(x1, y1, width, height))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.set_title(image_id)
    ax1.imshow(image)
    ax2.set_title(f"{len(r['rois'])} masks predicted again")
    if len(r['rois']) > 0:
        ax2.imshow(r['masks'].max(-1))  # get max (overlap) between all masks in this prediction
    ax3.set_title(f"{np.count_nonzero(image_id == ids_with_mask)} masks in csv")
    ax3.imshow(masks_as_color(sub.query(f"ImageId=='{image_id}'")['EncodedPixels'].values,
                              (ORIG_SIZE, ORIG_SIZE)))


#     print(f"ImageId=='{image_id}'", sub.query(f"ImageId=='{image_id}'")['EncodedPixels'])
def load_sample_data(data_dir: str):
    train_names = sorted(glob.glob(f'{data_dir}/*.dcm'))

    # training dataset
    SEGMENTATION = data_dir + '/train-rle-sample.csv'
    image_annotations = pd.read_csv(SEGMENTATION)
    # get rid of damn space in column name
    image_annotations.columns = ['ImageId', 'EncodedPixels']
    # %%
    # over-sample pneumothorax
    pneumothorax_anns = image_annotations[
        image_annotations.EncodedPixels != '-1'].ImageId.unique().tolist()
    print(f'Positive samples: {len(pneumothorax_anns)}/{len(image_annotations.ImageId.unique())}'
          f'{100 * len(pneumothorax_anns) / len(image_annotations.ImageId.unique()): .2f} % ')
    # %%
    ## use only pneumothorax images
    file_paths = [abspath(join(data_dir, fp)) for fp in image_annotations["ImageId"]]
    image_annotations["fp"] = file_paths
    # pozitive_image_paths = [abspath(join(data_dir,fp)) for fp in pneumothorax_anns]

    return image_annotations


def load_data(data_dir: str, first=None):
    train_dicom_dir = os.path.join(data_dir, 'dicom-images-train')
    test_dicom_dir = os.path.join(data_dir, 'dicom-images-test')

    train_glob = f'{train_dicom_dir}/*/*/*.dcm'
    test_glob = f'{test_dicom_dir}/*/*/*.dcm'

    train_names = [abspath(fp) for fp in glob.glob(train_glob)]
    test_names = [abspath(fp) for fp in glob.glob(test_glob)]

    print(len(train_names), len(test_names))
    # print(train_names[0], test_names[0])
    # !ls -l {os.path.join(train_dicom_dir, train_names[0])}
    # %%
    # training dataset
    SEGMENTATION = data_dir + '/train-rle.csv'
    image_annotations = pd.read_csv(SEGMENTATION)

    if bool(first) is not None:
        image_annotations = image_annotations.head(first)
        selected_ids = set(image_annotations['ImageId'])
        train_names = [name for name in train_names if basename(name).split('.dcm')[0] in selected_ids]
    # image_annotations.head(10)
    # %%
    # get rid of damn space in column name
    image_annotations.columns = ['ImageId', 'EncodedPixels']
    # %%
    # over-sample pneumothorax
    pneumothorax_anns = image_annotations[image_annotations.EncodedPixels != (' ' + '-1')].ImageId.unique().tolist()
    print(f'Positive samples: {len(pneumothorax_anns)}/{len(image_annotations.ImageId.unique())}'
          f'{100 * len(pneumothorax_anns) / len(image_annotations.ImageId.unique()): .2f} % ')
    # %%
    ## use only pneumothorax images
    pneumothorax_fps_train = [os.path.abspath(fp) for fp in train_names if
                              os.path.abspath(fp).split(os.sep)[-1][:-4] in pneumothorax_anns]

    return pneumothorax_anns, pneumothorax_fps_train, train_names, test_names, image_annotations


def sampled_train():
    image_annotations = load_sample_data("sample-data")

    ORIG_SIZE = 1024

    ## Create and prepare the training dataset
    # %%
    # prepare the training dataset
    dataset_train = DetectorDatasetDF(image_annotations, ORIG_SIZE, ORIG_SIZE)
    dataset_train.prepare()


def normal_train():
    pneumothorax_anns, pneumothorax_fps_train, train_names, test_names, image_annotations = load_data(DATA_DIR, first=100)

    image_fps_train, image_fps_val = train_test_split(pneumothorax_fps_train, test_size=0.1,
                                                      random_state=42)

    if debug:
        print('DEBUG subsampling from:', len(image_fps_train), len(image_fps_val),
              len(test_names))
        image_fps_train = image_fps_train[:150]
        image_fps_val = test_names[:150]
    #     test_image_fps = test_names[:150]

    print(len(image_fps_train), len(image_fps_val), len(test_names))

    ## Examine the annotation data
    ds = pydicom.read_file(train_names[0])  # read dicom image from filepath
    image = ds.pixel_array  # get image array
    # %%
    # Original image size: 1024 x 1024
    ORIG_SIZE = 1024

    ## Create and prepare the training dataset
    # %%
    # prepare the training dataset
    dataset_train = DetectorDataset(image_fps_train, image_annotations, ORIG_SIZE, ORIG_SIZE)
    dataset_train.prepare()
    # %%
    # prepare the validation dataset
    dataset_val = DetectorDataset(image_fps_val, image_annotations, ORIG_SIZE, ORIG_SIZE)
    dataset_val.prepare()

    ## Display a random image with bounding boxes
    # %%
    # Load and display random sample and their bounding boxes

    class_ids = [0]
    i = 0
    while class_ids[0] == 0:  ## look for a mask
        image_id = random.choice(dataset_val.image_ids)
        image_fp = dataset_val.image_reference(image_id)

        image = dataset_val.load_image(image_id)
        mask, class_ids = dataset_val.load_mask(image_id)
        print("Loaded %d %s %s: " % (i, class_ids, image_id), end="\r", flush=True)
        i += 1

    print(image.shape)

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    masked = np.zeros(image.shape[:2])
    for i in range(mask.shape[2]):
        masked += image[:, :, 0] * mask[:, :, i]
    plt.imshow(masked, cmap='gray')
    plt.axis('off')

    print(image_fp)
    print(class_ids)

    ## Image Augmentation
    # Image augmentation (light but constant)
    augmentation = iaa.Sequential([
        iaa.OneOf([  ## geometric transform
            iaa.Affine(
                scale={"x": (0.98, 1.02), "y": (0.98, 1.04)},
                translate_percent={"x": (-0.02, 0.02), "y": (-0.04, 0.04)},
                rotate=(-2, 2),
                shear=(-1, 1),
            ),
            iaa.PiecewiseAffine(scale=(0.001, 0.025)),
        ]),
        iaa.OneOf([  ## brightness or contrast
            iaa.Multiply((0.9, 1.1)),
            iaa.ContrastNormalization((0.9, 1.1)),
        ]),
        iaa.OneOf([  ## blur or sharpen
            iaa.GaussianBlur(sigma=(0.0, 0.1)),
            iaa.Sharpen(alpha=(0.0, 0.1)),
        ]),
    ])

    # test on the same image as above
    imggrid = augmentation.draw_grid(image[:, :, 0], cols=5, rows=2)
    plt.figure(figsize=(30, 12))
    _ = plt.imshow(imggrid[:, :, 0], cmap='gray')
    # get pixel statistics
    images = []
    for image_id in dataset_val.image_ids:
        image = dataset_val.load_image(image_id)
        images.append(image)

    images = np.array(images)

    config = DetectorConfig()
    # config.display()

    config.MEAN_PIXEL = images.mean(axis=(0, 1, 2)).tolist()
    VAR_PIXEL = images.var()
    print(config.MEAN_PIXEL, VAR_PIXEL)
    model = modellib.MaskRCNN(mode='training', config=config, model_dir=ROOT_DIR)

    # Train the Mask-RCNN Model
    LEARNING_RATE = 0.0006
    # %%
    ## train heads with higher lr to speedup the learning
    model.train(dataset_train, dataset_val,
                learning_rate=LEARNING_RATE * 10,
                epochs=1,
                layers='heads',
                augmentation=None)  ## no need to augment yet

    history = model.keras_model.history.history
    # %%
    model.train(dataset_train, dataset_val,
                learning_rate=LEARNING_RATE,
                epochs=3 if debug else 11,
                layers='all',
                augmentation=augmentation)

    new_history = model.keras_model.history.history
    for k in new_history: history[k] = history[k] + new_history[k]
    # %%
    model.train(dataset_train, dataset_val,
                learning_rate=LEARNING_RATE / 2,
                epochs=4 if debug else 18,
                layers='all',
                augmentation=augmentation)

    new_history = model.keras_model.history.history
    for k in new_history: history[k] = history[k] + new_history[k]
    # %%
    epochs = range(1, len(history['loss']) + 1)
    pd.DataFrame(history, index=epochs)
    # %%
    plt.figure(figsize=(21, 11))

    plt.subplot(231)
    plt.plot(epochs, history["loss"], label="Train loss")
    plt.plot(epochs, history["val_loss"], label="Valid loss")
    plt.legend()
    plt.subplot(232)
    plt.plot(epochs, history["rpn_class_loss"], label="Train RPN class ce")
    plt.plot(epochs, history["val_rpn_class_loss"], label="Valid RPN class ce")
    plt.legend()
    plt.subplot(233)
    plt.plot(epochs, history["rpn_bbox_loss"], label="Train RPN box loss")
    plt.plot(epochs, history["val_rpn_bbox_loss"], label="Valid RPN box loss")
    plt.legend()
    plt.subplot(234)
    plt.plot(epochs, history["mrcnn_class_loss"], label="Train MRCNN class ce")
    plt.plot(epochs, history["val_mrcnn_class_loss"], label="Valid MRCNN class ce")
    plt.legend()
    plt.subplot(235)
    plt.plot(epochs, history["mrcnn_bbox_loss"], label="Train MRCNN box loss")
    plt.plot(epochs, history["val_mrcnn_bbox_loss"], label="Valid MRCNN box loss")
    plt.legend()
    plt.subplot(236)
    plt.plot(epochs, history["mrcnn_mask_loss"], label="Train Mask loss")
    plt.plot(epochs, history["val_mrcnn_mask_loss"], label="Valid Mask loss")
    plt.legend()

    plt.show()
    # %%
    best_epoch = np.argmin(history["val_loss"])
    score = history["val_loss"][best_epoch]
    print(f'Best Epoch:{best_epoch + 1} val_loss:{score}')
    # %%
    # select trained model
    dir_names = next(os.walk(model.model_dir))[1]
    key = config.NAME.lower()
    dir_names = filter(lambda f: f.startswith(key), dir_names)
    dir_names = sorted(dir_names)

    if not dir_names:
        import errno

        raise FileNotFoundError(
            errno.ENOENT,
            "Could not find model directory under {}".format(model.model_dir))

    fps = []
    # Pick last directory
    for d in dir_names:
        dir_name = os.path.join(model.model_dir, d)
        # Find checkpoints
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            raise Exception(f'No weight files in {dir_name}')
        if best_epoch < len(checkpoints):
            checkpoint = checkpoints[best_epoch]
        else:
            checkpoint = checkpoints[-1]
        fps.append(os.path.join(dir_name, checkpoint))

    model_path = sorted(fps)[-1]
    print('Found model {}'.format(model_path))

    inference_config = InferenceConfig()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode='inference',
                              config=inference_config,
                              model_dir=ROOT_DIR)

    # Load trained weights (fill in path to trained weights here)
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    # set color for class
    def get_colors_for_class_ids(class_ids):
        colors = []
        for class_id in class_ids:
            if class_id == 1:
                colors.append((.941, .204, .204))
        return colors

    # Show few example of ground truth vs. predictions on the validation dataset
    dataset = dataset_val
    pneumothorax_ids_val = [fp.split('/')[-1][:-4] for fp in image_fps_val]
    pneumothorax_ids_val = [i for i, id in enumerate(pneumothorax_ids_val) if
                            id in pneumothorax_anns]
    fig = plt.figure(figsize=(10, 40))

    for i in range(8):
        image_id = random.choice(pneumothorax_ids_val)

        original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset_val, inference_config,
                                   image_id, use_mini_mask=False)

        #     print(original_image.shape)
        plt.subplot(8, 2, 2 * i + 1)
        visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                                    dataset.class_names,
                                    colors=get_colors_for_class_ids(gt_class_id), ax=fig.axes[-1])

        plt.subplot(8, 2, 2 * i + 2)
        results = model.detect([original_image])  # , verbose=1)
        r = results[0]
        visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                    dataset.class_names, r['scores'],
                                    colors=get_colors_for_class_ids(r['class_ids']),
                                    ax=fig.axes[-1])

    ## Basic Model Weigths Analysis

    # %%
    # Show stats of all trainable weights
    visualize.display_weight_stats(model)
    ### Click to expand output
    # %%
    # from https://github.com/matterport/Mask_RCNN/blob/master/samples/coco/inspect_weights.ipynb
    # Pick layer types to display
    LAYER_TYPES = ['Conv2D', 'Dense', 'Conv2DTranspose']
    # Get layers
    layers = model.get_trainable_layers()
    layers = list(filter(lambda l: l.__class__.__name__ in LAYER_TYPES,
                         layers))
    # Display Histograms
    fig, ax = plt.subplots(len(layers), 2, figsize=(10, 3 * len(layers)), gridspec_kw={"hspace": 1})
    for l, layer in enumerate(layers):
        weights = layer.get_weights()
        for w, weight in enumerate(weights):
            tensor = layer.weights[w]
            ax[l, w].set_title(f'Layer:{l}.{w} {tensor.name}')
            _ = ax[l, w].hist(weight[w].flatten(), 50)
    sub = pd.read_csv('/kaggle/input/siim-acr-pneumothorax-segmentation/sample_submission.csv')

    # based on https://www.kaggle.com/raddar/better-sample-submission Will this work in stage 2?
    positives = sub.groupby('ImageId').ImageId.count().reset_index(name='N').set_index('ImageId')
    positives = positives.loc[
        positives.N > 1]  # find image id's with more than 1 row -> has pneumothorax mask!

    positives.head()

    # %%
    # Make predictions on test images, write out submission file

    # %%
    submission_fp = os.path.join(ROOT_DIR, 'submission.csv')
    predict(test_names, filepath=submission_fp)
    print(submission_fp)
    # %%
    sub = pd.read_csv(submission_fp)
    print((sub.EncodedPixels != '-1').sum(), sub.ImageId.size, sub.ImageId.nunique())
    print(sub.EncodedPixels.nunique(), (sub.EncodedPixels != '-1').sum() / sub.ImageId.nunique())

    print('Unique samples:\n', sub.EncodedPixels.drop_duplicates()[:6])
    sub.head(10)

    # %%
    # show a few test image detection example

    for i in range(8):
        visualize_test()
    # %%
    os.chdir(ROOT_DIR)


if __name__ == "__main__":
    gc.enable()  # memory is tight

    # from tensorflow.python.client import device_lib
    #
    # print(str(device_lib.list_local_devices()))
    # assert 'GPU' in str(device_lib.list_local_devices())
    #
    # # confirm Keras sees the GPU
    # from keras import backend
    #
    # assert len(backend.tensorflow_backend._get_available_gpus()) > 0

    # montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)

    normal_train()
    # sampled_train()
