#!/usr/bin/env python
# coding:utf-8
# Author: ASU --<andrei.suiu@gmail.com>
# Purpose: 
# Created: 7/12/2019
import os
from unittest import TestCase, main

import mrcnn.model as modellib


from train_shapes_from_tutorial import MaskRCNNAuxFeatures, ShapesDataset, ShapesConfig

__author__ = 'ASU'


class TestShapesConfig(TestCase):
    def test_load_image(self):
        dataset_train = ShapesDataset()
        config = ShapesConfig()
        # 500
        dataset_train.load_shapes(5, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
        dataset_train.prepare()
        ROOT_DIR = os.path.abspath("pretrainings/")
        MODEL_DIR = os.path.join(ROOT_DIR, "logs")
        # Create model in training mode
        model = MaskRCNNAuxFeatures(mode="training", config=config,
                                  model_dir=MODEL_DIR)

        train_generator = modellib.data_generator(dataset_train, model.config, shuffle=True,
                                                  augmentation=None,
                                                  batch_size=model.config.BATCH_SIZE,
                                                  no_augmentation_sources=None)
        first_image = next(train_generator)
        self.assertTrue(first_image)


if __name__ == '__main__':
    main()
