data_dir = '.'

import os
import math
import imgaug
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import tensorflow as tf

from edocr2 import keras_ocr

dataset_ = keras_ocr.datasets.get_cocotext_recognizer_dataset()
dataset = [d for d in dataset_ if d[2] != '' and len(d[1]) == 4]

train, validation = sklearn.model_selection.train_test_split(
    dataset, train_size=0.8, random_state=42
)

generator_kwargs = {'width': 640, 'height': 640}
training_image_generator = keras_ocr.datasets.get_detector_image_generator(
    labels=train,
    **generator_kwargs
)
a = next(training_image_generator)
validation_image_generator = keras_ocr.datasets.get_detector_image_generator(
    labels=validation,
    **generator_kwargs
)

detector = keras_ocr.detection.Detector()

batch_size = 8
training_generator, validation_generator = [
    detector.get_batch_generator(
        image_generator=image_generator, batch_size=batch_size
    ) for image_generator in
    [training_image_generator, validation_image_generator]
]
b = next(training_generator)
detector.model.fit(
    training_generator,
    steps_per_epoch=math.ceil(len(train) / batch_size),
    epochs=1,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(restore_best_weights=True, patience=5),
        tf.keras.callbacks.CSVLogger(os.path.join(data_dir, 'edocr2/models/detector_coco.csv')),
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(data_dir, 'edocr2/models/detector_coco.keras'))
    ],
    validation_data=validation_generator,
    validation_steps=math.ceil(len(validation) / batch_size)
)