import sys, os
sys.path.insert(0, '..')
from keras.models import Model
import numpy as np
import keras
import pandas as pd, csv
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, Callback
from i3d_inception import Inception_Inflated3d
from keras.layers import Flatten
from keras.models import load_model
from keras.utils import Sequence, to_categorical, multi_gpu_model
from keras.optimizers import SGD
from multiprocessing import cpu_count
from NTU_segment_Loader import *
import argparse

parser = argparse.ArgumentParser(description='Segment Feature Extraction')
parser.add_argument('name', help='Train/Test/Validation')
parser.add_argument('split_path', help='Text file containing videos for each split(in data folder)')
parser.add_argument('frames_path', help='Path to video frames')
parser.add_argument('model_path', help='Path to pre-trained model')
parser.add_argument('granularity', type=int, help='Granularity for division')
args = parser.parse_args()

optim = SGD(lr = 0.01, momentum = 0.9)
batch_size = 16
model_3 = load_model(args.model_path)
model_3.compile(loss = 'categorical_crossentropy', optimizer = optim, metrics = ['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor = 0.1, patience = 10)
model_3d = Model(inputs=model_3.input, outputs=model_3.get_layer('global_avg_pool').output)
parallel_model = multi_gpu_model(model_3d, gpus=4)
parallel_model.compile(loss = 'categorical_crossentropy', optimizer = optim, metrics = ['accuracy'])
model_3d.compile(loss = 'categorical_crossentropy', optimizer = optim, metrics = ['accuracy'])

generators = list()
for index in range(args.granularity):
    test_generator = DataLoader_video_test_segment(args.split_path, args.frames_path, index, args.granularity, batch_size = batch_size)
    generators.append(test_generator)

if args.name=='Train':
    filename='train_CS'
elif args.name=='Test':
    filename='test_CS'
else:
    filename='validation_CS'

for index in range(args.granularity):
    features = np.squeeze(parallel_model.predict_generator(generators[index], max_queue_size = 48, workers = cpu_count() - 2, use_multiprocessing = True))
    features_flattened = features.reshape(len(features), 7*1024)
    np.save('../../data/NTU_CS/att_features_withoutSA/att_features_{}/{}_{}.csv.gz'.format(args.granularity, filename, index+1), features_flattened)

