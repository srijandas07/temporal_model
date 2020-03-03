from NTU_segment_reader import *
import sys, os
from keras.models import Model
import numpy as np
import keras

from models import *
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, Callback
from keras.utils import Sequence, to_categorical, multi_gpu_model
from multiprocessing import cpu_count


class CustomModelCheckpoint(Callback):

    def __init__(self, model_parallel, path):

        super(CustomModelCheckpoint, self).__init__()

        self.save_model = model_parallel
        self.path = path
        self.nb_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.nb_epoch += 1
        self.save_model.save(self.path + str(self.nb_epoch) + '.hdf5')

data_dim = 7168
data_dim_skl = 150
num_classes = 60
batch_size = 32
n_neuron = 512
n_dropout = 0.6
timesteps_seg1 = 2
timesteps_seg2 = 3
timesteps_seg3 = 4
timesteps_skl = 5
n_neuron_skl = 150
name = sys.argv[1]
att = 'att_features_withoutSA/'


csvlogger = CSVLogger(name+'_temporal_model.csv')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor = 0.1, patience = 10)
epochs = int(sys.argv[2])
model = temporal_model(n_neuron, n_neuron_skl, n_dropout, batch_size, timesteps_seg1, timesteps_seg2, timesteps_seg3, timesteps_skl, data_dim, data_dim_skl, num_classes)
parallel_model = multi_gpu_model(model, gpus=4)
parallel_model.compile(loss = 'categorical_crossentropy', optimizer = keras.optimizers.Adam(lr=0.001, clipnorm=1), metrics = ['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001, clipnorm=1), metrics=['accuracy'])
if not os.path.exists('./weights_'+name):
    os.makedirs('./weights_'+name)
model_checkpoint = CustomModelCheckpoint(model, './weights_'+name+'/epoch_')
print('Model Compiled !!!')

train_generator = DataGenerator('../data/train_CS.txt', 'train', att, batch_size = batch_size)
val_generator = DataGenerator('../data/validation_CS.txt', 'validation', att, batch_size = batch_size)
test_generator = DataGenerator('../data/test_CS.txt', 'test', att, batch_size = batch_size)


parallel_model.fit_generator(generator=train_generator,
                    validation_data=val_generator,
                    use_multiprocessing=True,
                    epochs=epochs,
		    callbacks = [csvlogger, reduce_lr, model_checkpoint],
                    max_queue_size = 48,
                    workers=cpu_count() - 2)

print(parallel_model.evaluate_generator(generator = test_generator, use_multiprocessing=True, max_queue_size = 48, workers=cpu_count() - 2))
