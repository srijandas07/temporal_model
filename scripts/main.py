from NTU_segment_reader import *
import sys, os
from keras.models import Model
import numpy as np
import keras

from models import *
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, Callback
from keras.utils import Sequence, to_categorical, multi_gpu_model
from multiprocessing import cpu_count
from options import parse


class CustomModelCheckpoint(Callback):

    def __init__(self, model_parallel, path):

        super(CustomModelCheckpoint, self).__init__()

        self.save_model = model_parallel
        self.path = path
        self.nb_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.nb_epoch += 1
        self.save_model.save(self.path + str(self.nb_epoch) + '.hdf5')


if __name__ == '__main__':
    args = parse()
    csvlogger = CSVLogger(args.name+'_temporal_model.csv')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor = 0.1, patience = 10)
    model = temporal_model(args.n_neuron, args.n_neuron_skl, args.n_dropout, args.batch_size, args.timesteps_seg1, args.timesteps_seg2, args.timesteps_seg3, args.timesteps_skl, args.data_dim, args.data_dim_skl, args.num_classes)
    parallel_model = multi_gpu_model(model, gpus=4)
    parallel_model.compile(loss = 'categorical_crossentropy', optimizer = keras.optimizers.Adam(lr=args.lr, clipnorm=1), metrics = ['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=args.lr, clipnorm=1), metrics=['accuracy'])
    if not os.path.exists('./weights_'+args.name):
        os.makedirs('./weights_'+args.name)
    model_checkpoint = CustomModelCheckpoint(model, './weights_'+args.name+'/epoch_')
    print('Model Compiled !!!')

    train_generator = DataGenerator(args.train_file, 'train', args.dataset, batch_size = args.batch_size)
    val_generator = DataGenerator(args.val_file, 'validation', args.dataset, batch_size = args.batch_size)
    test_generator = DataGenerator(args.test_file, 'test', args.dataset, batch_size = args.batch_size)


    parallel_model.fit_generator(generator=train_generator,
                    validation_data=val_generator,
                    use_multiprocessing=True,
                    epochs=args.epochs,
		    callbacks = [csvlogger, reduce_lr, model_checkpoint],
                    max_queue_size = 48,
                    workers=cpu_count() - 2)

    print(parallel_model.evaluate_generator(generator = test_generator, use_multiprocessing=True, max_queue_size = 48, workers=cpu_count() - 2))
