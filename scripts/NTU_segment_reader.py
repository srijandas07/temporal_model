import numpy as np
import keras
import pandas as pd
import random
import sys
from multiprocessing import cpu_count
import numpy as np
import glob
from skimage.io import imread
import cv2


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, path_video_files, split, att, batch_size=16):
        'Initialization'
        self.batch_size = batch_size
        self.path_video_files = path_video_files
        self.list_IDs = [i.strip() for i in open(path_video_files).readlines()]       
        self.n_classes = 60
        self.step_seg1 = 2
        self.step_seg2 = 3
        self.step_seg3 = 4
        self.dim = 7168
        self.dim_skl = 150
        self.path_skeleton = '../data/skeleton_npy/'
        self.step = 5
        self.split = split
        self.att = att
        self.dataset = 'NTU_CS/'
        self.data_seg4_t1 = np.load('../data/'+self.dataset+self.att+'att_features_4/'+self.split+'_CS_1.csv.gz.npy')
        self.data_seg4_t2 = np.load('../data/'+self.dataset+self.att+'att_features_4/'+self.split+'_CS_2.csv.gz.npy')
        self.data_seg4_t3 = np.load('../data/'+self.dataset+self.att+'att_features_4/'+self.split+'_CS_3.csv.gz.npy')
        self.data_seg4_t4 = np.load('../data/'+self.dataset+self.att+'att_features_4/'+self.split+'_CS_4.csv.gz.npy')
        self.data_seg3_t1 = np.load('../data/'+self.dataset+self.att+'att_features_3/'+self.split+'_CS_1.csv.gz.npy')
        self.data_seg3_t2 = np.load('../data/'+self.dataset+self.att+'att_features_3/'+self.split+'_CS_2.csv.gz.npy')
        self.data_seg3_t3 = np.load('../data/'+self.dataset+self.att+'att_features_3/'+self.split+'_CS_3.csv.gz.npy')
        self.data_seg2_t1 = np.load('../data/'+self.dataset+self.att+'att_features_2/'+self.split+'_CS_1.csv.gz.npy')
        self.data_seg2_t2 = np.load('../data/'+self.dataset+self.att+'att_features_2/'+self.split+'_CS_2.csv.gz.npy')
        self.list_IDs = self.list_IDs[0:len(self.data_seg4_t1)]
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X1, X2, X3, X_skl_21, X_skl_22, X_skl_31, X_skl_32, X_skl_33, X_skl_41, X_skl_42, X_skl_43, X_skl_44, y = self.__data_generation(list_IDs_temp)

        #return [X1,X2,X3,X_skl_21,X_skl_22,X_skl_31,X_skl_32,X_skl_33,X_skl_41,X_skl_42,X_skl_43,X_skl_44], y
	return [X1, X2, X3, X_skl_21, X_skl_22, X_skl_31, X_skl_32, X_skl_33, X_skl_41, X_skl_42, X_skl_43, X_skl_44], y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.split != 'test':
           np.random.shuffle(self.list_IDs)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X1 = np.empty((self.batch_size, self.step_seg1, self.dim))
        X2 = np.empty((self.batch_size, self.step_seg2, self.dim))
        X3 = np.empty((self.batch_size, self.step_seg3, self.dim))
        X_skl_21 = np.empty((self.batch_size, self.step, self.dim_skl))
        X_skl_22 = np.empty((self.batch_size, self.step, self.dim_skl))
        X_skl_31 = np.empty((self.batch_size, self.step, self.dim_skl))
        X_skl_32 = np.empty((self.batch_size, self.step, self.dim_skl))
        X_skl_33 = np.empty((self.batch_size, self.step, self.dim_skl))
        X_skl_41 = np.empty((self.batch_size, self.step, self.dim_skl))
        X_skl_42 = np.empty((self.batch_size, self.step, self.dim_skl))
        X_skl_43 = np.empty((self.batch_size, self.step, self.dim_skl))
        X_skl_44 = np.empty((self.batch_size, self.step, self.dim_skl))
        y = np.empty((self.batch_size), dtype=int)
        train_list = [i.strip() for i in open(self.path_video_files).readlines()]
        train_list = np.asarray(train_list)
        idx = []
        for i in list_IDs_temp:
            idx.append(np.where(train_list == np.asarray(i))[0][0])
            fea_seg4_t1 = np.expand_dims(self.data_seg4_t1[np.asarray(idx)], axis=1)
            fea_seg4_t2 = np.expand_dims(self.data_seg4_t2[np.asarray(idx)], axis=1)
            fea_seg4_t3 = np.expand_dims(self.data_seg4_t3[np.asarray(idx)], axis=1)
            fea_seg4_t4 = np.expand_dims(self.data_seg4_t4[np.asarray(idx)], axis=1)
            X3 = np.concatenate((fea_seg4_t1, fea_seg4_t2, fea_seg4_t3, fea_seg4_t4), axis=1)
            fea_seg3_t1 = np.expand_dims(self.data_seg3_t1[np.asarray(idx)], axis=1)
            fea_seg3_t2 = np.expand_dims(self.data_seg3_t2[np.asarray(idx)], axis=1)
            fea_seg3_t3 = np.expand_dims(self.data_seg3_t3[np.asarray(idx)], axis=1)
            X2 = np.concatenate((fea_seg3_t1, fea_seg3_t2, fea_seg3_t3), axis=1)
            fea_seg2_t1 = np.expand_dims(self.data_seg2_t1[np.asarray(idx)], axis=1)
            fea_seg2_t2 = np.expand_dims(self.data_seg2_t2[np.asarray(idx)], axis=1)
            X1 = np.concatenate((fea_seg2_t1, fea_seg2_t2), axis=1)

        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            unpadded_file = np.load(self.path_skeleton + ID + '.npy')
            unpadded_file = unpadded_file[0:len(unpadded_file)/2,:]
            origin = unpadded_file[0, 3:6]
            [row, col] = unpadded_file.shape
            origin = np.tile(origin, (row, 50))
            unpadded_file = unpadded_file - origin
            extra_frames = (len(unpadded_file) % self.step)
            if extra_frames < (self.step/2):
               padded_file = unpadded_file[0:len(unpadded_file) - extra_frames,:]
            else:
               [row, col] = unpadded_file.shape
               alpha = (len(unpadded_file)/self.step) + 1
               req_pad = np.zeros(((alpha * self.step)-row, col))
               padded_file = np.vstack((unpadded_file, req_pad))
            splitted_file = np.split(padded_file, self.step)
            splitted_file = np.asarray(splitted_file)
            row, col, width = splitted_file.shape
            sampled_file = []
            for k in range(0,self.step):
                c = np.random.choice(col,1)
                sampled_file.append(splitted_file[k,c,:])
            sampled_file = np.asarray(sampled_file)
            X_skl_21[i,] = np.squeeze(sampled_file)
            #X_skl_21 = np.asarray(X_skl_21)

	for i, ID in enumerate(list_IDs_temp):
            # Store sample
            unpadded_file = np.load(self.path_skeleton + ID + '.npy')
            unpadded_file = unpadded_file[len(unpadded_file)/2:len(unpadded_file)]
            origin = unpadded_file[0, 3:6]
            [row, col] = unpadded_file.shape
            origin = np.tile(origin, (row, 50))
            unpadded_file = unpadded_file - origin
            extra_frames = (len(unpadded_file) % self.step)
            if extra_frames < (self.step/2):
               padded_file = unpadded_file[0:len(unpadded_file) - extra_frames,:]
            else:
               [row, col] = unpadded_file.shape
               alpha = (len(unpadded_file)/self.step) + 1
               req_pad = np.zeros(((alpha * self.step)-row, col))
               padded_file = np.vstack((unpadded_file, req_pad))
            splitted_file = np.split(padded_file, self.step)
            splitted_file = np.asarray(splitted_file)
            row, col, width = splitted_file.shape
            sampled_file = []
            for k in range(0,self.step):
                c = np.random.choice(col,1)
                sampled_file.append(splitted_file[k,c,:])
            sampled_file = np.asarray(sampled_file)
            X_skl_22[i,] = np.squeeze(sampled_file)
            #X_skl_21 = np.asarray(X_skl_21)

	for i, ID in enumerate(list_IDs_temp):
            # Store sample
            unpadded_file = np.load(self.path_skeleton + ID + '.npy')
            unpadded_file = unpadded_file[len(unpadded_file)/2:len(unpadded_file)]
            origin = unpadded_file[0, 3:6]
            [row, col] = unpadded_file.shape
            origin = np.tile(origin, (row, 50))
            unpadded_file = unpadded_file - origin
            extra_frames = (len(unpadded_file) % self.step)
            if extra_frames < (self.step/2):
               padded_file = unpadded_file[0:len(unpadded_file) - extra_frames,:]
            else:
               [row, col] = unpadded_file.shape
               alpha = (len(unpadded_file)/self.step) + 1
               req_pad = np.zeros(((alpha * self.step)-row, col))
               padded_file = np.vstack((unpadded_file, req_pad))
            splitted_file = np.split(padded_file, self.step)
            splitted_file = np.asarray(splitted_file)
            row, col, width = splitted_file.shape
            sampled_file = []
            for k in range(0,self.step):
                c = np.random.choice(col,1)
                sampled_file.append(splitted_file[k,c,:])
            sampled_file = np.asarray(sampled_file)
            X_skl_22[i,] = np.squeeze(sampled_file)
            #X_skl_21 = np.asarray(X_skl_21)

	for i, ID in enumerate(list_IDs_temp):
            # Store sample
            unpadded_file = np.load(self.path_skeleton + ID + '.npy')
            unpadded_file = unpadded_file[0:len(unpadded_file)/3]
            origin = unpadded_file[0, 3:6]
            [row, col] = unpadded_file.shape
            origin = np.tile(origin, (row, 50))
            unpadded_file = unpadded_file - origin
            extra_frames = (len(unpadded_file) % self.step)
            if extra_frames < (self.step/2):
               padded_file = unpadded_file[0:len(unpadded_file) - extra_frames,:]
            else:
               [row, col] = unpadded_file.shape
               alpha = (len(unpadded_file)/self.step) + 1
               req_pad = np.zeros(((alpha * self.step)-row, col))
               padded_file = np.vstack((unpadded_file, req_pad))
            splitted_file = np.split(padded_file, self.step)
            splitted_file = np.asarray(splitted_file)
            row, col, width = splitted_file.shape
            sampled_file = []
            for k in range(0,self.step):
                c = np.random.choice(col,1)
                sampled_file.append(splitted_file[k,c,:])
            sampled_file = np.asarray(sampled_file)
            X_skl_31[i,] = np.squeeze(sampled_file)
            #X_skl_21 = np.asarray(X_skl_21)

	for i, ID in enumerate(list_IDs_temp):
            # Store sample
            unpadded_file = np.load(self.path_skeleton + ID + '.npy')
            unpadded_file = unpadded_file[len(unpadded_file)/3:2*len(unpadded_file)/3]
            origin = unpadded_file[0, 3:6]
            [row, col] = unpadded_file.shape
            origin = np.tile(origin, (row, 50))
            unpadded_file = unpadded_file - origin
            extra_frames = (len(unpadded_file) % self.step)
            if extra_frames < (self.step/2):
               padded_file = unpadded_file[0:len(unpadded_file) - extra_frames,:]
            else:
               [row, col] = unpadded_file.shape
               alpha = (len(unpadded_file)/self.step) + 1
               req_pad = np.zeros(((alpha * self.step)-row, col))
               padded_file = np.vstack((unpadded_file, req_pad))
            splitted_file = np.split(padded_file, self.step)
            splitted_file = np.asarray(splitted_file)
            row, col, width = splitted_file.shape
            sampled_file = []
            for k in range(0,self.step):
                c = np.random.choice(col,1)
                sampled_file.append(splitted_file[k,c,:])
            sampled_file = np.asarray(sampled_file)
            X_skl_32[i,] = np.squeeze(sampled_file)
            #X_skl_21 = np.asarray(X_skl_21)

	for i, ID in enumerate(list_IDs_temp):
            # Store sample
            unpadded_file = np.load(self.path_skeleton + ID + '.npy')
            unpadded_file = unpadded_file[2*len(unpadded_file)/3:len(unpadded_file)]
            origin = unpadded_file[0, 3:6]
            [row, col] = unpadded_file.shape
            origin = np.tile(origin, (row, 50))
            unpadded_file = unpadded_file - origin
            extra_frames = (len(unpadded_file) % self.step)
            if extra_frames < (self.step/2):
               padded_file = unpadded_file[0:len(unpadded_file) - extra_frames,:]
            else:
               [row, col] = unpadded_file.shape
               alpha = (len(unpadded_file)/self.step) + 1
               req_pad = np.zeros(((alpha * self.step)-row, col))
               padded_file = np.vstack((unpadded_file, req_pad))
            splitted_file = np.split(padded_file, self.step)
            splitted_file = np.asarray(splitted_file)
            row, col, width = splitted_file.shape
            sampled_file = []
            for k in range(0,self.step):
                c = np.random.choice(col,1)
                sampled_file.append(splitted_file[k,c,:])
            sampled_file = np.asarray(sampled_file)
            X_skl_33[i,] = np.squeeze(sampled_file)
            #X_skl_21 = np.asarray(X_skl_21)

	for i, ID in enumerate(list_IDs_temp):
            # Store sample
            unpadded_file = np.load(self.path_skeleton + ID + '.npy')
            unpadded_file = unpadded_file[0:len(unpadded_file)/4]
            origin = unpadded_file[0, 3:6]
            [row, col] = unpadded_file.shape
            origin = np.tile(origin, (row, 50))
            unpadded_file = unpadded_file - origin
            extra_frames = (len(unpadded_file) % self.step)
            if extra_frames < (self.step/2):
               padded_file = unpadded_file[0:len(unpadded_file) - extra_frames,:]
            else:
               [row, col] = unpadded_file.shape
               alpha = (len(unpadded_file)/self.step) + 1
               req_pad = np.zeros(((alpha * self.step)-row, col))
               padded_file = np.vstack((unpadded_file, req_pad))
            splitted_file = np.split(padded_file, self.step)
            splitted_file = np.asarray(splitted_file)
            row, col, width = splitted_file.shape
            sampled_file = []
            for k in range(0,self.step):
                c = np.random.choice(col,1)
                sampled_file.append(splitted_file[k,c,:])
            sampled_file = np.asarray(sampled_file)
            X_skl_41[i,] = np.squeeze(sampled_file)
            #X_skl_21 = np.asarray(X_skl_21)

	for i, ID in enumerate(list_IDs_temp):
            # Store sample
            unpadded_file = np.load(self.path_skeleton + ID + '.npy')
            unpadded_file = unpadded_file[len(unpadded_file)/4:len(unpadded_file)/2]
            origin = unpadded_file[0, 3:6]
            [row, col] = unpadded_file.shape
            origin = np.tile(origin, (row, 50))
            unpadded_file = unpadded_file - origin
            extra_frames = (len(unpadded_file) % self.step)
            if extra_frames < (self.step/2):
               padded_file = unpadded_file[0:len(unpadded_file) - extra_frames,:]
            else:
               [row, col] = unpadded_file.shape
               alpha = (len(unpadded_file)/self.step) + 1
               req_pad = np.zeros(((alpha * self.step)-row, col))
               padded_file = np.vstack((unpadded_file, req_pad))
            splitted_file = np.split(padded_file, self.step)
            splitted_file = np.asarray(splitted_file)
            row, col, width = splitted_file.shape
            sampled_file = []
            for k in range(0,self.step):
                c = np.random.choice(col,1)
                sampled_file.append(splitted_file[k,c,:])
            sampled_file = np.asarray(sampled_file)
            X_skl_42[i,] = np.squeeze(sampled_file)
            #X_skl_21 = np.asarray(X_skl_21)

	for i, ID in enumerate(list_IDs_temp):
            # Store sample
            unpadded_file = np.load(self.path_skeleton + ID + '.npy')
            unpadded_file = unpadded_file[len(unpadded_file)/2:3*len(unpadded_file)/4]
            origin = unpadded_file[0, 3:6]
            [row, col] = unpadded_file.shape
            origin = np.tile(origin, (row, 50))
            unpadded_file = unpadded_file - origin
            extra_frames = (len(unpadded_file) % self.step)
            if extra_frames < (self.step/2):
               padded_file = unpadded_file[0:len(unpadded_file) - extra_frames,:]
            else:
               [row, col] = unpadded_file.shape
               alpha = (len(unpadded_file)/self.step) + 1
               req_pad = np.zeros(((alpha * self.step)-row, col))
               padded_file = np.vstack((unpadded_file, req_pad))
            splitted_file = np.split(padded_file, self.step)
            splitted_file = np.asarray(splitted_file)
            row, col, width = splitted_file.shape
            sampled_file = []
            for k in range(0,self.step):
                c = np.random.choice(col,1)
                sampled_file.append(splitted_file[k,c,:])
            sampled_file = np.asarray(sampled_file)
            X_skl_43[i,] = np.squeeze(sampled_file)
            #X_skl_21 = np.asarray(X_skl_21)

	for i, ID in enumerate(list_IDs_temp):
            # Store sample
            unpadded_file = np.load(self.path_skeleton + ID + '.npy')
            unpadded_file = unpadded_file[3*len(unpadded_file)/4:len(unpadded_file)]
            origin = unpadded_file[0, 3:6]
            [row, col] = unpadded_file.shape
            origin = np.tile(origin, (row, 50))
            unpadded_file = unpadded_file - origin
            extra_frames = (len(unpadded_file) % self.step)
            if extra_frames < (self.step/2):
               padded_file = unpadded_file[0:len(unpadded_file) - extra_frames,:]
            else:
               [row, col] = unpadded_file.shape
               alpha = (len(unpadded_file)/self.step) + 1
               req_pad = np.zeros(((alpha * self.step)-row, col))
               padded_file = np.vstack((unpadded_file, req_pad))
            splitted_file = np.split(padded_file, self.step)
            splitted_file = np.asarray(splitted_file)
            row, col, width = splitted_file.shape
            sampled_file = []
            for k in range(0,self.step):
                c = np.random.choice(col,1)
                sampled_file.append(splitted_file[k,c,:])
            sampled_file = np.asarray(sampled_file)
            X_skl_44[i,] = np.squeeze(sampled_file)
            #X_skl_21 = np.asarray(X_skl_21)

        #y[i] = self.labels[ID]
        y = np.array([int(i[-3:]) for i in list_IDs_temp]) - 1      
        return X1, X2, X3, X_skl_21, X_skl_22, X_skl_31, X_skl_32, X_skl_33, X_skl_41, X_skl_42, X_skl_43, X_skl_44, keras.utils.to_categorical(y, num_classes=self.n_classes)


