from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, GRU, Bidirectional
from keras.layers import TimeDistributed, GaussianNoise, GaussianDropout, Dropout
from keras.models import Model
from keras import backend as K
import keras
from keras.layers import Activation, concatenate, Flatten, Reshape, Merge, Input, Add, RepeatVector, Permute
from keras import regularizers


def inflate_dense_temporal(x):
    a = RepeatVector(512)(x)
    a = Permute((2,1), input_shape=(512,2))(a)
    return a

def l1_reg(weight_mat):
    return 1*K.sum(K.square(1-weight_mat))


def inflate_dense_seg_2(x):
    a = RepeatVector(512)(x)
    a = Permute((2,1), input_shape=(512,2))(a)
    return a

def inflate_dense_seg_3(x):
    a = RepeatVector(512)(x)
    a = Permute((2,1), input_shape=(512,3))(a)
    return a

def inflate_dense_seg_4(x):
    a = RepeatVector(512)(x)
    a = Permute((2,1), input_shape=(512,4))(a)
    return a

def inflate_features(x):
    a = RepeatVector(1)(x)
    return a

def slice_1(x):
    return x[:,0,:]

def slice_2(x):
    return x[:,1,:]

def slice_3(x):
    return x[:,2,:]

def slice_4(x):
    return x[:,3,:]

def l1_reg(weight_mat):
    return 0.1*K.square((1-K.sum(weight_mat)))
    #return 0.1*K.sum(K.square(weight_mat))


def temporal_model(n_neuron, n_neuron_skl, n_dropout, batch_size, timesteps_seg1, timesteps_seg2, timesteps_seg3, timesteps_skl, data_dim, data_dim_skl, num_classes):
    print('Build segment fusion model!!!')
    x1 = Input(shape=(timesteps_seg1, data_dim), name='segment_1')
    main_lstm_1 = GRU(n_neuron, return_sequences=True)(x1)
    main_lstm_dropped_1 = Dropout(n_dropout)(main_lstm_1)
    #fc_tanh_1 = Dense(256, activation='tanh')(main_lstm_dropped_1)

    x2 = Input(shape=(timesteps_seg2, data_dim), name='segment_2')
    main_lstm_2 = GRU(n_neuron, return_sequences=True)(x2)
    main_lstm_dropped_2 = Dropout(n_dropout)(main_lstm_2)

    x3 = Input(shape=(timesteps_seg3, data_dim), name='segment_3')
    main_lstm_3 = GRU(n_neuron, return_sequences=True)(x3)
    main_lstm_dropped_3 = Dropout(n_dropout)(main_lstm_3)

    x4 = Input(shape=(timesteps_skl, data_dim_skl), name='skl_segment_21')
    main_lstm_4 = GRU(n_neuron_skl)(x4)
    main_lstm_dropped_4 = Dropout(n_dropout)(main_lstm_4)
    #fc_tanh_4 = Dense(256, activation='tanh')(main_lstm_dropped_4)

    x5 = Input(shape=(timesteps_skl, data_dim_skl), name='skl_segment_22')
    main_lstm_5 = GRU(n_neuron_skl)(x5)
    main_lstm_dropped_5 = Dropout(n_dropout)(main_lstm_5)
    #fc_tanh_5 = Dense(256, activation='tanh')(main_lstm_dropped_5)

    x6 = Input(shape=(timesteps_skl, data_dim_skl), name='skl_segment_31')
    main_lstm_6 = GRU(n_neuron_skl)(x6)
    main_lstm_dropped_6 = Dropout(n_dropout)(main_lstm_6)

    x7 = Input(shape=(timesteps_skl, data_dim_skl), name='skl_segment_32')
    main_lstm_7 = GRU(n_neuron_skl)(x7)
    main_lstm_dropped_7 = Dropout(n_dropout)(main_lstm_7)
   
    x8 = Input(shape=(timesteps_skl, data_dim_skl), name='skl_segment_33')
    main_lstm_8 = GRU(n_neuron_skl)(x8)
    main_lstm_dropped_8 = Dropout(n_dropout)(main_lstm_8)

    x9 = Input(shape=(timesteps_skl, data_dim_skl), name='skl_segment_41')
    main_lstm_9 = GRU(n_neuron_skl)(x9)
    main_lstm_dropped_9 = Dropout(n_dropout)(main_lstm_9)

    x10 = Input(shape=(timesteps_skl, data_dim_skl), name='skl_segment_42')
    main_lstm_10 = GRU(n_neuron_skl)(x10)
    main_lstm_dropped_10 = Dropout(n_dropout)(main_lstm_10)

    x11 = Input(shape=(timesteps_skl, data_dim_skl), name='skl_segment_43')
    main_lstm_11 = GRU(n_neuron_skl)(x11)
    main_lstm_dropped_11 = Dropout(n_dropout)(main_lstm_11)

    x12 = Input(shape=(timesteps_skl, data_dim_skl), name='skl_segment_44')
    main_lstm_12 = GRU(n_neuron_skl)(x12)
    main_lstm_dropped_12 = Dropout(n_dropout)(main_lstm_12)

    inflated_seg_21 = keras.layers.core.Lambda(inflate_features, output_shape=(1,n_neuron_skl))(main_lstm_dropped_4)
    inflated_seg_22 = keras.layers.core.Lambda(inflate_features, output_shape=(1,n_neuron_skl))(main_lstm_dropped_5)
    inflated_seg_31 = keras.layers.core.Lambda(inflate_features, output_shape=(1,n_neuron_skl))(main_lstm_dropped_6)
    inflated_seg_32 = keras.layers.core.Lambda(inflate_features, output_shape=(1,n_neuron_skl))(main_lstm_dropped_7)
    inflated_seg_33 = keras.layers.core.Lambda(inflate_features, output_shape=(1,n_neuron_skl))(main_lstm_dropped_8)
    inflated_seg_41 = keras.layers.core.Lambda(inflate_features, output_shape=(1,n_neuron_skl))(main_lstm_dropped_9)
    inflated_seg_42 = keras.layers.core.Lambda(inflate_features, output_shape=(1,n_neuron_skl))(main_lstm_dropped_10)
    inflated_seg_43 = keras.layers.core.Lambda(inflate_features, output_shape=(1,n_neuron_skl))(main_lstm_dropped_11)
    inflated_seg_44 = keras.layers.core.Lambda(inflate_features, output_shape=(1,n_neuron_skl))(main_lstm_dropped_12)

    multiplied_features_1 = keras.layers.concatenate([inflated_seg_21, inflated_seg_22], axis=1)
    mid_lstm_1_undropped = GRU(n_neuron_skl)(multiplied_features_1)
    mid_lstm_1 = Dropout(n_dropout)(mid_lstm_1_undropped)
    fc_T_attention_1 = Dense(timesteps_seg1, activation='softmax', name='mid_att_1', activity_regularizer=None, kernel_initializer='zeros', bias_initializer='zeros')(mid_lstm_1)
    inflated_seg_2_att = keras.layers.core.Lambda(inflate_dense_seg_2, output_shape=(timesteps_seg1,n_neuron))(fc_T_attention_1)
    temp_att_seg_2 = keras.layers.multiply([inflated_seg_2_att, main_lstm_dropped_1])
    seg_2_slice_1 = keras.layers.core.Lambda(slice_1)(temp_att_seg_2)
    seg_2_slice_2 = keras.layers.core.Lambda(slice_2)(temp_att_seg_2)
    seg_2_out = keras.layers.add([seg_2_slice_1,seg_2_slice_2])

    multiplied_features_2 = keras.layers.concatenate([inflated_seg_31, inflated_seg_32, inflated_seg_33], axis=1)
    mid_lstm_2_undropped = GRU(n_neuron_skl)(multiplied_features_2)
    mid_lstm_2 = Dropout(n_dropout)(mid_lstm_2_undropped)
    fc_T_attention_2 = Dense(timesteps_seg2, activation='softmax', name='mid_att_2', activity_regularizer=None, kernel_initializer='zeros', bias_initializer='zeros')(mid_lstm_2)
    inflated_seg_3_att = keras.layers.core.Lambda(inflate_dense_seg_3, output_shape=(timesteps_seg2,n_neuron))(fc_T_attention_2)
    temp_att_seg_3 = keras.layers.multiply([inflated_seg_3_att, main_lstm_dropped_2])
    seg_3_slice_1 = keras.layers.core.Lambda(slice_1)(temp_att_seg_3)
    seg_3_slice_2 = keras.layers.core.Lambda(slice_2)(temp_att_seg_3)
    seg_3_slice_3 = keras.layers.core.Lambda(slice_3)(temp_att_seg_3)
    seg_3_out = keras.layers.add([seg_3_slice_1,seg_3_slice_2,seg_3_slice_3])

    multiplied_features_3 = keras.layers.concatenate([inflated_seg_41, inflated_seg_42, inflated_seg_43, inflated_seg_44], axis=1)
    mid_lstm_3_undropped = GRU(n_neuron_skl)(multiplied_features_3)
    mid_lstm_3 = Dropout(n_dropout)(mid_lstm_3_undropped)
    fc_T_attention_3 = Dense(timesteps_seg3, activation='softmax', name='mid_att_3', activity_regularizer=None, kernel_initializer='zeros', bias_initializer='zeros')(mid_lstm_3)
    inflated_seg_4_att = keras.layers.core.Lambda(inflate_dense_seg_4, output_shape=(timesteps_seg3,n_neuron))(fc_T_attention_3)
    temp_att_seg_4 = keras.layers.multiply([inflated_seg_4_att, main_lstm_dropped_3])
    seg_4_slice_1 = keras.layers.core.Lambda(slice_1)(temp_att_seg_4)
    seg_4_slice_2 = keras.layers.core.Lambda(slice_2)(temp_att_seg_4)
    seg_4_slice_3 = keras.layers.core.Lambda(slice_3)(temp_att_seg_4)
    seg_4_slice_4 = keras.layers.core.Lambda(slice_4)(temp_att_seg_4)
    seg_4_out = keras.layers.add([seg_4_slice_1,seg_4_slice_2,seg_4_slice_3,seg_4_slice_4])

    multiplied_features_4 = keras.layers.concatenate([mid_lstm_1,mid_lstm_2,mid_lstm_3], axis=-1)
    fc_TS_attention = Dense(3, activation='softmax', name='end_att', activity_regularizer=None, kernel_initializer='zeros', bias_initializer='zeros')(multiplied_features_4)
    inflate_fc_TS_attention = keras.layers.core.Lambda(inflate_dense_seg_3, output_shape=(timesteps_seg2,n_neuron))(fc_TS_attention)
    TS_att_slice1 = keras.layers.core.Lambda(slice_1)(inflate_fc_TS_attention)
    TS_att_slice2 = keras.layers.core.Lambda(slice_2)(inflate_fc_TS_attention)
    TS_att_slice3 = keras.layers.core.Lambda(slice_3)(inflate_fc_TS_attention)
    temp_att_TS_seg1 = keras.layers.multiply([TS_att_slice1, seg_2_out])
    temp_att_TS_seg2 = keras.layers.multiply([TS_att_slice2, seg_3_out])
    temp_att_TS_seg3 = keras.layers.multiply([TS_att_slice3, seg_4_out])

    fc_final = keras.layers.concatenate([temp_att_TS_seg1, temp_att_TS_seg2, temp_att_TS_seg3], axis=-1)
    fc_classify = Dense(num_classes, activation='softmax', name='dense_final')(fc_final)
    model = Model(inputs=[x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12], outputs=fc_classify)
    return model

