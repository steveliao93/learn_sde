
import time
import random
import iisignature
import numpy as np
from keras import layers
from keras.layers import Conv1D, Input, Lambda, Reshape, Permute
from keras.models import Model
from tensorflow import keras  
from keras.optimizers import Adam
from keras.engine import InputSpec
from keras.models import Sequential
from keras.layers import LSTM, Dense, BatchNormalization, Dropout
import tensorflow as tf
from functools import partial
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from cus_layers import *

cha_train = np.load('cha_train.npy')
cha_val = np.load('cha_val.npy')
train_label = np.load('cha_train_label.npy')
val_label = np.load('cha_val_label.npy')

frame_nb = cha_train.shape[1]
skl_dim = cha_train.shape[2]
T_vec = np.linspace(0,1, frame_nb)
T_vec = T_vec.reshape(frame_nb,1)
C_T = partial(Cat_T, T_vec = T_vec)

# Model construct
def build_lin_Logsig_rnn_model(input_shape, n_hidden_neurons, output_shape, no_of_segments, deg_of_logsig, learning_rate, drop_rate1, drop_rate2, filter_size):
    logsiglen = iisignature.logsiglength(filter_size,deg_of_logsig)
    input_layer = Input(shape= input_shape)
    # Time path concatenation
    cat_layer = Lambda(lambda x:C_T(x), output_shape=(input_shape[0], input_shape[1]+1))(input_layer)
    # Convolutional layer
    lin_projection_layer = Conv1D(filter_size,1)(cat_layer)
    # Dropout
    drop_layer_1 = Dropout(drop_rate1)(lin_projection_layer)
    # Cumulative sum
    ps_layer = Lambda(lambda x:PS(x), output_shape=(input_shape[0],filter_size))(drop_layer_1)
#     BN_layer_0 = BatchNormalization()(lin_projection_layer)
    # Logsig layer
    hidden_layer_1 = Lambda(lambda x:CLF(x, no_of_segments, deg_of_logsig), \
                          output_shape=(no_of_segments,logsiglen))(ps_layer)
    hidden_layer_2 = Reshape((no_of_segments,logsiglen))(hidden_layer_1)
    # Batchnormalization
    BN_layer_1 = BatchNormalization()(hidden_layer_2)
    # LSTM
    lstm_layer = LSTM(units=n_hidden_neurons)(BN_layer_1)
#     BN_layer_2 = BatchNormalization()(lstm_layer)
    # Dropout
    drop_layer_2 = Dropout(drop_rate2)(lstm_layer)
    output_layer = Dense(output_shape, activation='softmax')(drop_layer_2)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.summary()
    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics = ['accuracy'])

    return model

n_hidden_neurons = 128
batch_size = 24 
learning_rate = 0.001
epochs = 500
number_of_segment = 4
deg_of_logsig = 2
drop_rate1 = 0.3
drop_rate2 = 0.5
filter_size = 30

output_shape = train_label.shape[1]
input_shape = [frame_nb, skl_dim]


start = time.time()
model = build_lin_Logsig_rnn_model(input_shape, n_hidden_neurons, output_shape, number_of_segment, deg_of_logsig, learning_rate, drop_rate, filter_size)

model_name = 'gesture_model_dr%d_fs%d_dg%d.hdf5' %(drop_rate, filter_size, deg_of_logsig)

reduce_lr = ReduceLROnPlateau(monitor='loss', patience=10, verbose=1, factor=0.8, min_lr=0.000001)
mcp_save = ModelCheckpoint(model_name, save_best_only=True, monitor='val_acc', mode='max')

hist = model.fit(cha_train, train_label, epochs=epochs, batch_size=batch_size,shuffle=True, verbose=1,validation_data=(cha_val, val_label),
          callbacks = [ reduce_lr, mcp_save])  

print(max(hist.history['val_acc']))
print((time.time()-start)/3600)

