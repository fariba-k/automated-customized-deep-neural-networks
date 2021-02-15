import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Reshape, UpSampling1D, Dropout
import numpy as np
from pandas import DataFrame
import math
from collections import Counter
from sklearn.metrics import confusion_matrix
tf.random.set_seed(42)



'''
this is a class of simple deep neural network containing only fully-connected dense layers.
'''
class DNN():
    def __init__(self, layer_size):
        self.layer_size = layer_size    # list of integers containing number of nodes in each hidden layer, the last number is the number of nodes in the output layer
        self.create_network()

    def create_network(self):
        input_data = Input(shape=self.layer_size[0])
        if len(self.layer_size)==2:
            output = Dense(self.layer_size[1], activation='relu')(input_data)
        elif len(self.layer_size)>2:
            x = Dense(self.layer_size[1], activation='relu')(input_data)
            for i  in range(2,len(self.layer_size)-1):
                x = Dense(self.layer_size[i], activation='relu')(x)
            output = Dense(self.layer_size[-1], activation='sigmoid')(x)
        else:
            print('Error: Layers sizes are unacceptable!')
        self.dnn_model = keras.models.Model(input_data, output, name='deep_neural_network')
        self.dnn_model.compile(optimizer='adam', loss='categorical_crossentropy')

    def print_summary(self):
        self.dnn_model.summary()


class AE():
    def __init__(self, encoder_layer_size, dropout=0.0001):
        self.layer_size = encoder_layer_size
        self.dropout = dropout
        self.create_network()

    def create_network(self):
        input_data = Input(shape=self.layer_size[0])
        x = Dropout(self.dropout)(input_data)
        for i  in range(1,len(self.layer_size)-1):
            x = Dense(self.layer_size[i], activation='relu')(x)
        encoded = Dense(self.layer_size[-1], activation='relu')(x)
        x = Dropout(self.dropout)(encoded)
        for i  in range(len(self.layer_size)-2,0,-1):
            x = Dense(self.layer_size[i], activation='relu')(x)
        decoded = Dense(self.layer_size[0], activation='relu')(x)
        self.ae_model = keras.models.Model(input_data, decoded, name='autoencoder')
        self.e_model = keras.models.Model(input_data, encoded, name='encoder')
        self.ae_model.compile(optimizer='adam', loss='mse')

    def print_summary(self):
        self.ae_model.summary()
        self.e_model.summary()


class HNN():
    def __init__(self, encoder_layer_size, num_classes, dropout=0.0001):
        self.layer_size = encoder_layer_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.create_network()

    def create_network(self):
        input_data = Input(shape=self.layer_size[0])
        x = Dropout(self.dropout)(input_data)
        for i  in range(1,len(self.layer_size)-1):
            x = Dense(self.layer_size[i], activation='relu')(x)
        encoded = Dense(self.layer_size[-1], activation='relu')(x)
        x = Dropout(self.dropout)(encoded)
        for i  in range(len(self.layer_size)-2,0,-1):
            x = Dense(self.layer_size[i], activation='relu')(x)
        decoded = Dense(self.layer_size[0], activation='relu')(x)
        x1 = Flatten()(encoded)
        class_output = Dense(self.num_classes, activation='sigmoid')(x1)
        self.hnn_model = keras.models.Model(input_data, [class_output,decoded], name='hybrid_neural_network')
        self.e_model = keras.models.Model(input_data, encoded, name='encoder')
        self.hnn_model.compile(optimizer='adam', loss=['categorical_crossentropy','mse'])

    def print_summary(self):
        self.hnn_model.summary()
        self.e_model.summary()


class CNN():
    def __init__(self, input_dim, layers='cmcmd' , layers_param=[64,2,16,2,32,10], kernel_size=10, dropout=0.0001):
        self.input_dim = input_dim
        self.layers = layers
        self.layers_param = layers_param
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.create_network()

    def create_network(self):
        current_size = self.input_dim[0]
        input_data = Input(shape=self.input_dim)
        x = Dropout(self.dropout)(input_data)
        for i  in range(len(self.layers)):
            l = self.layers[i]
            param = self.layers_param[i]
            if l=='c':
                x = Conv1D(param, self.kernel_size, activation='tanh', padding='same')(x)
            elif l=='m':
                x = MaxPooling1D(param, padding='same')(x)
                current_size = math.ceil(current_size/param)
            elif l=='d':
                x = Flatten()(x)
                x = Dense(param, activation='tanh')(x)
        output = Dense(self.layers_param[-1], activation='sigmoid')(x)
        self.cnn_model = keras.models.Model(input_data, output, name='cnn')
        self.cnn_model.compile(optimizer='adam', loss=['categorical_crossentropy'])

    def print_summary(self):
        self.cnn_model.summary()


class CAE():
    def __init__(self, input_dim, num_outputs=2, layers='cmcmdcucu' , layers_param=[64,2,16,2,32,16,2,64], num_classes=0, kernel_size=10, dropout=0.0001):
        self.input_dim = input_dim
        self.num_outputs = num_outputs
        self.layers = layers
        self.layers_param = layers_param
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.create_network()

    def create_network(self):
        current_size = self.input_dim[0]
        input_data = Input(shape=self.input_dim)
        x = Dropout(self.dropout)(input_data)
        for i  in range(len(self.layers)):
            l = self.layers[i]
            param = self.layers_param[i]
            if l=='c':
                x = Conv1D(param, self.kernel_size, activation='tanh', padding='same')(x)
            elif l=='m':
                x = MaxPooling1D(param, padding='same')(x)
                current_size = math.ceil(current_size/param)
            elif l=='d':
                x = Flatten()(x)
                encoded = Dense(param, activation='tanh')(x)
                x = Dense(int(current_size*self.layers_param[i-2]), activation='linear')(encoded)
                x = Reshape((current_size, self.layers_param[i-2]))(x)
            elif l=='u':
                x = UpSampling1D(param)(x)
                current_size = int(current_size*param)
        decoded = Conv1D(self.input_dim[1], int(current_size-self.input_dim[0])+1, activation='tanh')(x)
        if self.num_outputs==1:
            self.cae_model = keras.models.Model(input_data, decoded, name='convolutional_autoencoder')
            self.ce_model = keras.models.Model(input_data, encoded, name='convolutional_encoder')
            self.cae_model.compile(optimizer='adam', loss=['categorical_crossentropy','mse'])
        elif self.num_outputs==2:
            x1 = Flatten()(encoded)
            class_output = Dense(self.num_classes, activation='sigmoid')(x1)
            self.cae_model = keras.models.Model(input_data, [class_output,decoded], name='hybrid_convolutional_autoencoder')
            self.ce_model = keras.models.Model(input_data, encoded, name='convolutional_encoder')
            self.cae_model.compile(optimizer='adam', loss=['categorical_crossentropy','mse'])

    def print_summary(self):
        self.cae_model.summary()
        self.ce_model.summary()
