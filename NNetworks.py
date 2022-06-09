#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 11:23:33 2021

@author: cbunn
"""

"""
Contains the various Neural networks used in this implementation
Discrimators and Generators and loss functions
"""

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

#create a Layer subclass for the Linear learning layer, transposes an input vector into a 128 dimension vector

class LinearLayer(keras.layers.Layer):
    
    def __init__(self, units, inputShape, name, bias=True, trainable= True, **kwargs):
        super(LinearLayer, self).__init__(trainable=trainable, name=name, **kwargs)
        self.units = units
        self.inputShape = inputShape
        self.bias = bias
        self.trainable = trainable
        self.wInitialiser = tf.initializers.truncated_normal(stddev=0.001)
        self.bInitialiser = tf.constant_initializer(0.0)

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer=self.wInitialiser, trainable=self.trainable)
        if self.bias:
            self.b = self.add_weight(shape=(self.units), initializer=self.bInitialiser, trainable=self.trainable)
        else:
            self.b = 0.0
        
    def call(self, inputs):
        product = tf.tensordot(inputs, self.w, axes=[[2], [0]])
        if self.bias:
            product += self.b
        
        return product
    
#Linear Layer for 2d input
class LinearLayer2D(keras.layers.Layer):
    
    def __init__(self, units, inputShape, name, bias=True, trainable= True, **kwargs):
        super(LinearLayer2D, self).__init__(trainable=trainable, name=name, **kwargs)
        self.units = units
        self.inputShape = inputShape
        self.bias = bias
        self.trainable = trainable
        self.wInitialiser = tf.initializers.truncated_normal(stddev=0.001)
        self.bInitialiser = tf.constant_initializer(0.0)

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer=self.wInitialiser, trainable=self.trainable)
        if self.bias:
            self.b = self.add_weight(shape=(self.units), initializer=self.bInitialiser, trainable=self.trainable)
        else:
            self.b = 0.0
        
    def call(self, inputs):
        product = tf.tensordot(inputs, self.w, axes=[[1], [0]])
        if self.bias:
            product += self.b
        
        return product

#add the noise as the Z dimension in the pose.
class addZLayer(keras.layers.Layer):
    
    def __init__(self, inputShape, name, bias=False, trainable= False, **kwargs):
        super(addZLayer, self).__init__(trainable=trainable, name=name, **kwargs)
        self.inputShape = inputShape
        self.bias = bias
        self.trainable = trainable
#        self.wInitialiser = tf.initializers.truncated_normal(stddev=0.001)
#        self.bInitialiser = tf.constant_initializer(0.0)

    def build(self, input_shape):
        return
 #       self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer=self.wInitialiser, trainable=self.trainable)
 #       if self.bias:
 #           self.b = self.add_weight(shape=(self.units), initializer=self.bInitialiser, trainable=self.trainable)
 #       else:
 #           self.b = 0.0
        
    def call(self, inputs):
#        print(inputs)
#        print(inputs[0].shape)
#        print(inputs[1].shape)
        d1, d2, d3 = inputs[0].shape
        dd1, dd2, dd3, dd4 = inputs[1].shape
        inData = tf.reshape(inputs[0], [-1, d2, dd3, 2])
#        print(inData.shape)
#        print(inputs[1].shape)

        inData = tf.concat([inData, inputs[1]], axis=3)
#        print(inData)

        inData = tf.reshape(inData, [-1, d2, d3+dd3])
#        print(inData)
        return inData
        
#layer to introduce the noise into the generator states
class NoiseLayer(keras.layers.Layer):
    
    def __init__(self, units, inputShape, name, trainable=False, **kwargs):
        super(NoiseLayer, self).__init__(trainable=trainable, name=name, **kwargs)
        self.units = units
        self.inputShape = inputShape
        self.trainable = trainable
        self.wInitialiser = tf.initializers.truncated_normal(stddev=0.001)
        
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer=self.wInitialiser, trainable=self.trainable)
        
    def call(self, inputs):
        product = tf.matmul(inputs, self.w)
        return product
    
#layer to introduce the noise into the Z dimension of each pose 
class NoiseLayerZ(keras.layers.Layer):
    
    def __init__(self, zShape, zDimension, inputShape, name, trainable=False, **kwargs):
        super(NoiseLayerZ, self).__init__(trainable=trainable, name=name, **kwargs)
        self.zShape = zShape
        self.zDimension = zDimension
        self.inputShape = inputShape
        self.trainable = trainable
        self.wInitialiser = tf.initializers.truncated_normal(stddev=0.001)
        
    def build(self, input_shape):
        
        d1, d2, d3 = self.zDimension        
        self.w = self.add_weight(shape=(input_shape[-1], d1*d2), initializer=self.wInitialiser, trainable=self.trainable)
        
    def call(self, inputs):
#        print(inputs.shape)
#        print(self.w)
#        print("hello")
        d1, d2, d3 = self.zDimension
        product = tf.matmul(inputs, self.w)
        product = tf.reshape(product, [-1, d1, d2, d3])
#        print(product.shape)
                
        return product

#layer to introduce output of the encoder into the decoder
class CenterLayer(keras.layers.Layer):
    
    def __init__(self, units, name, trainable=False, **kwargs):
        super(CenterLayer, self).__init__(trainable=trainable, name=name, **kwargs)
        self.units = units
        self.trainable = trainable
        self.wInitialiser = tf.initializers.truncated_normal(stddev=0.001)
        
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer=self.wInitialiser, trainable=self.trainable)
        
    def call(self, inputs):
        product = tf.matmul(inputs, self.w)
        return product

# the layer feeds the critic and discriminator
class CDFeedLayer(keras.layers.Layer):
    def __init__(self, units, name=None, trainable=True, **kwargs):
        super(CDFeedLayer, self).__init__(trainable=trainable, name=name, **kwargs)
        self.units = units
#        self.inputShape = input_shape
        self.trainable = trainable
        
    def build(self, inputs):
        self.w = self.add_weight(shape=(inputs[-1], self.units), initializer=tf.initializers.truncated_normal(), trainable=self.trainable)
        print()
        
    def call(self, inputs):
        product = tf.tensordot(inputs, self.w, axes=[[2],[0]])    
        
        return product
    
    
# NN for both the critic and discriminator 

def NNDiscriminator(inputShape, activation = 'relu', seed=2050, num_neurons=512, depth=128):
#    tf.random_normal_initializer(seed)
#    inlayer = keras.layers.Flatten(input_shape=inputShape)
    layer0  = keras.layers.Dense(depth, input_shape=inputShape)
#    layer00 = keras.layers.InputLayer(input_shape=inputShape)
#    layer0  = CDFeedLayer(depth)
    layer0a = keras.layers.Flatten()
    layer1  = keras.layers.Dense(num_neurons, activation=activation)
    layer2  = keras.layers.Dense(num_neurons, activation=activation)
    layer3  = keras.layers.Dense(num_neurons, activation=activation)
    output  = keras.layers.Dense(1, activation='sigmoid')
        
    modelDNN = keras.Sequential([layer0, layer0a, layer1, layer2, layer3, output])
        
    return modelDNN

def NNCritic(inputShape, activation = 'relu', seed=2050, num_neurons=512, depth=128):
#    tf.random_normal_initializer(seed)
#    inlayer = keras.layers.Flatten(input_shape=inputShape)
    layer0  = keras.layers.Dense(depth, input_shape=inputShape)
#    layer00 = keras.layers.InputLayer(input_shape=inputShape)
#    layer0  = CDFeedLayer(depth)
    layer0a = keras.layers.Flatten()
    layer1  = keras.layers.Dense(num_neurons, activation=activation)
    layer2  = keras.layers.Dense(num_neurons, activation=activation)
    layer3  = keras.layers.Dense(num_neurons, activation=activation)
    output  = keras.layers.Dense(1, activation=None)   #kernel_regularizer=keras.regularizers.l2(l2=0.001)
        
    modelDNN = keras.Sequential([layer0, layer0a, layer1, layer2, layer3, output])
        
    return modelDNN

def NNDiscriminator2(inputShape, activation = 'relu', seed=2050, num_neurons=512):
#    tf.random_normal_initializer(seed)
#    inlayer = keras.layers.Input(shape=inputShape)
    inlayer = keras.layers.Flatten(input_shape=inputShape)
    layer0  = keras.layers.Dense(3840, activation=None)
    layer1  = keras.layers.Dense(num_neurons, activation=activation)
    layer2  = keras.layers.Dense(num_neurons, activation=activation)
    layer3  = keras.layers.Dense(num_neurons, activation=activation)
    output  = keras.layers.Dense(1, activation='sigmoid')
        
    modelDNN = keras.Sequential([inlayer, layer0, layer1, layer2, layer3, output])
        
    return modelDNN

def NNCritic2(inputShape, activation = 'relu', seed=2050, num_neurons=512): #(input shape = 30, 75)
#    tf.random_normal_initializer(seed)

#    inlayer = keras.layers.Input(shape=inputShape)
    inlayer = keras.layers.Flatten(input_shape=inputShape)
    layer0 = keras.layers.Dense(3840, activation=None)
    layer1  = keras.layers.Dense(num_neurons, activation=activation)
    layer2  = keras.layers.Dense(num_neurons, activation=activation)
    layer3  = keras.layers.Dense(num_neurons, activation=activation)
    output  = keras.layers.Dense(1, activation='sigmoid')   #kernel_regularizer=keras.regularizers.l2(l2=0.001)
        
    modelDNN = keras.Sequential([inlayer, layer0, layer1, layer2, layer3, output])
        
    return modelDNN

def NNDiscriminator3(inputShape, activation = 'relu', seed=2050, num_neurons=512):
#    tf.random_normal_initializer(seed)
#    inlayer = keras.layers.Input(shape=inputShape)
    inlayer = keras.layers.Flatten(input_shape=inputShape)
    layer0  = keras.layers.Dense(3840, activation=None)
    layer1  = keras.layers.Dense(num_neurons, activation=activation)
    layer2  = keras.layers.Dense(num_neurons, activation=activation)
    layer3  = keras.layers.Dense(num_neurons, activation=activation)
    output  = keras.layers.Dense(1, activation='sigmoid')
        
    modelDNN = keras.Sequential([inlayer, layer0, layer1, layer2, layer3, output])
        
    return modelDNN

def NNCritic3(inputShape, activation = 'relu', seed=2050, num_neurons=512): #(input shape = 30, 75)
#    tf.random_normal_initializer(seed)

#    inlayer = keras.layers.Input(shape=inputShape)
    inlayer = keras.layers.Flatten(input_shape=inputShape)
    layer0 = keras.layers.Dense(3840, activation=None)
    layer1  = keras.layers.Dense(num_neurons, activation=activation)
    layer2  = keras.layers.Dense(num_neurons, activation=activation)
    layer3  = keras.layers.Dense(num_neurons, activation=activation)
    output  = keras.layers.Dense(1, activation='sigmoid')   #kernel_regularizer=keras.regularizers.l2(l2=0.001)
        
    modelDNN = keras.Sequential([inlayer, layer0, layer1, layer2, layer3, output])
        
    return modelDNN


# RNN for the generator, two parts encoder and decoder
    
def RNNGenerator(inputShape, num_neurons=1500):
    
    encoderInput = keras.layers.Input(shape=(None, 75), name="encoder-In")
    encoderLyr1 = keras.layers.GRU(num_neurons, return_sequences=True, name="encoder-GRU1")
    encoderLyr1 = encoderLyr1(encoderInput)
     
    encoderLyr2 = keras.layers.GRU(num_neurons, return_state= True, name="encoder-GRU2")
    encoderOutput, stateH = encoderLyr2(encoderLyr1)

    #decoder component of generator structure
    decoderInput = keras.layers.Input(shape=(encoderOutput.shape), name="decoder-in")
    decoderLyr1  = keras.layers.GRU(num_neurons, return_sequences=True, name="decoder-GRU1")
    decoderLyr1  = decoderLyr1(decoderInput, initial_state=stateH)

    decoderLyr2 = keras.layers.GRU(num_neurons, name="decoder-GRU2")
    decoderOutput = decoderLyr2(decoderLyr1)
    
    decoderReshape = keras.layers.Reshape((-1,20,25,3))
    decoderOutput  = decoderReshape(decoderOutput)
    
    model = keras.Model([encoderInput, decoderInput], outputs=decoderOutput)
        
    return model

def RNNGenerator2(neurons, inputShape): #zData is 1024 vector of uiform random values
    
    #encoder side
    encoderCells = []
    encoderInputs = keras.layers.Input(shape=(None, 75), name="encoder-input")
    encoderCells.append(keras.layers.GRUCell(neurons, name="GRU-layer1"))
    encoderCells.append(keras.layers.GRUCell(neurons, name="GRU-layer2")) 
    
    encoder = keras.layers.RNN(encoderCells, return_state=True, name="encoder-RNN")
    encoderOutStates = encoder(encoderInputs)
    encoderStates = encoderOutStates[1:]
    
    #decoder side
    decoderCells = []
    decoderInputs = keras.layers.Input(shape=(None, neurons), name="decoder-input")
    decoderCells.append(keras.layers.GRUCell(neurons, name="GRU-layer3"))
    decoderCells.append(keras.layers.GRUCell(neurons, name="GRU-layer4"))
    decoder = keras.layers.RNN(decoderCells, return_sequences=True, return_state=True, name="decoder-RNN")
    
    decoderOutStates = decoder(decoderInputs, initial_state=encoderStates)
    decoderOutputs = decoderOutStates[0]
    
    
    decoderReshape = keras.layers.Reshape((20,25,3))
 #   decoderDense = keras.layers.Dense(neurons, activation="relu")
    decoderOutputs = decoderReshape(decoderOutputs)
    
    model = keras.Model(inputs=[encoderInputs, decoderInputs], outputs=decoderOutputs)
    
    return model
    

def RNNGenerator3(neurons, inputShape, zData):
    
    #encoder side
    encoderCells = []
    encoderInputs = keras.layers.Input(shape=(None, 75), name="encoder-input")

    encoderCells.append(keras.layers.GRUCell(neurons, name="GRU-layer1"))
    encoderCells.append(keras.layers.GRUCell(neurons, name="GRU-layer2")) 
    
    encoder = keras.layers.RNN(encoderCells, return_state=True, name="encoder-RNN")
    encoderOutStates = encoder(encoderInputs)
    encoderStates  = encoderOutStates[1:]
    #encoderOutData = encoderOutStates[0]
    
    #encoderOutData = tf.compat.v1.placeholder_with_default(encoderOutStates[0], [None, None, 1500])
    
    encoderOutData = tf.reshape(encoderOutStates[0], [-1, 1, 1500])
    
    # add  the z to the initial weights
    Wsi = tf.compat.v1.get_variable("Wsi", shape=[zData.shape[1], neurons], initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.001))
    encoderState1 = encoderStates[0] + tf.matmul(zData, Wsi)
    encoderState2 = encoderStates[1] + tf.matmul(zData, Wsi)
    newEncoderStates = [encoderState1, encoderState2]    
    
    #decoder side
    decoderCells = []
    decoderInputs = keras.layers.Input(shape=(None, neurons), name="decoder-input")
    decoderCells.append(keras.layers.GRUCell(neurons, name="GRU-layer3"))
    decoderCells.append(keras.layers.GRUCell(neurons, name="GRU-layer4"))
    decoder = keras.layers.RNN(decoderCells, return_sequences=True, return_state=True, name="decoder-RNN")
    
    decoderOutStates = decoder(encoderOutData, initial_state=newEncoderStates)
    decoderOutputs = decoderOutStates[0]
    
    
    decoderReshape = keras.layers.Reshape((20,25,3))
 #   decoderDense = keras.layers.Dense(neurons, activation="relu")
    decoderOutputs = decoderReshape(decoderOutputs)
    
    model = keras.Model(inputs=[encoderInputs], outputs=decoderOutputs)
    
    return model

def RNNGenerator4(neurons, inputShape, zShape):
  
    zInput       = keras.layers.Input(shape=(zShape), name="zInput")
#    latentDense  = keras.layers.Dense(neurons)(zInput)
#    latentDense  = keras.layers.ReLU()(latentDense)
#    latentOutput = keras.layers.Reshape((neurons))(latentDense)
    
    #encoder side
    encoderCells = []
    
    encoderInputs = keras.layers.Input(shape=(None, 75), name="encoder-input")
    
    encoderCells.append(keras.layers.GRUCell(neurons, name="GRU-layer1"))
    encoderCells.append(keras.layers.GRUCell(neurons, name="GRU-layer2")) 
    
    encoder = keras.layers.RNN(encoderCells, return_state=True, name="encoder-RNN")
    encoderOutStates = encoder(encoderInputs)
    encoderStates  = encoderOutStates[1:]
    #encoderOutData = encoderOutStates[0]
        
    encoderOutData = tf.reshape(encoderOutStates[0], [-1, 1, neurons])
    
    # add  the z to the initial weights
    Wsi = tf.compat.v1.get_variable("Wsi", shape=[zInput.shape[1], neurons], initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.001))
    encoderState1 = encoderStates[0] + tf.matmul(zInput, Wsi) #latentDense
    encoderState2 = encoderStates[1] + tf.matmul(zInput, Wsi) #latentDense
    newEncoderStates = [encoderState1, encoderState2]    
    
    #decoder side
    decoderCells = []
#    decoderInputs = keras.layers.Input(shape=(None, neurons), name="decoder-input")
    decoderCells.append(keras.layers.GRUCell(neurons, name="GRU-layer3"))
    decoderCells.append(keras.layers.GRUCell(neurons, name="GRU-layer4"))
    decoder = keras.layers.RNN(decoderCells, return_sequences=True, return_state=True, name="decoder-RNN")
    
    decoderOutStates = decoder(encoderOutData, initial_state=newEncoderStates)
    decoderOutputs = decoderOutStates[0]
    
    
    decoderReshape = keras.layers.Reshape((20,25,3))
    decoderOutputs = decoderReshape(decoderOutputs)
    
    model = keras.Model(inputs=[encoderInputs, zInput], outputs=decoderOutputs)
    
    return model

def RNNGenerator5(neurons, inputShape, zShape):
    
    zInput       = keras.layers.Input(shape=(zShape), name="zInput")
    latentDense  = keras.layers.Dense(neurons, activation='relu', name="latentDense")(zInput)
#    latentDense  = keras.layers.ReLU()(latentDense)
    
    #encoder side
    encoderCells = []
    
    encoderInputs = keras.layers.Input(shape=(None, 75), name="encoder-input")
    encoderDense  = keras.layers.Dense(neurons, activation=None, name="encoderDense")(encoderInputs)
    
    encoderCells.append(keras.layers.GRUCell(neurons, name="GRU-layer1"))
    encoderCells.append(keras.layers.GRUCell(neurons, name="GRU-layer2")) 
    
    encoder = keras.layers.RNN(encoderCells, return_sequences=False, return_state=True, name="encoder-RNN")
#    encoderOutStates = encoder(encoderInputs)
    encoderOutStates = encoder(encoderDense)
    encoderStates  = encoderOutStates[1:]
    encoderOutData = encoderOutStates[0]
        
    encoderOutData = tf.reshape(encoderOutStates[0], [-1, 1, neurons])
 #   encoderOutData = encoderOutData[:, -1, :]
    
    encoderState1 = encoderStates[0] + latentDense 
    encoderState2 = encoderStates[1] + latentDense 
    newEncoderStates = [encoderState1, encoderState2]    
    
    #decoder side
    decoderCells = []
#    decoderInputs = keras.layers.Input(shape=(None, neurons), name="decoder-input")
    decoderCells.append(keras.layers.GRUCell(neurons, name="GRU-layer3"))
    decoderCells.append(keras.layers.GRUCell(neurons, name="GRU-layer4"))
    decoder = keras.layers.RNN(decoderCells, return_sequences=True, return_state=True, name="decoder-RNN")
    
    decoderOutStates = decoder(encoderOutData, initial_state=newEncoderStates)
    decoderOutputs = decoderOutStates[0]

    decoderDense = keras.layers.Dense(1500, activation=None, name="decoderDense")(decoderOutputs)
    decoderReshape = keras.layers.Reshape((20,25,3))

#    decoderOutputs = decoderReshape(decoderOutputs)
    decoderOutputs = decoderReshape(decoderDense)
    
    model = keras.Model(inputs=[encoderInputs, zInput], outputs=decoderOutputs)
    
    return model

def RNNGenerator6(neurons, inputShape, zShape):
    
    zInput       = keras.layers.Input(shape=(zShape), name="zInput")
    latentDense  = keras.layers.Dense(neurons, activation='relu', name="latentDense")(zInput)
#    latentDense  = keras.layers.ReLU()(latentDense)
    
    #encoder side
    encoderCells = []
    
    encoderInputs = keras.layers.Input(shape=(None, 75), name="encoder-input")
    encoderDense  = keras.layers.Dense(neurons, activation=None, name="encoderDense")(encoderInputs)
    
    encoderCells.append(keras.layers.GRUCell(neurons, name="GRU-layer1"))
    encoderCells.append(keras.layers.GRUCell(neurons, name="GRU-layer2")) 
    
    encoder = keras.layers.RNN(encoderCells, return_sequences=False, return_state=True, name="encoder-RNN")
#    encoderOutStates = encoder(encoderInputs)
    encoderOutStates = encoder(encoderDense)
    encoderStates  = encoderOutStates[1:]
    encoderOutData = encoderOutStates[0]
        
    encoderOutData = tf.reshape(encoderOutStates[0], [-1, 1, neurons])
 #   encoderOutData = encoderOutData[:, -1, :]
    
    encoderState1 = encoderStates[0] + latentDense 
    encoderState2 = encoderStates[1] + latentDense 
    newEncoderStates = [encoderState1, encoderState2]    
    
    #decoder side
    decoderCells = []
#    decoderInputs = keras.layers.Input(shape=(None, neurons), name="decoder-input")
    decoderCells.append(keras.layers.GRUCell(neurons, name="GRU-layer3"))
    decoderCells.append(keras.layers.GRUCell(neurons, name="GRU-layer4"))
    decoder = keras.layers.RNN(decoderCells, return_sequences=True, return_state=True, name="decoder-RNN")
    
    decoderOutStates = decoder(encoderOutData, initial_state=newEncoderStates)
    decoderOutputs = decoderOutStates[0]

    decoderDense = keras.layers.Dense(1500, activation=None, name="decoderDense")(decoderOutputs)
    decoderReshape = keras.layers.Reshape((20,25,3))

 #   decoderOutputs = decoderReshape(decoderOutputs)
    decoderOutputs = decoderReshape(decoderDense)
    
    model = keras.Model(inputs=[encoderInputs, zInput], outputs=decoderOutputs)
    
    return model

def RNNGenerator7(neurons, inputShape, zShape):
    
    jts = int(inputShape[1] / 3)
    out = 20 * inputShape[1]
    
    zInput       = keras.layers.Input(shape=zShape, name="zInput")
    latentNoise  = NoiseLayer(neurons, zShape, name="latentNoise", trainable=False)(zInput)
#    latentDense  = keras.layers.Dense(neurons, activation='relu', name="latentDense")(zInput)
    
    #encoder side
    encoderInputs = keras.layers.Input(shape=inputShape, name="encoder-input")
#    encoderDense  = keras.layers.Dense(zShape, activation=None, name="encoderDense")(encoderInputs)
    encoderLinear = LinearLayer(zShape, inputShape, name="encoderLinear", bias=True)(encoderInputs)
    
    encoderCells = []
    encoderCells.append(keras.layers.GRUCell(neurons, name="GRU-layer1"))
    encoderCells.append(keras.layers.GRUCell(neurons, name="GRU-layer2")) 
    
    encoder = keras.layers.RNN(encoderCells, return_sequences=False, return_state=True, name="encoder-RNN")
    encoderOutStates = encoder(encoderLinear)
    encoderStates  = encoderOutStates[1:]
    encoderOutData = encoderOutStates[0]
        
#    midDense = keras.layers.Dense(zShape, input_shape=encoderOutData.shape, name="midDense")(encoderOutData)
    midLinear = CenterLayer(zShape, name="midLayer")(encoderOutData)
    
    midDense = keras.layers.Reshape((-1, zShape))(midLinear)
    #encoderOutData = tf.reshape(encoderOutStates[0], [-1, 1, neurons])
 #   encoderOutData = encoderOutData[:, -1, :]
    
    encoderState1 = encoderStates[0] + latentNoise 
    encoderState2 = encoderStates[1] + latentNoise 
    newEncoderStates = [encoderState1, encoderState2]    
    
    #decoder side
    decoderCells = []
    decoderCells.append(keras.layers.GRUCell(neurons, name="GRU-layer3"))
    decoderCells.append(keras.layers.GRUCell(neurons, name="GRU-layer4"))
    decoder = keras.layers.RNN(decoderCells, return_sequences=True, return_state=True, name="decoder-RNN")
    
    decoderOutStates = decoder(midDense, initial_state=newEncoderStates)
    decoderOutputs = decoderOutStates[0]

    decoderDense = keras.layers.Dense(out, activation=None, name="decoderDense")(decoderOutputs)
    decoderReshape = keras.layers.Reshape((20,jts,3))(decoderDense)

 #   decoderOutputs = decoderReshape(decoderOutputs)
 #   decoderOutputs = decoderReshape(decoderDense)
    
    model = keras.Model(inputs=[encoderInputs, zInput], outputs=decoderReshape)
    
    return model

#generator model for 2d to 3d generation, 
def D2TOD3Generator(neurons, inputShape, zShape):
    
    jts = int(inputShape[1] / 2)
    out = 20 * jts * 3
    
    zInput       = keras.layers.Input(shape=zShape, name="zInput-2d")
    latentNoise  = NoiseLayer(neurons, zShape, name="latentNoise-2d", trainable=False)(zInput)
    latentDense  = keras.layers.Dense(neurons, activation='relu', name="latentDense-2d")(zInput)
    
    #encoder side
    encoderInputs = keras.layers.Input(shape=inputShape, name="encoder-input-2d")
#    encoderDense  = keras.layers.Dense(zShape, activation=None, name="encoderDense")(encoderInputs)
    encoderLinear = LinearLayer(zShape, inputShape, name="encoderLinear-2d", bias=True)(encoderInputs)
    
    encoderCells = []
    encoderCells.append(keras.layers.GRUCell(neurons, name="GRU-layer1-2d"))
    encoderCells.append(keras.layers.GRUCell(neurons, name="GRU-layer2-2d")) 
    
    encoder = keras.layers.RNN(encoderCells, return_sequences=False, return_state=True, name="encoder-RNN-2d")
    encoderOutStates = encoder(encoderLinear)
    encoderStates  = encoderOutStates[1:]
    encoderOutData = encoderOutStates[0]
        
#    midDense = keras.layers.Dense(zShape, input_shape=encoderOutData.shape, name="midDense")(encoderOutData)
    midLinear = CenterLayer(zShape, name="midLayer-2d")(encoderOutData)
    
    midDense = keras.layers.Reshape((-1, zShape))(midLinear)
    #encoderOutData = tf.reshape(encoderOutStates[0], [-1, 1, neurons])
 #   encoderOutData = encoderOutData[:, -1, :]
    
    encoderState1 = encoderStates[0] + latentNoise 
    encoderState2 = encoderStates[1] + latentNoise 
    newEncoderStates = [encoderState1, encoderState2]    
    
    #decoder side
    decoderCells = []
    decoderCells.append(keras.layers.GRUCell(neurons, name="GRU-layer3-2d"))
    decoderCells.append(keras.layers.GRUCell(neurons, name="GRU-layer4-2d"))
    decoder = keras.layers.RNN(decoderCells, return_sequences=True, return_state=True, name="decoder-RNN-2d")
    
    decoderOutStates = decoder(midDense, initial_state=newEncoderStates)
    decoderOutputs = decoderOutStates[0]

    decoderDense = keras.layers.Dense(out, activation=None, name="decoderDense-2d")(decoderOutputs)
    decoderReshape = keras.layers.Reshape((20,jts,3))(decoderDense)

 #   decoderOutputs = decoderReshape(decoderOutputs)
 #   decoderOutputs = decoderReshape(decoderDense)
    
    model = keras.Model(inputs=[encoderInputs, zInput], outputs=decoderReshape)
#    model = keras.Model(inputs=[encoderInputs], outputs=decoderReshape)
    
    return model

#generator model for 2d to 3d generation, 
def D2TOD3Generator10x10(neurons, inputShape, zShape):
    
    jts = int(inputShape[1] / 2)
    out = 10 * jts * 3
    
    zInput       = keras.layers.Input(shape=zShape, name="zInput-2d")
    latentNoise  = NoiseLayer(neurons, zShape, name="latentNoise-2d", trainable=False)(zInput)
    latentDense  = keras.layers.Dense(neurons, activation='relu', name="latentDense-2d")(zInput)
    
    #encoder side
    encoderInputs = keras.layers.Input(shape=inputShape, name="encoder-input-2d")
    encoderLinear = LinearLayer(zShape, inputShape, name="encoderLinear-2d", bias=True)(encoderInputs)
    
    encoderCells = []
    encoderCells.append(keras.layers.GRUCell(neurons, name="GRU-layer1-2d"))
    encoderCells.append(keras.layers.GRUCell(neurons, name="GRU-layer2-2d")) 
    
    encoder = keras.layers.RNN(encoderCells, return_sequences=False, return_state=True, name="encoder-RNN-2d")
    encoderOutStates = encoder(encoderLinear)
    encoderStates  = encoderOutStates[1:]
    encoderOutData = encoderOutStates[0]
        
    midLinear = CenterLayer(zShape, name="midLayer-2d")(encoderOutData)
    
    midDense = keras.layers.Reshape((-1, zShape))(midLinear)
    
    encoderState1 = encoderStates[0] + latentNoise 
    encoderState2 = encoderStates[1] + latentNoise 
    newEncoderStates = [encoderState1, encoderState2]    
    
    #decoder side
    decoderCells = []
    decoderCells.append(keras.layers.GRUCell(neurons, name="GRU-layer3-2d"))
    decoderCells.append(keras.layers.GRUCell(neurons, name="GRU-layer4-2d"))
    decoder = keras.layers.RNN(decoderCells, return_sequences=True, return_state=True, name="decoder-RNN-2d")
    
    decoderOutStates = decoder(midDense, initial_state=newEncoderStates)
    decoderOutputs = decoderOutStates[0]

    decoderDense = keras.layers.Dense(out, activation=None, name="decoderDense-2d")(decoderOutputs)
    decoderReshape = keras.layers.Reshape((10,jts,3))(decoderDense)
    
    model = keras.Model(inputs=[encoderInputs, zInput], outputs=decoderReshape)
    
    return model

#generator model for 2d to 3d generation, 
def D2TOD3Generator10x10_Z(neurons, inputShape, zDimension, zShape):
    
    jts = int(inputShape[1] / 2)
    out = 10 * jts * 3
    
    zInput       = keras.layers.Input(shape=zShape, name="zInput-2d")
    latentNoise  = NoiseLayerZ(zShape, zDimension, inputShape, name="latentNoise-2d", trainable=False)(zInput)
    
    #encoder side
    encoderInputs = keras.layers.Input(shape=inputShape, name="encoder-input-2d")
    encoderAddZ = addZLayer(inputShape, name="Add-Z-dimension", trainable=False, bias=False)([encoderInputs, latentNoise])
    
    encoderLinear = LinearLayer(zShape, inputShape, name="encoderLinear-2d", bias=True)(encoderAddZ)
    
    encoderCells = []
    encoderCells.append(keras.layers.GRUCell(neurons, name="GRU-layer1-2d"))
    encoderCells.append(keras.layers.GRUCell(neurons, name="GRU-layer2-2d")) 
    
    encoder = keras.layers.RNN(encoderCells, return_sequences=False, return_state=True, name="encoder-RNN-2d")
    encoderOutStates = encoder(encoderLinear)
    encoderStates  = encoderOutStates[1:]
    encoderOutData = encoderOutStates[0]
        
    midLinear = CenterLayer(zShape, name="midLayer-2d")(encoderOutData)
    
    midDense = keras.layers.Reshape((-1, zShape))(midLinear)
    
    encoderState1 = encoderStates[0] 
    encoderState2 = encoderStates[1]
    newEncoderStates = [encoderState1, encoderState2]    
    
    #decoder side
    decoderCells = []
    decoderCells.append(keras.layers.GRUCell(neurons, name="GRU-layer3-2d"))
    decoderCells.append(keras.layers.GRUCell(neurons, name="GRU-layer4-2d"))
    decoder = keras.layers.RNN(decoderCells, return_sequences=True, return_state=True, name="decoder-RNN-2d")
    
    decoderOutStates = decoder(midDense, initial_state=newEncoderStates)
    decoderOutputs = decoderOutStates[0]

    decoderDense = keras.layers.Dense(out, activation=None, name="decoderDense-2d")(decoderOutputs)
    decoderReshape = keras.layers.Reshape((10,jts,3))(decoderDense)
    
    model = keras.Model(inputs=[encoderInputs, zInput], outputs=decoderReshape)
    
    return model

#generator model for 2d to 3d generation, this model adds the noise to the third dimension, z dimension 
"""
    zInput       = keras.layers.Input(shape=zShape, name="zInput")
    latentNoise  = NoiseLayer(neurons, zShape, name="latentNoise", trainable=False)(zInput)
#    latentDense  = keras.layers.Dense(neurons, activation='relu', name="latentDense")(zInput)
    
    #encoder side
    encoderInputs = keras.layers.Input(shape=inputShape, name="encoder-input")
#    encoderDense  = keras.layers.Dense(zShape, activation=None, name="encoderDense")(encoderInputs)
    encoderLinear = LinearLayer(zShape, inputShape, name="encoderLinear", bias=True)(encoderInputs)
    
    encoderCells = []
    encoderCells.append(keras.layers.GRUCell(neurons, name="GRU-layer1"))
    encoderCells.append(keras.layers.GRUCell(neurons, name="GRU-layer2")) 
    
    encoder = keras.layers.RNN(encoderCells, return_sequences=False, return_state=True, name="encoder-RNN")
    encoderOutStates = encoder(encoderLinear)
    encoderStates  = encoderOutStates[1:]
    encoderOutData = encoderOutStates[0]
        
#    midDense = keras.layers.Dense(zShape, input_shape=encoderOutData.shape, name="midDense")(encoderOutData)
    midLinear = CenterLayer(zShape, name="midLayer")(encoderOutData)
    
    midDense = keras.layers.Reshape((-1, zShape))(midLinear)
    #encoderOutData = tf.reshape(encoderOutStates[0], [-1, 1, neurons])
 #   encoderOutData = encoderOutData[:, -1, :]
    
    encoderState1 = encoderStates[0] + latentNoise 
    encoderState2 = encoderStates[1] + latentNoise 
    newEncoderStates = [encoderState1, encoderState2]    
    
    #decoder side
    decoderCells = []
    decoderCells.append(keras.layers.GRUCell(neurons, name="GRU-layer3"))
    decoderCells.append(keras.layers.GRUCell(neurons, name="GRU-layer4"))
    decoder = keras.layers.RNN(decoderCells, return_sequences=True, return_state=True, name="decoder-RNN")
    
    decoderOutStates = decoder(midDense, initial_state=newEncoderStates)
    decoderOutputs = decoderOutStates[0]

    decoderDense = keras.layers.Dense(1500, activation=None, name="decoderDense")(decoderOutputs)
    decoderReshape = keras.layers.Reshape((20,25,3))(decoderDense)

 #   decoderOutputs = decoderReshape(decoderOutputs)
 #   decoderOutputs = decoderReshape(decoderDense)
    
    model = keras.Model(inputs=[encoderInputs, zInput], outputs=decoderReshape)
    
    return model
"""

def D2TOD3GeneratorZ(neurons, inputShape, zDimension, zShape):
    
    jts = int(inputShape[1] / 2)
    out = 20 * jts * 3
    
    zInput       = keras.layers.Input(shape=zShape, name="zInput-2d")
    latentNoise  = NoiseLayerZ(zShape, zDimension, inputShape, name="latentNoise-2d", trainable=False)(zInput)
    
    #encoder side
    encoderInputs = keras.layers.Input(shape=inputShape, name="encoder-input-2d")

    encoderAddZ = addZLayer(inputShape, name="Add-Z-dimension", trainable=False, bias=False)([encoderInputs, latentNoise])

    encoderLinear = LinearLayer(zShape, inputShape, name="encoderLinear-2d", bias=True)(encoderAddZ)
    
    encoderCells = []
    encoderCells.append(keras.layers.GRUCell(neurons, name="GRU-layer1-2d"))
    encoderCells.append(keras.layers.GRUCell(neurons, name="GRU-layer2-2d")) 
    
    encoder = keras.layers.RNN(encoderCells, return_sequences=False, return_state=True, name="encoder-RNN-2d")
    encoderOutStates = encoder(encoderLinear)
    encoderStates  = encoderOutStates[1:]
    encoderOutData = encoderOutStates[0]
        
#    midDense = keras.layers.Dense(zShape, input_shape=encoderOutData.shape, name="midDense")(encoderOutData)
    midLinear = CenterLayer(zShape, name="midLayer-2d")(encoderOutData)
    
    midDense = keras.layers.Reshape((-1, zShape))(midLinear)
    #encoderOutData = tf.reshape(encoderOutStates[0], [-1, 1, neurons])
 #   encoderOutData = encoderOutData[:, -1, :]
    
    encoderState1 = encoderStates[0]  
    encoderState2 = encoderStates[1]  
    newEncoderStates = [encoderState1, encoderState2]    
    
    #decoder side
    decoderCells = []
    decoderCells.append(keras.layers.GRUCell(neurons, name="GRU-layer3-2d"))
    decoderCells.append(keras.layers.GRUCell(neurons, name="GRU-layer4-2d"))
    decoder = keras.layers.RNN(decoderCells, return_sequences=True, return_state=True, name="decoder-RNN-2d")
    
    decoderOutStates = decoder(midDense, initial_state=newEncoderStates)
    decoderOutputs = decoderOutStates[0]

    decoderDense = keras.layers.Dense(out, activation=None, name="decoderDense-2d")(decoderOutputs)
    decoderReshape = keras.layers.Reshape((20,jts,3))(decoderDense)

 #   decoderOutputs = decoderReshape(decoderOutputs)
 #   decoderOutputs = decoderReshape(decoderDense)
    
    model = keras.Model(inputs=[encoderInputs, zInput], outputs=decoderReshape)
#    model = keras.Model(inputs=[encoderInputs], outputs=decoderReshape)
    
    return model

#  create a number of predictions for one sample of 10 stating images.
#  dont use the whole bactch just the first [0] of the pastData.
def generateSkeltons(inputData, pastData, zData, gan, normaliser, batchIndex=0):
    
    skeletonData  = []
    probabilities = []
    
    _, zSize = zData.shape
    
    #save the ground thruth for the first line of the display
    _, sequenceLength, _, _ = inputData.shape
    
    groundTruth = normaliser.unnormalize(inputData[0])
    skeletonData.append(groundTruth)
    
    #reshape to suit the generator input batch, seq length, 25*3
    inputSeqLength, joints, dims = pastData[0].shape
#    inputBatch = np.reshape(inputBatch, (11, sequenceLength, 25*3))
    pastReshaped  = np.reshape(pastData[0], (1, pastData[0].shape, joints*dims)) 
    
    for z in zData:
        z = np.reshape(z, (1, zSize))
        prediction = gan.predict([pastReshaped, z])
        sequence = np.concatenate((pastData[0], prediction[0]))
        dsequence = np.reshape(sequence, (1, sequenceLength, joints*dims))

        prob = gan.discriminator(dsequence, training=False)
#        sequence = normaliser.mean_std_unnormalize(sequence, std_factor=2.0)
        sequence = normaliser.unnormalize(sequence)

        skeletonData.append(sequence)
        probabilities.append(prob[0].numpy()[0])

    return skeletonData, probabilities

#  create a number of predictions for one sample of 10 drive 2D images,
#  dont use the whole bactch just the first [0] of the pastData.
# skeletonData return generated pose sequences for the noise zData.  
# skeletonData[0] contains 2d ground truth
# skeletonData[1] contains 3d ground truth


def generateSkeltons2D(inputData, pastData, d3GroundT, zData, gan, normaliser3d, normaliser2d, batchIndex=0):
    #d3GroundT orginal 
    skeletonData  = []
    probabilities = []
    
    _, zSize = zData.shape
    
    #save the ground thruth for the first line of the display
    _, sequenceLength, jts, _ = inputData.shape
    _, inputLength, _, _ = pastData.shape

    #add the 2D ground truth as first line
    skeletonData.append(normaliser2d.unnormalise(inputData[0]))

    #add the 3D ground truth as second line
    groundTruth = normaliser3d.unnormalize(d3GroundT)
    skeletonData.append(groundTruth[0])
    
    #get normalised and unnormalised first10 of the 3G ground truth
#    d3First10_un = groundTruth[:,0:inputLength,:,:] #for joining into 20 generated 3D sequence
#    d3First10_n  = d3GroundT[:,0:inputLength,:,:]

    d2First10_n = inputData[0:1,0:10,:,:] #3D version of the driver sequence for adding to the front of the prediction

    #reshape to suit the generator input batch, seq length, 25*3
    inputSeqLength, joints, dims = pastData[0].shape
    pastReshaped = np.reshape(pastData[0], (1, inputSeqLength, joints*dims)) #2D
#    pastReshaped = np.reshape(d2First10_n[0], (1, inputSeqLength, joints*3)) #3D
 
    for z in zData:
        z = np.reshape(z, (1, zSize))
        
        prediction = gan.predict([pastReshaped, z])

        sequence = prediction
#        sequence = np.concatenate((d2First10_n, prediction), axis=1) #removed for the 10 lenght sample sequence, was 30
        
        #only add the discriminator probabilities if 30 sequence length
        if sequenceLength == 30:
            seq2 = np.reshape(sequence,(1, sequenceLength, jts*3))
            prob = gan.discriminator(seq2, training=False)
            probabilities.append(prob[0].numpy()[0])

        sequence = normaliser2d.unnormalise(sequence)        
        sequence = np.reshape(sequence,(sequenceLength, jts, 3))  #+10 is a hack for 10 length sampled sequences, remove if longer sequences are sampled.
        
        skeletonData.append(sequence)

    return skeletonData, probabilities

#  create a number of predictions for one sample of 10 stating images.
#  dont use the whole bactch just the first [0] of the pastData.
def generateSkeltons2(inputData, pastData, zData, hpgan, normaliser, batchIndex=0):
    
    skeletonData  = []
    probabilities = []
    
    _, zSize = zData.shape
    
    #save the ground thruth for the first line of the display
    sequenceLength, _, _ = inputData.shape
    groundTruth = normaliser.unnormalize(inputData)
    skeletonData.append(groundTruth)
    
    #reshape to suit the generator input batch, seq length, 25*3
    inputSeqLength, joints, dims = pastData.shape
#    inputBatch = np.reshape(inputBatch, (11, sequenceLength, 25*3))
    pastReshaped  = np.reshape(pastData, (1, inputSeqLength, joints*dims)) 
    
    for z in zData:
        z = np.reshape(z, (1, zSize))
        prediction = hpgan.predict([pastReshaped, z])
        sequence = np.concatenate((pastData, prediction[0]))
        dsequence = np.reshape(sequence, (1, sequenceLength, joints*dims))

        prob = hpgan.discriminator(dsequence, training=False)
        sequence = normaliser.unnormalize(sequence)

        skeletonData.append(sequence)
        probabilities.append(prob[0].numpy()[0])

    return skeletonData, probabilities

# implementation of a normaliser as used by the HP-GAN for the 3D data but for the pelvis centered 2D representation.
class Stats2D(object):
    def __init__(self, inputs, factor=2.0, meanType="local"):
        self.globalMean = np.mean(inputs, axis=(0,1,2))
        self.globalStd  = np.std(inputs, axis=(0,1,2))        
        self.mean = np.mean(inputs, axis=(0,1))
        self.std  = np.std(inputs, axis=(0,1))
        #at the pelvis the mean will be zero, adjust to avoid divide by zero
        self.std = np.where(self.std == 0.0, 0.00001, self.std)
        self.max = np.max(inputs, axis=(0,1))
        self.min = np.min(inputs, axis=(0,1))
        self.globalMax = np.max(inputs, axis=(0,1,2))
        self.globalMin = np.min(inputs, axis=(0,1,2))
        # remove 0 to avoid divide by zero issues
#        self.max = np.where(self.max == 0.0, 0.000001, self.max)
#        self.min = np.where(self.min == 0.0, 0.00001, self.min)
        self.factor = factor
        self.meanType = meanType
        
        #for compatability with other calling functions
    def normalize(self, inputs):
        return self.normalise(inputs)        
    
    def unnormalize(self, inputs):
        return self.unnormalise(inputs)
    
    def normalise(self, inputs):
        normed=None
        if self.meanType == "local":
            normed = self._mean_std_normalize(inputs)
        elif self.meanType == "linear":
            normed = self._range_normalise(inputs)
        elif self.meanType == "global":
            normed = self._global_mean_normalize(inputs)
        elif self.meanType == "global_linear":
            normed = self._global_range_normalise(inputs)
        return normed
        
    def unnormalise(self, inputs):
        unnormed=None
        if self.meanType == "local":
            unnormed = self._mean_std_unnormalize(inputs)
        elif self.meanType == "linear":
            unnormed = self._range_unnormalise(inputs)
        elif self.meanType == "global":
            unnormed = self._global_mean_unnormalize(inputs)
        elif self.meanType == "global_linear":
            unnormed = self._global_range_unnormalise(inputs)
        return unnormed
    
    # normalise between -1 and 1
    def _range_normalise(self, inputs):
        mdiff = self.max - self.min
        mdiff = np.where(mdiff==0, 0.00001, mdiff)
        normed = np.where(inputs == 0.0, 0.0, (2 * ((inputs - self.min) / mdiff)) - 1.0 )
#        normed = (2 * ((inputs - self.min) / (self.max - self.min))) - 1.0
        return normed
    
    #unnormalise the -1 to 1 normalisation
    def _range_unnormalise(self, inputs):
        unnorm = np.where(inputs == 0.0, 0.0, ((self.max - self.min) * ((inputs + 1.0) / 2.0)) + self.min)
#        unnorm = ((self.max - self.min) * ((inputs + 1.0) / 2.0)) + self.min
        return unnorm
    
    # normalise between -1 and 1
    def _global_range_normalise(self, inputs):
        mdiff = self.globalMax - self.globalMin
        mdiff = np.where(mdiff==0, 0.00001, mdiff)
        normed = np.where(inputs == 0.0, 0.0, (2 * ((inputs - self.globalMin) / mdiff)) - 1.0 )
#        normed = (2 * ((inputs - self.min) / (self.max - self.min))) - 1.0
        return normed
    
    #unnormalise the -1 to 1 normalisation
    def _global_range_unnormalise(self, inputs):
        unnorm = np.where(inputs == 0.0, 0.0, ((self.globalMax - self.globalMin) * ((inputs + 1.0) / 2.0)) + self.globalMin)
#        unnorm = ((self.max - self.min) * ((inputs + 1.0) / 2.0)) + self.min
        return unnorm
    
    
    def _mean_std_normalize(self, inputs):
        normed =  (inputs - self.mean) / (self.std * self.factor)
        return normed
    
    def _mean_std_unnormalize(self, inputs):
        normed = (inputs * self.std * self.factor) + self.mean 
        return normed
    
    def _global_mean_normalize(self, inputs):
        normed =  (inputs - self.globalMean) / (self.globalStd * self.factor)
        return normed
    
    def _global_mean_unnormalize(self, inputs):
        unnormed = (inputs * self.globalStd * self.factor) + self.globalMean 
        return unnormed

if __name__ == "__main__":
#    disc = NNDiscriminator((30, 25*3))
#    disc.compile(optimizer=keras.optimizers.Adam(0.00005), loss=keras.losses.binary_crossentropy)
#    disc.summary()

    
#    critic = NNCritic((30, 25*3))
#    critic.summary()
#    critic.compile(optimizer=keras.optimizers.Adam(0.00005), loss=keras.losses.categorical_crossentropy)

    zData = tf.random.uniform([128], maxval=0.1, minval=-0.1)
    zData = tf.reshape(zData, [-1, 128])
    optimiser = keras.optimizers.Adam(learning_rate=0.00005)
#    generator = RNNGenerator3(1500, 25*3, zData)
    dim = 128
#    generator = RNNGenerator4(1500, 25*3, (dim))
#    generator = RNNGenerator7(1024, (10, 25*3), (dim))

#    generator = D2TOD3Generator(1024, (10, 25*2), (dim))
#    generator = D2TOD3GeneratorZ(1024, (10, 25*2), (10, 25, 1), (dim))
    generator = D2TOD3Generator10x10_Z(1024, (10, 25*2), (10, 25, 1), (dim))


    generator.summary()
    generator.compile(optimizer=optimiser, loss="mse")
    
       
        
 