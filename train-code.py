
# -train ./Human-Ready/train_map.csv -test ./Human-Ready/test_map.csv -dataset human36m -ccf metadata.xml -out results -epochs 125 -dnf ./Human-Ready/data_statistics.h5
# -train ./Data-Prepared/train_map.csv -test ./Data-Prepared/test_map.csv -dataset nturgbd -out results -epochs 125 -dnf ./Data-Prepared/data_statistics.h5

import sys
import time
import os
import math
import csv
import argparse
import numpy as np
import logging
import random as rnd
import h5py

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Filter out all logs.

import tensorflow as tf

from braniac import nn as nn
from braniac.format import SourceFactory
from braniac.utils import DataPreprocessing, NormalizationMode
from braniac.viz import Skeleton2D
from braniac.readers.body import SequenceBodyReader
from braniac.models.body import RNNDiscriminator, NNDiscriminator, SequenceToSequenceGenerator
from braniac.format.kinect_v2 import Body, Joint, JointType


import matplotlib.pyplot as plt

import myPlotter as myplt
import tensorflow.keras as keras
import tensorflow.keras.utils as utils
from math import floor
from sklearn import metrics
import NNetworks as nnet #my NNetworks, discriminator, critic and generator
import DataHouse as dh #my activity sequence processing library

def plotter(genLoss, criticLoss, discrimLoss, epoch, save=True):
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    ax1.grid()
    ax2.grid()
    ax3.grid()

    #generator loss
    ax1.plot(genLoss)
    ax1.set_title("Generator Loss")
    ax1.set(ylabel="Loss")
    ax1.set(xlabel="Epoch")
    ax1.legend(["generator"])
    
    #Critic
    ax2.plot(criticLoss)
    ax2.set_title("Critic Loss")
    ax2.set(ylabel="Loss")
    ax2.set(xlabel="Epoch")
    ax2.legend(["critic"])
    
    #discriminator
    ax3.plot(discrimLoss)
    ax3.set_title("Discriminator Loss")
    ax3.set(ylabel="Loss")
    ax3.set(xlabel="Epoch")
    ax3.legend(["discriminator"])
    
    fig.tight_layout()
#    plt.show()
    
    if save:
        fig.savefig("results/output/plots/trainingPlot_{}.png".format(epoch), format='png')

def plotter2(boneLoss, consistancyLoss, numSample, save=True):
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.grid()
    ax2.grid()

    #generator loss
    ax1.plot(boneLoss)
    ax1.set_title("Bone Loss")
    ax1.set(ylabel="Loss")
    ax1.set(xlabel="numSample")
    ax1.legend(["Bone Loss"])
    
    #Critic
    ax2.plot(consistancyLoss)
    ax2.set_title("Consistancy Loss")
    ax2.set(ylabel="Loss")
    ax2.set(xlabel="numSample")
    ax2.legend(["critic"])
    
    fig.tight_layout()
#    plt.show()
    
    if save:
        fig.savefig("results/output/plots/testingPlot_{}.png".format(numSample), format='png')

# setting up the model

# class that manges the training step and loss functions and associated model updates
class HPGAN(keras.Model):
    def __init__(self, critic, generator, discriminator, zGenerator, criticSteps=10, genSteps=2, latentDims=128):
        super(HPGAN, self).__init__()
        self.critic = critic
        self.generator = generator
        self.discriminator = discriminator
        self.zGenerator = zGenerator
        self.criticSteps = criticSteps
        self.genSteps = genSteps
        self.latentDims = latentDims
                
    def compile(self, cOptim, gOptim, dOptim, criticLoss, genLoss, discLoss):
        super(HPGAN, self).compile()
        self.cOptim = cOptim
        self.gOptim = gOptim
        self.dOptim = dOptim
        self.criticLoss = criticLoss
        self.genLoss = genLoss
        self.discLoss = discLoss
    
    def setLengths(self, batchSize, inputLength, outputLength):
        self.batchSize = batchSize
        self.inputLength = inputLength
        self.outputLength = outputLength
        self.sequenceLength = inputLength + outputLength

    def setConstants(self, body_inf, lamda=10.0, alpha=0.001, beta=0.01):
        self.body_inf = body_inf
        self.lamda = lamda #applied to gradient penalty
        self.alpha = alpha #applied to regularisers
        self.beta  = beta #applied to bone loss
        #random z data generator parameters
        self.z_rand_type = 'uniform'
        self.z_rand_params = {'low':-0.1, 'high':0.1, 'mean':0.0, 'std':0.2}
        
    #gradient penality as per the paper
    def gradientPenalty(self, realSeq, fakeSeq):        
#        epilson = tf.random.uniform([self.batchSize, self.sequenceLength, (25*3)], 0.0, 1.0)   
        epilson = tf.random.uniform([self.batchSize, 1, 1], 0.0, 1.0)   
        
        xHat = (epilson*realSeq) + ((1.0-epilson) * fakeSeq)
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(xHat)
            dxHat = self.critic(xHat, training=True)
            
        gradients   = gp_tape.gradient(dxHat, xHat)
        gradientsL2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1,2]))
        gpLoss = tf.reduce_mean(tf.square(gradientsL2 - 1.0))
  #      tf.print(gpLoss)
        
        return gpLoss
    
    #as per the orginal WGAN-GP paper
    def gradientPenaltyN(self, realSeq, fakeSeq):  
        epilson = tf.random.uniform([self.batchSize, self.sequenceLength, (25*3)], 0.0, 1.0)

        diff = fakeSeq - realSeq
        interpolate = realSeq + (epilson*diff)
        
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolate)
            localPred = self.critic(interpolate, training=True)  
            
        gradients   = gp_tape.gradient(localPred, [interpolate])[0]
        gradientsL2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1,2]))
        gpLoss = tf.reduce_mean((gradientsL2 - 1.0) ** 2)

        return gpLoss
    
    def call(self, inputs):
        return inputs

    def train_step(self, data):
        
        realSequence = data   
        _, _, joints, dims = realSequence.shape
#        cLoss, gLoss, dLoss, gradPen, bLoss, pgLoss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        zData = self.zGenerator(self.z_rand_type, self.z_rand_params, shape=[self.batchSize, self.latentDims])
        realSequence = tf.reshape(realSequence, [self.batchSize, self.sequenceLength, (joints*dims)])
        first10 = realSequence[:,:self.inputLength,:]
        #train critic
        for i in range(self.criticSteps):
#            zData   = self.zGenerator.getZdata(latentDims, batchSize=self.batchSize).astype("float32")

            with tf.GradientTape() as tape:

            #fakeSeq = self.generator([realSequence, zData], training=True)  #this likely shoulld be chnages to only the first10 rather than the full 30
                fakeSeq = self.generator([first10, zData], training=True)
    
                fakeSeq = tf.reshape(fakeSeq,[self.batchSize, self.outputLength, (joints*dims)])
                sequence   = tf.concat([first10, fakeSeq], axis=1)
                
                try:
                    tf.debugging.check_numerics(realSequence, "Critic train - realSequence")
                except Exception as e:
                    assert "Checking for NaN " in e.message
                    
                try:
                    tf.debugging.check_numerics(sequence, "Critic train - sequence")
                except Exception as e:
                   assert "Checking for NaN " in e.message
                
                fakeLogits = self.critic(sequence, training=True)
                realLogits = self.critic(realSequence, training=True)
                
                try:
                    tf.debugging.check_numerics(fakeLogits, "Critic train - fakeLogits")
                except Exception as e:
                    assert "Checking for NaN " in e.message
                
            #    tf.print(realLogits)                
                try:
                    tf.debugging.check_numerics(realLogits, "Critic train - realLogits")
                except Exception as e:
                    assert "Checking for NaN " in e.message
                
                cLoss = self.criticLoss(realLogits, fakeLogits)
#                gradPen = self.gradientPenaltyN(realSequence, sequence)
                gradPen2 = self.gradientPenalty(realSequence, sequence) #both gradient penalty calculations result in the same answer

                #calculate the regualriser
                regulariserC = tf.add_n([tf.nn.l2_loss(p) for p in self.critic.weights])
                cLoss = cLoss + (self.lamda * gradPen2) + (self.alpha * regulariserC)    
            cGradient = tape.gradient(cLoss, self.critic.trainable_variables)
            self.cOptim.apply_gradients(zip(cGradient, self.critic.trainable_variables))
        
        #train generator
        for i in range(self.genSteps):
#            zData   = self.zGenerator.getZdata(latentDims, batchSize=self.batchSize).astype("float32")
            with tf.GradientTape() as tape:
#                generatedSeq = self.generator([realSequence, zData], training=True) #this likely shoulld be chnages to only the first10 rather than the full 30
                generatedSeq = self.generator([first10, zData], training=True)

                generatedSeq = tf.reshape(generatedSeq,[self.batchSize, self.outputLength, (joints*dims)])
                sequence = tf.concat([first10, generatedSeq], axis=1)
                genLogits = self.critic(sequence, training=False)
                
                advLoss, pgLoss, bLoss = self.genLoss(realSequence, generatedSeq, first10, genLogits, self.body_inf)
                generatorLoss = self.alpha*pgLoss + self.beta*bLoss
                gLoss = advLoss + generatorLoss
    
            genGradient = tape.gradient(gLoss, self.generator.trainable_variables)
#            for j in range(15):
#                tf.print(tf.reduce_mean(genGradient[j]))
            self.gOptim.apply_gradients(zip(genGradient, self.generator.trainable_variables))

        #train discriminator
        #create a fake sequence
        with tf.GradientTape() as tape:
            dSequence = tf.concat([first10, generatedSeq], axis=1)
            fakeLogits = self.discriminator(dSequence, training=True)
            realLogits = self.discriminator(realSequence, training=True)

            #calculate the regualriser
            regulariserD = tf.add_n([tf.nn.l2_loss(p) for p in self.discriminator.weights])  
            dLoss = self.discLoss(realLogits, fakeLogits) + (self.alpha * regulariserD)
            
        dGradient = tape.gradient(dLoss, self.discriminator.trainable_variables)
        self.dOptim.apply_gradients(zip(dGradient, self.discriminator.trainable_variables))

        return {"criticLoss":cLoss, "ganLoss":gLoss, "generatorLoss":generatorLoss, "dLoss":dLoss, "gradPen2":gradPen2, "boneLoss":bLoss, 
                "poseConsist":pgLoss, "regulariserC":regulariserC, "regulariserD":regulariserD}

    def test_step(self, data):
        bSize, _, jts, dims = data[0].shape
        realSequence = tf.reshape(data[0], [self.batchSize, self.sequenceLength, (jts*dims)])
#        zData   = self.zGenerator.getZdata(latentDims, batchSize=self.batchSize).astype("float32")
        zData = self.zGenerator(self.z_rand_type, self.z_rand_params, shape=[self.batchSize, self.latentDims])

        first10 = realSequence[:,:self.inputLength,:]
        
#        generatedSeq = self.generator([realSequence, zData], training=False)
        generatedSeq = self.generator([first10, zData], training=False)

        generatedSeq = tf.reshape(generatedSeq,[self.batchSize, self.outputLength, (jts*dims)])
        sequence = tf.concat([first10, generatedSeq], axis=1)
        
        #get the critic losses
        fakeLogits = self.critic(sequence, training=False)
        realLogits = self.critic(realSequence, training=False)
        cLoss = self.criticLoss(realLogits, fakeLogits) 
        gradPen2 = self.gradientPenalty(realSequence, sequence)
        cLoss += (self.lamda * gradPen2)
        
        #generator losses
        advLoss, pgLoss, bLoss = self.genLoss(realSequence, generatedSeq, first10, fakeLogits, self.body_inf)
        generatorLoss = self.alpha*pgLoss + self.beta*bLoss
        gLoss = advLoss + self.alpha*pgLoss + self.beta*bLoss
        
        #discriminator loss
        fakeLogits = self.discriminator(sequence, training=False)
        realLogits = self.discriminator(realSequence, training=False)
        dLoss = self.discLoss(realLogits, fakeLogits)
        
        return {"criticLoss":cLoss, "ganLoss":gLoss, "generatorLoss":generatorLoss, "dLoss":dLoss, "gradPen2":gradPen2, "boneLoss":bLoss, "poseConsist":pgLoss}

    def predict_step(self, data):
#        realSequence = tf.reshape(data[0][0], [self.batchSize, self.sequenceLength, (25*3)])
#        first10 = realSequence[:,:self.inputLength,:]
#        zData = self.zGenerator(self.z_rand_type, self.z_rand_params, shape=[self.batchSize, self.latentDims])
#        zData = tf.reshape(data[0][1], [1, self.latentDims])
        generatedSeq = self.generator([data[0][0], data[0][1]], training=False)
        
        return generatedSeq
        
        
#loss functions
# critic loss
    
def criticLoss(real, pred):

    try:
        tf.debugging.check_numerics(real, "Critic loss - real")
    except Exception as e:
        assert "Checking for NaN " in e.message
        
    try:
        tf.debugging.check_numerics(pred, "Critic loss - pred")
    except Exception as e:
        assert "Checking for NaN " in e.message
    
    xyzLoss = tf.reduce_mean(pred - real) 
    try:
        tf.debugging.check_numerics(xyzLoss, "Critic loss - xyzLoss")
    except Exception as e:
        assert "Checking for NaN " in e.message

#    tf.print(xyzLoss)
    return xyzLoss


#discriminator loss
def discriminatorLoss(real, pred):
    
#        pt1 = real * tf.math.log(pred)
#        pt2 = (1.0-real) * tf.math.log(1.0 - pred)
 
    pt1 = tf.math.log(real)
    pt2 = tf.math.log(1.0 - pred)    
    loss = -tf.reduce_mean(pt1+pt2)
    try:
        tf.debugging.check_numerics(loss, "Discrim Loss")
    except Exception as e:
        assert "Checking for NaN " in e.message

#    tf.print(loss)
    return loss

#generator loss
def generatorLoss(real, pred, past, dxHat, body_inf):

#    predS = tf.reshape(pred, [16,20,75])
#    sequence = tf.concat([past, pred], 1)           
#    dxHat = critic(sequence)
    bSize, _, tdims = real.shape
    jts = int(tdims / 3)
    dims = 3
    
    advLoss = -tf.reduce_mean(dxHat)
 
    #offset the generated poses so the consistancy 
    #last of the seeding poses passed to the generator
    pastOne = past[:,inputSequenceLength-1:inputSequenceLength,:]
    #first 19 of the predictions
    first19 = pred[:,0:output_sequence_length-1,:] 
    gPast = tf.concat([pastOne, first19], 1)        

    #putposly change back to 25 * 3 for the loss function calculations
    pred = tf.reshape(pred, [batchSize, output_sequence_length, jts, 3])
    gPast = tf.reshape(gPast, [batchSize, output_sequence_length, jts, 3])
        
    normie = tf.sqrt(tf.reduce_sum(tf.square(pred - gPast)))
    try:
       tf.debugging.check_numerics(normie, "Normie")
    except Exception as e:
        assert "Checking for NaN " in e.message
    
    pgLoss = tf.maximum(0.0001, normie / (bSize*20))
    
    real20 = real[:,inputSequenceLength:sequenceLength,:] #get the 20 ground truth sequences to the generated sequences
    real20 = tf.reshape(real20, [batchSize, output_sequence_length, jts, 3])
    
 #   bLoss = nn.bone_loss(real20, pred, body_inf) / (16 * 20)  #batchSize and sequence length
    bLoss = nn.bone_loss(gPast, pred, body_inf) / (bSize * 20)  #batchSize and sequence length
    
    try:
        tf.debugging.check_numerics(bLoss, "Bonie")
    except Exception as e:
        assert "Checking for NaN " in e.message

    return advLoss, pgLoss, bLoss

class GANMonitor(keras.callbacks.Callback):
    def __init__(self, printBatch, printPast, printZData, activityIDs, subjectIDs, outFolder, normaliser, hpgan, batchSize=16, logging=logging):
        self.printBatch  = printBatch
        self.printPast   = printPast
        self.printZData  = printZData
        self.activityIDs = activityIDs
        self.subjectIDs  = subjectIDs
        self.outFolder   = outFolder
        self.normaliser  = normaliser
        self.hpgan       = hpgan
        self.batchSize   = batchSize
        self.logging     = logging
        
    def on_epoch_end(self, epoch, logs=None):
        skeletonData, probs = nnet.generateSkeltons(self.printBatch, self.printPast, self.printZData, self.hpgan, self.normaliser, batchIndex=0)
        skeleton2D.draw_to_file(skeletonData, self.subjectIDs[0], os.path.join(self.outFolder, "training_S{}_A{}_epoch{}.png".format(self.subjectIDs[0], self.activityIDs[0], epoch+1)))
        self.logging.info("Epoch:{}".format(epoch))
        self.logging.info("discriminator - training/validation loss: {}/{}".format(logs["dLoss"], logs["val_dLoss"]))
        self.logging.info("gan ----------- training/validation loss: {}/{}".format(logs["ganLoss"], logs["val_ganLoss"]))
        self.logging.info("generator ----- training/validation loss: {}/{}".format(logs["generatorLoss"], logs["val_generatorLoss"]))
        self.logging.info("critic -------- training/validation loss: {}/{}".format(logs["criticLoss"], logs["val_criticLoss"]))
        self.logging.info("bone loss ----- training/validation loss: {}/{}".format(logs["boneLoss"], logs["val_boneLoss"]))
        self.logging.info("consistancy --- training/validation loss: {}/{}".format(logs["poseConsist"], logs["val_poseConsist"]))
        self.logging.info("grad penalty -- training/validation loss: {}/{}".format(logs["gradPen2"], logs["val_gradPen2"]))

        self.logging.info("probabilies per seq epoch_{} - {}".format(epoch+1, probs))        

class SkelGenerator(utils.Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.dataGenerator = x_set 
        self.y = y_set
        self.batchSize = batch_size

    def __len__(self):
        count = self.dataGenerator.getSize()
        batchCount = floor(count / self.batchSize) - 1
        return batchCount

    def __getitem__(self, idx):
        batch, activityIDs, subjectIDs = self.dataGenerator.next_minibatch(self.batchSize)
        return batch, self.y

    def on_epoch_end(self):
        self.dataGenerator.reset()

#2D to 3D uplift model and also the 2D to 3D uplift with prediction model
class D2D3Gan(keras.Model):
    def __init__(self, critic, generator, discriminator, zGenerator, criticSteps=10, genSteps=2, latentDims=128, upliftOnly=False):
        super(D2D3Gan, self).__init__()
        self.critic = critic
        self.generator = generator
        self.discriminator = discriminator  
        self.zGenerator = zGenerator
        self.criticSteps = criticSteps
        self.genSteps = genSteps
        self.latentDims = latentDims
        self.upliftOnly = upliftOnly

    def compile(self, gOptim, cOptim, genLoss, criticLoss):
        super(D2D3Gan, self).compile()
        self.gOptim = gOptim
        self.cOptim = cOptim
        self.genLoss = genLoss
        self.criticLoss = criticLoss
    
    def setLengths(self, batchSize, inputLength, outputLength):
        self.batchSize = batchSize
        self.inputLength = inputLength
        self.outputLength = outputLength
        if self.upliftOnly:
            self.sequenceLength = inputLength
        else:
            self.sequenceLength = inputLength + outputLength

    def setConstants(self, body_inf, lamda=10.0, alpha=0.001, beta=0.01, z_rand_params = {'low':-0.1, 'high':0.1, 'mean':0.0, 'std':0.2}):
        self.body_inf = body_inf
        self.lamda = lamda #applied to gradient penalty
        self.alpha = alpha #applied to regularisers
        self.beta  = beta #applied to bone loss
        #random z data generator parameters
        self.z_rand_type = 'uniform'
        self.z_rand_params = z_rand_params    
        
    #gradient penality as per the paper
    def gradientPenalty2D(self, realSeq, fakeSeq):        
        epilson = tf.random.uniform([self.batchSize, 1, 1], 0.0, 1.0)   
        
        xHat = (epilson*realSeq) + ((1.0-epilson) * fakeSeq)
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(xHat)
            dxHat = self.critic(xHat, training=True)
            
        gradients   = gp_tape.gradient(dxHat, xHat)
        gradientsL2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1,2]))
        gpLoss = tf.reduce_mean(tf.square(gradientsL2 - 1.0))
  #      tf.print(gpLoss)
        
        return gpLoss

    def train_step(self, data):
        
        d3Truth = data
        d2Truth = data[:,:,:,0:2]   
        
        _, _, jts, dims = d3Truth.shape
        
        zData = self.zGenerator(self.z_rand_type, self.z_rand_params, shape=[self.batchSize, self.latentDims])
        
        d2Sequence = tf.reshape(d2Truth, [self.batchSize, self.sequenceLength, (jts*2)]) 
        d2first10 = d2Sequence[:,:self.inputLength,:]

        d3Truth = tf.reshape(d3Truth, [self.batchSize, self.sequenceLength, (jts*3)])
        d3first10 = d3Truth[:,:self.inputLength,:]
        
        #if we are only doing just 2D to 3D uplift, ie no prediction then the ground truth is the first 10 concatenated to the first 10
        if self.upliftOnly:
            d3Truth = tf.concat([d3first10, d3first10], axis=1)
        
        #train the critic
        for i in range(self.criticSteps):
            with tf.GradientTape() as tape:

                fakeSeq = self.generator([d2first10, zData], training=True)
    
                fakeSeq = tf.reshape(fakeSeq,[self.batchSize, self.outputLength, (jts*3)])
                sequence   = tf.concat([d3first10, fakeSeq], axis=1)
                
                fakeLogits = self.critic(sequence, training=True)
                realLogits = self.critic(d3Truth, training=True)
                
                d2_cLoss = self.criticLoss(realLogits, fakeLogits)
#                gradPen = self.gradientPenaltyN(realSequence, sequence)
                gradPen2 = self.gradientPenalty2D(d3Truth, sequence) #both gradient penalty calculations result in the same answer

                #calculate the regualriser
                d2_regulariserC = tf.add_n([tf.nn.l2_loss(p) for p in self.critic.weights])
                d2_cLoss = d2_cLoss + (self.lamda * gradPen2) + (0.00005 * d2_regulariserC) #(self.alpha * d2_regulariserC)    
            d2_cGradient = tape.gradient(d2_cLoss, self.critic.trainable_variables)
            self.cOptim.apply_gradients(zip(d2_cGradient, self.critic.trainable_variables))
        
        #train the generator
        for i in range(self.genSteps):
            with tf.GradientTape() as tape:
                generated = self.generator([d2first10, zData], training=True)
#                generated = self.generator([d2first10], training=True)

#                generated = self.generator([d3first10, zData], training=True)
                
                generated3d = tf.reshape(generated,[self.batchSize, self.outputLength, (jts*3)])      
                sequence3d = tf.concat([d3first10, generated3d], axis=1)
                
                loss = -tf.reduce_mean(self.critic(sequence3d, training=False))
                    
                d2_posLoss, d2_bLoss, d2_poseLoss= self.genLoss(d3Truth, generated, self.body_inf)                
                d2_generatorLoss = self.alpha * d2_posLoss + self.beta* d2_bLoss 
                
                d2_gLoss = loss + d2_generatorLoss
            genGradient = tape.gradient(d2_gLoss, self.generator.trainable_variables)
            
            self.gOptim.apply_gradients(zip(genGradient, self.generator.trainable_variables))
    
        return {"2d_criticLoss":d2_cLoss, "2d_ganLoss":d2_gLoss, "2d_generatorLoss":d2_generatorLoss, "2d_boneLoss":d2_bLoss, "2d_positionLoss":d2_posLoss, "2d_poseLoss":d2_poseLoss, "d2_regulariserC": d2_regulariserC}

    def test_step(self, data):
        
            jts = data[1].shape[2]
            d2Sequence = tf.reshape(data[0], [self.batchSize, self.sequenceLength, (jts*2)])
            d3Sequence = tf.reshape(data[1], [self.batchSize, self.sequenceLength, (jts*3)])
                        
            zData = self.zGenerator(self.z_rand_type, self.z_rand_params, shape=[self.batchSize, self.latentDims])
    
            d2first10 = d2Sequence[:,:self.inputLength,:]
            d3first10 = d3Sequence[:,:self.inputLength,:]
            
            if self.upliftOnly:
                d3Sequence = tf.concat([d3Sequence, d3first10], axis=1)
            
            generatedSeq = self.generator([d2first10, zData], training=False)
#            generatedSeq = self.generator([d2first10], training=False)

#            generatedSeq = self.generator([d3first10, zData], training=False)
    
            generatedSeq3d = tf.reshape(generatedSeq,[self.batchSize, self.outputLength, (jts*3)])
            sequence = tf.concat([d3first10, generatedSeq3d], axis=1)
            
            #get the critic losses
            fakeLogits = self.critic(sequence, training=False)
            realLogits = self.critic(d3Sequence, training=False)
 #           d2_cLoss = -tf.reduce_mean(self.criticLoss(realLogits, fakeLogits)) 
            d2_cLoss = self.criticLoss(realLogits, fakeLogits)
                    
            #generator losses
            d2_posLoss, d2_bLoss, d2_poseLoss = self.genLoss(d3Sequence, generatedSeq, self.body_inf)
            d2_generatorLoss = self.alpha * d2_posLoss + self.beta * d2_bLoss
            d2_gLoss = d2_cLoss + d2_generatorLoss
            
            #discriminator classification
            d2_discValue = 0 #self.discriminator(sequence, training=False)
            d2_discValue = tf.reduce_mean(d2_discValue)
            
            return {"2d_criticLoss":d2_cLoss, "2d_ganLoss":d2_gLoss, "2d_generatorLoss":d2_generatorLoss, "2d_dValue":d2_discValue, "2d_boneLoss":d2_bLoss, "2d_positionLoss":d2_posLoss, "2d_poseLoss":d2_poseLoss}

    def predict_step(self, data):
        generatedSeq = self.generator([data[0][0], data[0][1]], training=False)

        return generatedSeq
    
class d2d3Monitor(keras.callbacks.Callback):
    def __init__(self, d3GroundT, printBatch, printPelvis, printPast, printZData, activityIDs, subjectIDs, outFolder, normaliser3d, normaliser2d, gan, batchSize=16, logging=logging):
        self.d3GroundT   = d3GroundT
        self.printBatch  = printBatch
        self.printPelvis = printPelvis
        self.printPast   = printPast
        self.printZData  = printZData
        self.activityIDs = activityIDs
        self.subjectIDs  = subjectIDs
        self.outFolder   = outFolder
        self.normaliser3d  = normaliser3d
        self.normaliser2d  = normaliser2d
        self.gan         = gan
        self.batchSize   = batchSize
        self.logging     = logging
        
    #on return skeletonData[0] is the 2D gt representation, skeletonData[1] original 3d gt.  skeletonData[2] onwards contains generated sequences
    def on_epoch_end(self, epoch, logs=None):
        skeletonData, probs = nnet.generateSkeltons2D(self.printBatch, self.printPast, self.d3GroundT, self.printZData, self.gan, self.normaliser3d, self.normaliser2d, batchIndex=0)
        skeleton2D.draw2d_to_file([skeletonData[0]], self.subjectIDs[0], os.path.join(self.outFolder, "training_2D_S{}_A{}_epoch{}.png".format(self.subjectIDs[0], self.activityIDs[0], epoch+1)))
        
        #rather than this add the pelvis back to bring into the orginal 3D data space.
        sequence = np.array(skeletonData[2:]) 
        
#        genPelvis = sequence[:,:,0:1]
#        sequence = sequence - genPelvis
        sequence = sequence + self.printPelvis[0]
        
        skeleton2D.draw_to_file(sequence, self.subjectIDs[0], os.path.join(self.outFolder, "training_2D3D_S{}_A{}_epoch{}.png".format(self.subjectIDs[0], self.activityIDs[0], epoch+1)))
        
        skeleton2D.draw_to_file([skeletonData[1]], self.subjectIDs[0], os.path.join(self.outFolder, "training_2D3D_GT_S{}_A{}_epoch{}.png".format(self.subjectIDs[0], self.activityIDs[0], epoch+1)))
        
#        myplt.plot3DSkel(body_inf, skeletonData[2][10], "generated") # when sequence is > 10
#        myplt.plot3DSkel(body_inf, self.d3GroundT[0][10], "generated") # when sequence is > 10
        myplt.plot3DSkel(body_inf, skeletonData[2][0], "generated")
        myplt.plot3DSkel(body_inf, self.d3GroundT[0][0], "generated")


        self.logging.info("Epoch:{}".format(epoch))
        self.logging.info("gan ----------- training/validation loss: {}/{}".format(logs["2d_ganLoss"], logs["val_2d_ganLoss"]))
        self.logging.info("generator ----- training/validation loss: {}/{}".format(logs["2d_generatorLoss"], logs["val_2d_generatorLoss"]))
        self.logging.info("critic -------- training/validation loss: {}/{}".format(logs["2d_criticLoss"], logs["val_2d_criticLoss"]))
        self.logging.info("bone loss ----- training/validation loss: {}/{}".format(logs["2d_boneLoss"], logs["val_2d_boneLoss"]))
        self.logging.info("position loss - training/validation loss: {}/{}".format(logs["2d_positionLoss"], logs["val_2d_positionLoss"]))
        self.logging.info("pose loss ----- training/validation loss: {}/{}".format(logs["2d_poseLoss"], logs["val_2d_poseLoss"]))
        self.logging.info("probabilies per seq epoch_{} - {}".format(epoch+1, probs))        
    
if __name__ == "__main__":

    '''
    Main entry point that drive GAN training for body and skeleton data.

    Args:
        args: arg parser object, contains all arguments provided by the user.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-train",
                        "--train_file",
                        type=str,
                        help="Provide the path of your train file.",
                        required=True)
    parser.add_argument("-test",
                        "--test_file",
                        type=str,
                        help="Provide the path of your test file.")
    parser.add_argument("-ccf",
                        "--camera_calibration_file",
                        type=str,
                        help="Provide the path to your camera calibration file.")
    parser.add_argument("-dnf",
                        "--data_normalization_file",
                        type=str,
                        help="Provide the path to your data normalization file. Which should contain mean and standard deviation of the input dataset.")
    parser.add_argument("-out",
                        "--output_folder",
                        type=str,
                        help="Provide the path of your output folder.",
                        required=True)
    parser.add_argument("-dataset",
                        "--dataset_name",
                        type=str,
                        help="Provide the name of the dataset (nturgbd or human36m).",
                        default="nturgbd")
    parser.add_argument("-epochs",
                        "--max_epochs",
                        type=int,
                        help="Maximum number of epochs (default 300).",
                        default=300)
    parser.add_argument("-clip",
                        "--record_clip", 
                        help="Record a video clip of the predicted action.",
                        action="store_true")
    parser.add_argument("-gantype",
                        "--gan_type",
                        type=str,
                        help="Gan type")
    parser.add_argument("-restore",
                        "--restore",
                        type=bool,
                        default=False)
    
    args = parser.parse_args()

    # setting up paths and log information.
    base_folder = args.output_folder
    output_path = os.path.join(base_folder, 'output')
    output_folder = os.path.join(output_path, "")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    output_models_folder = os.path.join(output_folder, "models")
    if not os.path.exists(output_models_folder):
        os.makedirs(output_models_folder)
    
    output_videos_folder = os.path.join(output_folder, "videos")
    if not os.path.exists(output_videos_folder):
        os.makedirs(output_videos_folder)
    
    output_tensorboard_folder = os.path.join(output_folder, "tensorboard")
    if not os.path.exists(output_tensorboard_folder):
        os.makedirs(output_tensorboard_folder)
    
    output_plots_folder = os.path.join(output_folder, "plots")
    if not os.path.exists(output_plots_folder):
        os.makedirs(output_plots_folder)
        
    output_tests_folder = os.path.join(base_folder, "tests")
    if not os.path.exists(output_tests_folder):
        os.makedirs(output_tests_folder)
        
    output_2D_folder = os.path.join(base_folder, "2D_representation")
    if not os.path.exists(output_2D_folder):
        os.makedirs(output_2D_folder)
        
    dataset = args.dataset_name
    
    #set up the logger
    logging.basicConfig(filename=os.path.join(args.output_folder, "train.log"), filemode='w', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    # Return sensor and body information for the specific dataset.
    source = SourceFactory(dataset, args.camera_calibration_file)
    sensor = source.create_sensor()
    body_inf = source.create_body()

    data_preprocessing = None

    inputSequenceLength = 10
    output_sequence_length = 0 #set this to govern the data sequence length when building data 0 for uplift, 20 for prediction
    sequenceLength = inputSequenceLength + output_sequence_length

    # prepare the data.
    logging.info("Loading data...")
    if args.data_normalization_file is not None:
        data_preprocessing = DataPreprocessing(args.data_normalization_file,
                                               normalization_mode=NormalizationMode.MinAndMax)
    
    if dataset == 'nturgbd':
        logging.info("Loading training data...")
        dataGeneratorTrain = SequenceBodyReader(args.train_file, 
                                               sequenceLength,
                                               dataset,
                                               skip_frame=0,
                                               data_preprocessing=data_preprocessing,
                                               random_sequence=False)
        logging.info("Loading test data...")
        dataGeneratorTest = SequenceBodyReader(args.test_file, 
                                               sequenceLength,
                                               dataset,
                                               skip_frame=0,
                                               data_preprocessing=data_preprocessing,
                                               random_sequence=False)
        
    elif dataset == 'human36m':
        logging.info("Loading training data...")
        dataGeneratorTrain = SequenceBodyReader(args.train_file, 
                                               sequenceLength,
                                               dataset,
                                               skip_frame=0,
                                               data_preprocessing=data_preprocessing,
                                               random_sequence=True)
        
        logging.info("Loading test data...")
        dataGeneratorTest = SequenceBodyReader(args.test_file, 
                                               sequenceLength,
                                               dataset,
                                               skip_frame=0,
                                               data_preprocessing=data_preprocessing,
                                               random_sequence=True)
    else:
        raise ValueError("Invalid dataset value.")   
    
       
    #configure tensorboard
    tbCallBack = keras.callbacks.TensorBoard(log_dir=output_tensorboard_folder, histogram_freq=1, update_freq='batch', write_grads=True)
#    trainData, testingData = readProcessFiles(args)
    
#%% Cell 1

    # both trainData nd testingData contain the full data load from file store in Body objects.
    # this need to be trimmed down in skeleton sequences ready for consumptionby the NN's
    # creat a list of NN ready data that can be consumed by the generator
#    normaliser = dh.NormaliseData(args.data_normalization_file)
    
#    nnReadyTrainingData, nnReadyTestData = buildNNReady(trainData, testingData, args.data_normalization_file, normaliser)  
    
   
    
#%% Cell 2 encoder_inputs
    # define a number of constants
 #   featureShape = (25,3) 
    JOINTCNT, AXISIS = dataGeneratorTrain.element_shape
    
    latentDims   = 128 #size of the z data vector for generator weight adjustment
    numNeuronsG  = 1024 #number of neurons in the RNN layers
    numNeuronsD  = 512 # number of neurons in the critic and discriminator layers
    
    batchSize   = 16
   
    generatorLR = 5e-5 # learning rate
    criticLR    = 5e-5 # learning rate
    discrimLR   = criticLR / 2.0 #learning rate
    
    epochs = args.max_epochs # the number of epochs
    
    decayRate = 0.5 #criticLR / epochs

    
    #training iterations
    discrimITR = 1  # number of iterations per epoch
    generatorITL = 2 # number iterations per epoch
    criticITL = 10    # number iterations per epoch
    
    #define the generator with z effect on the weights
    z_rand_type = 'uniform'
    z_rand_params = {'low':-0.1, 'high':0.1, 'mean':0.0, 'std':0.2}
    zData = nn.generate_random(z_rand_type, z_rand_params, shape=[batchSize, latentDims])
    
    # Return sensor and body information for the specific dataset.
    # ready for drawing skeletons
#    source = SourceFactory(args.dataset_name, None)
#    sensor = source.create_sensor()
#    body_inf = source.create_body()   
    skeleton2D = Skeleton2D(sensor, body_inf)
    
    # Return sensor and body information for the specific dataset.
#    source = SourceFactory(dataset, args.camera_calibration_file)
#    sensor = source.create_sensor()
#    body_inf = source.create_body()

#%% cell 3

    # configure the various NN's, discriminator, critic and generator
    discriminator = nnet.NNDiscriminator((sequenceLength, JOINTCNT*AXISIS))
    discriminator.summary()

    criticInShape = (sequenceLength, JOINTCNT*AXISIS)
    critic = nnet.NNCritic(criticInShape)
    critic.summary()
    
#    generator2 = nnet.RNNGenerator3(1500, (25 * 3), zD)
    generator2 = nnet.RNNGenerator7(numNeuronsG, (inputSequenceLength, JOINTCNT*AXISIS), (latentDims))
    generator2.summary()
     
    #optimisers 
    dOptimiser=keras.optimizers.Adam(learning_rate=discrimLR, beta_1=decayRate)
    cOptimiser=keras.optimizers.Adam(learning_rate=criticLR, beta_1=decayRate)
    gOptimiser=keras.optimizers.Adam(learning_rate=generatorLR, beta_1=decayRate)
    
    #create the HP-GAN model
    hpgan = HPGAN(critic, generator2, discriminator, nn.generate_random, criticSteps=10, genSteps=1, latentDims=latentDims)
    hpgan.setLengths(batchSize, inputSequenceLength, output_sequence_length)
    hpgan.setConstants(body_inf)
    hpgan.compile(cOptimiser, gOptimiser, dOptimiser, criticLoss, generatorLoss, discriminatorLoss)
    
    # setup the z data for generting and printing on training skeletons, use the same z data for this purpose
    printZData = nn.generate_random(z_rand_type, z_rand_params, shape=[11, latentDims])
    skeletonData = []
    #print skeletons generated for this epoch to file, only require 11 batch for printing
    dataGeneratorTrain.reset()   
    
    trainRecCount = dataGeneratorTrain.size()# - (dataGeneratorTrain.size() % batchSize) #number of records to retrive the full batchsize aligned amount
#    testRecCount  = int(dataGeneratorTest.size() / 2) - (int(dataGeneratorTest.size() / 2)  % batchSize)
 
#    testRecCount  = dataGeneratorTest.size() - (dataGeneratorTest.size() % batchSize)
    testRecCount  = dataGeneratorTest.size()
   
    inputBatch, targets, rntBatchSize, activityIDs, subjectIDs = dataGeneratorTrain.next_minibatch(trainRecCount) #15296
#    validateBatch, valTargets, valRntBatchSize, valActivityIDs, valSubjectIDs = dataGeneratorTest.next_minibatch(testRecCount) #1904
    testBatch, testTargets, testRntBatchSize, testActivityIDs, testSubjectIDs = dataGeneratorTest.next_minibatch(testRecCount) #1904
    
    trainCnt = 0 # default for nturgbd
    testCnt  = 0
    if dataset == 'human36m': # need to sample many times as the dataset is small and sequences long
        trainCnt = 350 #for 10x10  
        testCnt  = 45 #45 for 10x10
    
    for i in range(trainCnt):
        dataGeneratorTrain.reset()   
        inputBatch2, targets2, rntBatchSize2, activityIDs2, subjectIDs2 = dataGeneratorTrain.next_minibatch(trainRecCount)
        inputBatch  = np.concatenate([inputBatch, inputBatch2])
        activityIDs = np.concatenate([activityIDs, activityIDs2])
        subjectIDs  = np.concatenate([subjectIDs, subjectIDs2])
    
    for i in range(testCnt):
        dataGeneratorTest.reset()
        testBatch2, testTargets2, testRntBatchSize2, testActivityIDs2, testSubjectIDs2 = dataGeneratorTest.next_minibatch(testRecCount)    
        testBatch       = np.concatenate([testBatch, testBatch2])
        testActivityIDs = np.concatenate([testActivityIDs, testActivityIDs2])
        testSubjectIDs  = np.concatenate([testSubjectIDs, testSubjectIDs2])
        
    # make testBatch into batchsize segments
    #split the test data in two for test and validation
    even = len(testBatch) - (len(testBatch) % 2) #trim to fit the batchsize
    testBatch = testBatch[:even]
    testActivityIDs = testActivityIDs[:even]
    testSubjectIDs  = testSubjectIDs[:even]
    
    testBatch, validateBatch = np.split(testBatch,2)
    testActivityIDs, valActivityIDs = np.split(testActivityIDs, 2)
    testSubjectIDs,  valSubjectIDs  = np.split(testSubjectIDs, 2)
    
    rmd = len(testBatch) % batchSize  #trim to fit the batchsize
    testBatch       = testBatch[:-rmd]
    testActivityIDs = testActivityIDs[:-rmd]
    testSubjectIDs  = testSubjectIDs[:-rmd]
    
    validateBatch  = validateBatch[:-rmd]
    valActivityIDs = valActivityIDs[:-rmd]
    valSubjectIDs  = valSubjectIDs[:-rmd]
    
    rmd = len(inputBatch) % batchSize
    inputBatch  = inputBatch[:-rmd]
    activityIDs = activityIDs[:-rmd]
    subjectIDs  = subjectIDs[:-rmd]

    printBatch = inputBatch[0:11,:,:]
    printActIDs = activityIDs[0:11]
    printSubIDs = subjectIDs[0:11]
    printPast  = printBatch[:, 0:inputSequenceLength, :, :]  #driver generator
    
    # with my normalisation method
#    monitor = GANMonitor(printBatch, printPast, printZData, printActIDs, printSubIDs, args.output_folder, train3Dstats, hpgan, batchSize=batchSize, logging=logging)
    monitor = GANMonitor(printBatch, printPast, printZData, printActIDs, printSubIDs, args.output_folder, data_preprocessing, hpgan, batchSize=batchSize, logging=logging)

#    skeletonData, probs = nnet.generateSkeltons(printBatch, printPast, printZData, hpgan, train3Dstats, batchIndex=0) #my normalisation
#    skeletonData, probs = nnet.generateSkeltons(printBatch, printPast, printZData, hpgan, data_preprocessing, batchIndex=0)

#    skeleton2D.draw_to_file(skeletonData, printSubIDs[0], os.path.join(args.output_folder, "training_S{}_A{}_epoch{}.png".format(printSubIDs[0], printActIDs[0], -1)))    
    dataGeneratorTrain.reset()        
    dataGeneratorTest.reset()
    y1 = np.ones((len(validateBatch), 1), dtype=np.int32)
    
#    history = hpgan.fit(inputBatch, validation_data=(validateBatch, y1), batch_size=batchSize, epochs=1, callbacks=[monitor, tbCallBack], shuffle=True)
       
#    cLoss = history.history["criticLoss"]
#    gLoss = history.history["ganLoss"]
#    dLoss = history.history["dLoss"]
#    myplt.trainingPlotter2(history.history, epochs, save=True, show=True)
    
    print("all Done Training")   
    

#%% Cell 4
    #ecludian distance and MMD metrics and generate associated test samples.
    
    def tests(testData, testSubjects, testActivity, gan, normaliser3d, normaliser2d=None, d3GroundT=None, pelvisData=None):
    
        def eDistance(generated, groundTruth):
            poses, joints, pts = generated.shape
            
            jDists = []
            for i in range(poses):
                 poseDist = np.square(generated[i] - groundTruth[i])
                 poseDist = np.sum(poseDist)
                 jDists.append(poseDist)
            
            return jDists
        
        #maximum mean difference
        def mmdRBF(real, generated, gamma=1.0):
            poses, joints, pts = generated.shape
            
            mmd = []
            for i in range(poses):         
                RR = metrics.pairwise.rbf_kernel(real[i], real[i], gamma)
                GG = metrics.pairwise.rbf_kernel(generated[i], generated[i], gamma)
                RG = metrics.pairwise.rbf_kernel(real[i], generated[i], gamma)
                dist = RR.mean() + GG.mean() - 2*RG.mean()
                mmd.append(dist)
                
            return mmd        
        
        #testing of generate skeletons
        testNumber = 3
        zData = nn.generate_random(z_rand_type, z_rand_params, shape=[testNumber, latentDims])
    
        #track and store the MMD, ecluid and dist values for each activity for a table in the report
        numActions = 50
        lst = []
        trackEuclid = np.empty(numActions, dtype=object)
        trackMMD    = np.empty(numActions, dtype=object)
        trackDisc   = np.empty(numActions, dtype=object)
        for i in range(numActions):
            trackEuclid[i] = []
            trackMMD[i]    = []
            trackDisc[i]   = []
    
        totalEculDist = []
        totalMMD = []
            
        testPast   = testData[:, 0:inputSequenceLength, :, :]  #driver sequence
        testPast2d = testData[:, 0:inputSequenceLength, :, 0:2]  #driver sequence

        for i in range(len(testBatch)):
            second20 = None
            genatd20 = None
            #switch here on the test set, it HPGAN or D2D3 GAN
            if type(gan).__name__ == "HPGAN":
                
                skeletonData, probs = nnet.generateSkeltons2(testData[i], testPast[i], zData, gan, normaliser3d, batchIndex=0)
                skeleton2D.draw_to_file(skeletonData, testSubjects[i], os.path.join(args.output_folder, "tests/testgen{}_S{}_A{}.png".format(i+1, testSubjects[i], testActivity[i])))
            #first line of skeleton data is the ground truth, the next 10 lines are prediction from different z noise. (first 10 poses are driver sequence)
            #compute the eucludian distance and mmd on the first generated sequence only
                second20 = skeletonData[0][inputSequenceLength:sequenceLength, :,:] #ground truth
                genatd20 = skeletonData[1][inputSequenceLength:sequenceLength, :,:] #generated sequence
                
            elif type(gan).__name__ == "D2D3Gan":                
                skeletonData, probs = nnet.generateSkeltons2D(testData[i:i+1,:,:,:], testPast2d[i:i+1,:,:,:], d3GroundT[i:i+1,:,:,:], zData, gan, normaliser3d, normaliser2d, batchIndex=0)
                skeleton2D.draw2d_to_file([skeletonData[0]], testSubjects[i], os.path.join(args.output_folder, "tests/testgen_2D_S{}_A{}_epoch{}.png".format(testSubjects[i], testActivity[i], i)))
                #rather than this add the pelvis back to bring into the orginal 3D data space.
                sequence = skeletonData[2:] + pelvisData[i]
                skeleton2D.draw_to_file(sequence, testSubjects[i], os.path.join(args.output_folder, "tests/testgen_2D3D_S{}_A{}_epoch{}.png".format(testSubjects[i], testActivity[i], i)))
                skeleton2D.draw_to_file([skeletonData[1]], testSubjects[i], os.path.join(args.output_folder, "tests/testgen_2D3D_GT_S{}_A{}_epoch{}.png".format(testSubjects[i], testActivity[i], i)))
                
 #               skeleton2D.draw_to_video_file(sequence[0], os.path.join(args.output_folder, "tests/testVideo_2D3D_GT_S{}_A{}_epoch{}.mp4".format(testSubjects[i], testActivity[i], i)))            
                second20 = skeletonData[1][inputSequenceLength:sequenceLength, :,:] #ground truth
                genatd20 = skeletonData[2][inputSequenceLength:sequenceLength, :,:] #generated sequence
                
                
            jointDists = eDistance(genatd20, second20)   
            seqMMD = mmdRBF(second20, genatd20)
                
            totalEculDist.append(jointDists)
            totalMMD.append(seqMMD)
            
            fmtProbs = " " + " ".join(["{:.4f}"]*len(probs))
            fmtMetrc = " " + " ".join(["{:.3f}"]*len(jointDists))
    
            logging.info("Test batch index_{}".format(i))
            logging.info("Test probabilies for generated skels_S{}_A{}- {}".format(testSubjects[i], testActivity[i], fmtProbs.format(*probs)))
            logging.info("Pose euclidian distance for each pose_S{}_A{}- {}".format(testSubjects[i], testActivity[i], fmtMetrc.format(*jointDists)))
            logging.info("Mean maximum distance_S{}_A{}- {}".format(testSubjects[i], testActivity[i], fmtMetrc.format(*seqMMD)))
            logging.info("Average euclidian distance for_S{}_A{}- {:.3f}".format(testSubjects[i], testActivity[i], np.mean(jointDists)))
            logging.info("Average mmd_S{}_A{}- {:.3f}".format(testSubjects[i], testActivity[i], np.mean(seqMMD)))
            logging.info("")
            
            trackEuclid[testActivity[i]].append(np.mean(jointDists))
            trackMMD[testActivity[i]].append(np.mean(seqMMD))
            trackDisc[testActivity[i]].append(np.mean(probs))
            
        logging.info("Total Test Average Euclidian Distance = {:.4}".format(np.mean(totalEculDist)))
        logging.info("Total Test Average MMD Distance = {:.4}".format(np.mean(totalMMD)))
    
        logging.info("Summay for report table")
        for i in range(1, numActions):
            if len(trackEuclid[i]) > 0:
                logging.info("Action - {}".format(i))
                logging.info("Euclidean for --action_{}, mean_{:.3f}, max_{:.3f} min_{:.3f}".format(i, np.mean(trackEuclid[i]), np.amax(trackEuclid[i]), np.amin(trackEuclid[i])))
                logging.info("MMD for --------action_{}, mean_{:.3f}, max_{:.3f} min_{:.3f}".format(i, np.mean(trackMMD[i]), np.amax(trackMMD[i]), np.amin(trackMMD[i])))
                logging.info("Discrimator for action_{}, mean_{:.3f}, max_{:.3f} min_{:.3f}".format(i, np.mean(trackDisc[i]), np.amax(trackDisc[i]), np.amin(trackDisc[i])))
    
        print("All Done - Test images")

#    tests(testBatch, testSubjectIDs, testActivityIDs, hpgan, data_preprocessing)    

#   sys.exit()

    
#%% Cell plot the joint positions for a number of real and generated joints
    
    def positionMetrics(inputData, gan, subjectID, activityID, normaliser):

        leftHand  = JointType(7)
        rightHand = JointType(11)
        leftFoot  = JointType(15)
        rightFoot = JointType(19)
        head      = JointType(3)
        pelvis    = JointType(0)
        
        indexs = [63]
        zData = nn.generate_random(z_rand_type, z_rand_params, shape=[1, latentDims])
        
        joints = ["leftHand", "rightHand", "leftFoot", "rightFoot", "head", "pelvis"]
        titles = ["Left Hand", "Right Hand", "Left Foot", "Right Foot", "Head", "Pelvis"]
    
        for i in range(0, len(testBatch)): #indexs:
            sequence = inputData[i]
            
            #if HPGAN first10 is 25,3 if 2D3DGan 25,2
            first10 = None
            if type(gan).__name__ == "HPGAN":
                first10 = sequence[0:inputSequenceLength,:,:]
            elif type(gan).__name__ == "D2D3Gan":
                first10 = sequence[0:inputSequenceLength,:,0:2]
            a, b, c = first10.shape
            first10 = np.reshape(first10, (1,a,b*c))
            prediction = gan.predict([first10, zData])
    
            #collect z, y, z positions for the specified body parts from the ground truth
            sequence = normaliser.unnormalize(sequence)
            lfh = sequence[inputSequenceLength:sequenceLength, leftHand.value:leftHand.value+1:]
            rgh = sequence[inputSequenceLength:sequenceLength, rightHand.value:rightHand.value+1:]
        
            lff = sequence[inputSequenceLength:sequenceLength, leftFoot.value:leftFoot.value+1:]
            rgf = sequence[inputSequenceLength:sequenceLength, rightFoot.value:rightFoot.value+1:]
            
            hd   = sequence[inputSequenceLength:sequenceLength, head.value:head.value+1:]
            plvs = sequence[inputSequenceLength:sequenceLength, pelvis.value:pelvis.value+1:]
             
            realSeq = {
                "leftHand": lfh,
                "rightHand": rgh,
                "leftFoot": lff,
                "rightFoot": rgf,
                "head": hd,
                "pelvis": plvs
            }
            
            #collect z, y, z positions for the specified body parts from the generated sequence
            a, b, c, d = prediction.shape
            prediction = normaliser.unnormalize(prediction)
            prediction = np.reshape(prediction, (b ,c, d))
    
            lfhP = prediction[:, leftHand.value:leftHand.value+1:]
            rghP = prediction[:, rightHand.value:rightHand.value+1:]
        
            lffP = prediction[:, leftFoot.value:leftFoot.value+1:]
            rgfP = prediction[:, rightFoot.value:rightFoot.value+1:]
            
            hdP   = prediction[:, head.value:head.value+1:]
            plvsP = prediction[:, pelvis.value:pelvis.value+1:]
            
            fakeSeq = {
                "leftHand": lfhP,
                "rightHand": rghP,
                "leftFoot": lffP,
                "rightFoot": rgfP,
                "head": hdP,
                "pelvis": plvsP
            }
            for idx, title in zip(joints, titles):
                title = title + "_S{}_A{}_Track".format(subjectID[i], activityID[i])
                myplt.plotSequence(realSeq, fakeSeq, idx, title)
    
#    positionMetrics(testBatch, hpgan, testSubjectIDs, testActivityIDs, data_preprocessing)
    sys.exit()

#%% 3dplot

    firstPose = testBatch[0]
    first10 = firstPose[0:inputSequenceLength,:,:]
    firstPose = firstPose[inputSequenceLength:sequenceLength, :, :]
    
    a, b, c = first10.shape
    first10 = np.reshape(first10, (1,a,b*c))
    prediction = hpgan.predict([first10, zData])
    
    firstPose = data_preprocessing.unnormalize(firstPose)
    prediction = data_preprocessing.unnormalize(prediction)

    myplt.plot3DPose(firstPose, prediction[0], "3D")

#%% Cell 5

# evaluate the model

#    dataGeneratorTest.reset()
#    inputBatch, activityIDs, subjectIDs = dataGeneratorTest.next_minibatch(batchSize=3824) #3824
#    create labeling for the discriminator 
    y1 = np.ones((testRecCount, 1), dtype=np.int32)
    y0 = np.zeros((batchSize, 1), dtype=np.int32)
    y  = np.vstack((y1, y0))
    eHistory = hpgan.evaluate(validateBatch, y1, batch_size=batchSize, verbose=1, callbacks=[tbCallBack])
    
    y1 = np.ones((testRecCount, 1), dtype=np.int32)
    y0 = np.zeros((batchSize, 1), dtype=np.int32)
    y  = np.vstack((y1, y0))
    eHistory = hpgan.evaluate(testBatch, y1, batch_size=batchSize, verbose=1, callbacks=[tbCallBack])
    
    #put the test data through and see what discriminator returns
    dataGeneratorTest.reset()
    dataGeneratorTest.next_minibatch(testRecCount)
    inputBatch, targets, rntBatchSize, activityIDsT, subjectIDsT = dataGeneratorTest.next_minibatch(11)
    ib = np.reshape(inputBatch, (11,30,75))
    prob = hpgan.discriminator(ib, training=False)
    logging.info("Test probabilies for test sequences - {}".format(prob))
    

#%% Cell 6

#save the trained model, save needing to build it again.
   
#
#    hpgan.generator.save_weights("./results/output/models/generator.tf", save_format="tf")
#    hpgan.critic.save("results/output/models/critic.tf")
#    hpgan.discriminator.save("results/output/models/discrim.tf")
    
#%% Cell 2dTO3D

    sample = inputBatch[0:5,:,:,:]
    subID  = subjectIDs[0:5]
    sample = data_preprocessing.unnormalize(sample)
    skeleton2D.draw_to_file(sample, subID, "./results/2D_representation/2d_{}.png".format(subID[0]))
    
    sample2d = sample[:,:,:,0:2]
    skeleton2D.draw2d_to_file(sample, subID, "./results/2D_representation/3d2d_{}.png".format(subID[0]))  
    
#%% Cell Construct the 2d to 3d model GAN

    #cacluation of mean per joint position error for a set of sequences. # this maybe need to unnormalise the data for mm distances
    
    #also maybe necessary to add the pelvis back to take back for comparsion with orginal ground truth.
    #protocol 1
    def mpjpeP1(gTruth, prediction):
        
        error = np.mean(np.linalg.norm(prediction-gTruth, axis=len(gTruth.shape)-1), axis=len(gTruth.shape)-2)
        return error
    
    #protocol 2
    # code from https://github.com/Vegetebird/MHFormer
    def mpjpeP2(gTruth, prediction):
        
        ax0 = len(gTruth.shape)-3 # 0 when using a single of shape of 20,25,3
        ax1 = len(gTruth.shape)-2 # 1
        ax2 = len(gTruth.shape)-1 # 2
        
        muX = np.mean(gTruth, axis=ax1, keepdims=True)     # 20,1,3
        muY = np.mean(prediction, axis=ax1, keepdims=True) # 20,1,3
        
        X0 = gTruth - muX      # 20,25,3
        Y0 = prediction - muY  # 20,25,3
        
        normX = np.sqrt(np.sum(X0 ** 2, axis=(ax1,ax2), keepdims=True)) # 20,1,1
        normY = np.sqrt(np.sum(Y0 ** 2, axis=(ax1,ax2), keepdims=True)) # 20,1,1
                        
        X0 /= normX # 20,25,3
        Y0 /= normY # 20,25,3
        
        H = np.matmul(X0.transpose(ax0,ax2,ax1), Y0) # 20,3,3
        U, s, Vt = np.linalg.svd(H)  # 20,3,3    20,3    20,3,3
        V = Vt.transpose(ax0,ax2,ax1) #20,3,3
        R = np.matmul(V, U.transpose(ax0,ax2,ax1)) #20,3,3
        
        sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=ax1)) #20,1
        V[:,:,-1] *= sign_detR  #20,3 * 20,1 = 20,3,3
#        s[:,:,-1] *= sign_detR[:,:,-1] #sign_detR.flatten() #20, 3
        s[:,-1] *= sign_detR.flatten() #20, 3
        R = np.matmul(V, U.transpose(ax0,ax2,ax1)) #20,3,3
        
        tr = np.expand_dims(np.sum(s, axis=ax1, keepdims=True), axis=ax2) #20,1,1
        
        a = tr * normX / normY   #20,1,1
        t = muX - a * np.matmul(muY, R) #20,1,3
        
        predicted_aligned = a * np.matmul(prediction, R) + t #20,25,3
        
#        error = np.mean(np.linalg.norm(predicted_aligned - gTruth, axis=ax2)) #result for the sequence
        error = np.mean(np.linalg.norm(predicted_aligned - gTruth, axis=len(gTruth.shape) - 1), axis=len(gTruth.shape) - 2) #result per pose
 
        return error
        
    def mpjperror(gTruth,  gan, normaliser, inLength, dims, single=False, pelvis=None): # single = True for when using driver sequence as ground truth. 
        seqCnt, seqLen, jts, _ = gTruth.shape
        driver = gTruth[:,:inLength,:,0:dims]
        driver = np.reshape(driver, (seqCnt, inLength, jts * dims))
        
        z = nn.generate_random(z_rand_type, z_rand_params, shape=[seqCnt, latentDims]) #get a  noise vectors 
        prediction = gan.predict([driver, z])

        real = gTruth[:,:inLength,:,:]  #actually real 10, the first 10 frames
        if not single:
            real = gTruth[:,inLength:,:,:]  #real 20, the 20 future frames
        
        prediction = normaliser.unnormalize(prediction)
        real = normaliser.unnormalize(real)
        
        #if pelvis data supplied add the pelvis back before calculations
        if pelvis is not None:
            prediction += pelvis
            real += pelvis
            
        errorP1 = mpjpeP1(real, prediction)
        errorP2 = []
        
        for r, p in zip(real, prediction):
            errP = mpjpeP2(r, p)
            errorP2.append(errP)
        return errorP1, np.array(errorP2)

    # this version accepts orginal data with not pelvis centering.
    def mpjperrorOriginal(gTruth, orginalData, gan, normaliser, inLength, dims, single=False, pelvis=None, protocol="one"): # single = True for when using driver sequence as ground truth. 
        seqCnt, seqLen, jts, _ = gTruth.shape
        driver = gTruth[:,:inLength,:,0:dims]
        driver = np.reshape(driver, (seqCnt, inLength, jts * dims))
        
        z = nn.generate_random(z_rand_type, z_rand_params, shape=[seqCnt, latentDims]) #get a  noise vectors 
        prediction = gan.predict([driver, z])
    
        real = orginalData[:,:inLength,:,:]  #actually real 10, the first 10 frames
        if not single:
            real = orginalData[:,inLength:,:,:]  #real 20, the 20 future frames
        
        prediction = normaliser.unnormalise(prediction)
        
        #if pelvis data supplied add the pelvis back before calculations
        if pelvis is not None:
            prediction += pelvis
        
        if protocol == "one":
            error = mpjpeP1(real, prediction)
        else:
            error = mpjpeP2(real, prediction)
                
        return error
    
    #generate some noise at 5% for adding to skeletons to test ability to handle noise input like video
    def getNoise(size=1):
        noise = np.random.uniform(-0.1, 0.1, size=size)
        return noise
        
    #calculate the 2d representation of the 3D pose data
    # This unnnormalises the 3D representation
    # Centers on the pelvis
    # Computes the statistics on the recentred on the pelvis data
    # normalise pelvis centered data
    
    inputSequenceLength = 10
    output_sequence_length = 20
    sequenceLength = inputSequenceLength + output_sequence_length
    latentDims = 128
    
    #training, validation and test data, unnormalise and recentre on the pelvis
#    dataGeneratorTrain.reset()    
#    inputBatch, targets, rntBatchSize, activityIDs, subjectIDs = dataGeneratorTrain.next_minibatch(15296)   #15296
    #unnormalise
    normalInBatch = data_preprocessing.unnormalize(inputBatch)   
    normalValBtch = data_preprocessing.unnormalize(validateBatch)
    normalTestBtch = data_preprocessing.unnormalize(testBatch)

    # recenter all batches on the pelvis
    pelvis = JointType(0)

    pelvisPosTT  = normalInBatch[:,:,pelvis.value:pelvis.value+1]
    pelvisTrain3D = normalInBatch - pelvisPosTT
    #change the pelvis 0.0 value to a small value, to enable learning on 
    pelvisTrain3D = np.where(pelvisTrain3D == 0.0, 0.0001, pelvisTrain3D)
    pelvis2Dtrain = pelvisTrain3D[:,:,:,0:2]
    
    pelvisPos = normalValBtch[:,:,pelvis.value:pelvis.value+1]
    pelvisVal3D = normalValBtch - pelvisPos
    #change the pelvis 0.0 value to a small value, to enable learning on 
    pelvisVal3D = np.where(pelvisVal3D == 0.0, 0.0001, pelvisVal3D)
    pelvisVal2D = pelvisVal3D[:,:,:,0:2]
    
    pelvisPos = normalTestBtch[:,:,pelvis.value:pelvis.value+1]
    pelvisTest3D = normalTestBtch - pelvisPos
    #change the pelvis 0.0 value to a small value, to enable learning on 
    pelvisTest3D = np.where(pelvisTest3D == 0.0, 0.0001, pelvisTest3D)
    pelvisTest2D = pelvisTest3D[:,:,:,0:2]
   
    #compute stats for the full dataset
    d2stats = nnet.Stats2D(np.concatenate((pelvisTrain3D, pelvisVal3D, pelvisTest3D)), meanType="global_linear")
    
 #   skeleton2D.draw_to_file(normalInBatch[0:5,:,:,:], subjectIDs[0], os.path.join(args.output_folder, "2D_representation/train_3D_GT{}_S{}_A{}.png".format(0, subjectIDs[0:5], activityIDs[0:5])))
 #   skeleton2D.draw2d_to_file(pelvisTrain3D[0:5,:,:,0:2], subjectIDs[0], os.path.join(args.output_folder, "2D_representation/train_2D_GT{}_S{}_A{}.png".format(0, subjectIDs[0:5], activityIDs[0:5])))
    myplt.plot2DSkel(body_inf, pelvisTrain3D[0][0])    
    myplt.plot3DSkel(body_inf, pelvisTrain3D[0][0], "Original 3D Pose")
    
    #normalise
    pelvis3Dtrain_n = d2stats.normalise(pelvisTrain3D)
    pelvis2Dtrain_n = pelvis2Dtrain[:,:,:,0:2]
    
    pelvisVal3D_n = d2stats.normalise(pelvisVal3D)
    pelvisVal2D_n = pelvisVal3D_n[:,:,:,0:2]
            
    pelvisTest3D_n = d2stats.normalise(pelvisTest3D)
    pelvisTest2D_n = pelvisTest3D_n[:,:,:,0:2]
    
    y1 = np.ones((len(validateBatch), 1), dtype=np.int32)
    
 #   skeleton2D.draw_to_file(normalValBtch[0:5,:,:,:], valSubjectIDs[0], os.path.join(args.output_folder, "2D_representation/val_3D_GT{}_S{}_A{}.png".format(0, valSubjectIDs[0:5], valActivityIDs[0:5])))
 #   skeleton2D.draw2d_to_file(pelvisVal3D[0:5,:,:,0:2], valSubjectIDs[0], os.path.join(args.output_folder, "2D_representation/val_2D_GT{}_S{}_A{}.png".format(0, valSubjectIDs[0:5], valActivityIDs[0:5])))

 #   skeleton2D.draw_to_file(normalTestBtch[0:5,:,:,:], testSubjectIDs[0], os.path.join(args.output_folder, "2D_representation/test_3D_GT{}_S{}_A{}.png".format(0, testSubjectIDs[0:5], testActivityIDs[0:5])))
 #   skeleton2D.draw2d_to_file(pelvisTest3D[0:5,:,:,0:2], testSubjectIDs[0], os.path.join(args.output_folder, "2D_representation/test_2D_GT{}_S{}_A{}.png".format(0, testSubjectIDs[0:5], testActivityIDs[0:5])))
    
    #free up memory
    del normalInBatch
    del normalValBtch
    del normalTestBtch
    
    # generator loss function for the d2 to d3 generator
    def d2d3generatorLoss(real, pred, body_inf):
                
#            advLoss = -tf.reduce_mean(dxHat)
            jts = pred.shape[2]

            real20 = real[:,inputSequenceLength:sequenceLength,:] #get the 20 ground truth sequences
            real20 = tf.reshape(real20, [batchSize, output_sequence_length, jts, 3])
            
            pred = tf.reshape(pred, [batchSize, output_sequence_length, jts, 3])
            
            #construct pose offset array for pose consistancy calculation
            firstOne = real[:,inputSequenceLength-1:inputSequenceLength,:]
            firstOne = tf.reshape(firstOne, [batchSize, 1 , jts, 3])
            last19   = pred[:,0:-1:,:]
            offsetSeq = tf.concat([firstOne, last19], axis=1)
            poseLoss = tf.sqrt(tf.reduce_sum(tf.square(pred-offsetSeq)))    
            poseLoss = tf.maximum(0.0001, poseLoss) / (16*20)
         
    #        posLoss = tf.reduce_sum(tf.square(real20-pred20)) / (16 * 20)
    #        posLoss = tf.reduce_mean(tf.square(real20-pred20)) # MSE
            posLoss = tf.sqrt(tf.reduce_sum(tf.square(real20-pred)))    #l2 norm
            posLoss = tf.maximum(0.0001, posLoss) / (16*20)
            
            bLoss = nn.bone_loss(real20, pred, body_inf) / (16 * 20)  #batchSize and sequence length
    #        bLoss = nn.bone_loss(offsetSeq, pred, body_inf) / (16 * 20)  #batchSize and sequence length
            
            try:
                tf.debugging.check_numerics(bLoss, "Bonie")
            except Exception as e:
                assert "Checking for NaN " in e.message

            return posLoss, bLoss, poseLoss
    
    critic2d3d = nnet.NNCritic((sequenceLength, JOINTCNT*AXISIS))   
    critic2d3d.summary()

    d2d3Generator = nnet.D2TOD3Generator(1024, (inputSequenceLength, JOINTCNT*(AXISIS-1)), (latentDims))
#    d2d3Generator = nnet.RNNGenerator7(1024, (inputSequenceLength, 25*2), (latentDims))

    d2d3Generator.summary()    
    
    d2d3GAN = D2D3Gan(critic2d3d, d2d3Generator, hpgan.discriminator, nn.generate_random, criticSteps=10, genSteps=2, latentDims=latentDims) 
#    d2d3GAN = D2D3Gan(hpgan.critic, hpgan.generator, hpgan.discriminator, nn.generate_random, criticSteps=10, genSteps=2, latentDims=latentDims) 

    d2d3GAN.setLengths(batchSize, inputSequenceLength, output_sequence_length)
    d2d3GAN.setConstants(body_inf)
    
    d2LearnRate = 5e-5 # learning rate
    d2d3G_Optimiser = keras.optimizers.Adam(learning_rate=d2LearnRate, beta_1=decayRate)
    d2d3C_Optimiser = keras.optimizers.Adam(learning_rate=criticLR, beta_1=decayRate)
    d2d3GAN.compile(d2d3G_Optimiser, d2d3G_Optimiser, d2d3generatorLoss, criticLoss)
    
    #setup here the end of epoch print of the training status,  ## change to training data for production
    printZData = nn.generate_random(z_rand_type, z_rand_params, shape=[11, latentDims]) #get a new noise vector for printing
    printBatch = pelvis3Dtrain_n[0:11,:,:]  #3D ground truth for the full sequence, pelvis centered for 2D representation and normalised
    printPelvis = pelvisPosTT[0:11,:,:]
    printActIDs = activityIDs[0:11]
    printSubIDs = subjectIDs[0:11]
    printPast  = printBatch[:, 0:inputSequenceLength, :, 0:2]  #driver sequence

    d3GroundT = inputBatch[0:11,:,:,:] #3D ground truth of 3D data, for sequence image generation only
    
    d2d3Mon = d2d3Monitor(d3GroundT, printBatch, printPelvis, printPast, printZData, printActIDs, printSubIDs, output_2D_folder, data_preprocessing, d2stats, d2d3GAN, batchSize=batchSize, logging=logging)

#    skeletonData, probs = nnet.generateSkeltons2D(printBatch, printPast, d3GroundT, printZData, d2d3GAN, data_preprocessing, d2stats, batchIndex=0)

#    history = d2d3GAN.fit(pelvis3Dtrain_n, validation_data=(pelvisVal2D_n, pelvisVal3D_n, y1), batch_size=batchSize, epochs=5, callbacks=[d2d3Mon,tbCallBack], shuffle=True)
     
    #plot training statistics
#    myplt.trainingPlotter2_2D(history.history, epochs, save=True, show=True)
    
    
#%% Cell 2d3d results
    #run tests, eculidian distances and MMD and print test generations
    logging.info("--------- Starting 10->20 testing, noise added to states ------------")

    tests(pelvisTest3D_n, testSubjectIDs, testActivityIDs, d2d3GAN, data_preprocessing, normaliser2d=d2stats, d3GroundT=testBatch, pelvisData=pelvisPos)
    positionMetrics(pelvisTest3D_n, d2d3GAN, testSubjectIDs, testActivityIDs, d2stats)
    
    # calculate mpjpe
    errorP1, errorP2 = mpjperror(pelvisTest3D_n, d2d3GAN, d2stats, inputSequenceLength, 2) #for 2d to 3d lift
    logging.info("MPJPE error: {}".format(errorP1))

    logging.info("--------- Completed 10->20 testing, noise added to states ------------")
    
#%% HPGAN calculations 
    #these are here as for hpgan testing results
    # calculate mpjpe
    errorP1, errorP2 = mpjperror(testBatch, hpgan, data_preprocessing, inputSequenceLength, 3) #for 2d to 3d lift
    logging.info("--------- Starting hpgan 10->20 testing, noise added to states ------------")

    fmtMetrc = " " + " ".join(["{:.3f}"]*len(errorP1[0]))
    for i in range(len(errorP1)):
        logging.info("------------------------")
        logging.info("--MPJPE P1 10-->20: Subject_{}, activity_{} value_{:.3f}".format(testSubjectIDs[i], testActivityIDs[i], np.mean(errorP1[i])))
        logging.info("--MPJPE P1 10-->20: Full sequence: {}".format(fmtMetrc.format(*errorP1[i])))
        logging.info("--MPJPE P2 10-->20: Subject_{}, activity_{} value_{:.3f}".format(testSubjectIDs[i], testActivityIDs[i], np.mean(errorP2[i])))
        logging.info("--MPJPE P2 10-->20: Full sequence: {}".format(fmtMetrc.format(*errorP2[i])))
    
#    error = mpjperrorOriginal(pelvisTest3D_n, orginalTestBatch, d2d3GAN_ZD, d2stats, inputSequenceLength, 2, pelvis=pelvisPos[:,inputSequenceLength:,:,:]) #for 2d to 3d lift
#    logging.info("MPJPE error: {}".format(error))
    logging.info("---- error by activity summary -----")
    for i in range(1,np.max(testActivityIDs)):
        ep1 = np.mean(np.array(errorP1[np.where(testActivityIDs == i)]))
        ep2 = np.mean(np.array(errorP2[np.where(testActivityIDs == i)]))
        logging.info("MPJPE P1 summary for activity_{} = {:.3f}".format(i, ep1))
        logging.info("MPJPE P2 summary for activity_{} = {:.3f}".format(i, ep2))
        logging.info("---- end error by activity summary -----")
    
    logging.info("--------- Completed hpgan 10->20 testing, noise added to z dimension ------------")
    

#%% Cell - 2D to 3D lift with 10 - 10 approach, single pose estimation

    #this loss function compares the driver sequence (10) with the first 10 of the generated sequence
    def d2d3generatorLoss10(real, pred, body_inf):
            
        jts = pred.shape[2]
        pred = tf.reshape(pred, [batchSize, output_sequence_length, jts, 3])

        real10 = real[:,0:inputSequenceLength,:] #get the first 10 ground truth sequences (driver sequence)
        real10 = tf.reshape(real10, [batchSize, inputSequenceLength, jts, 3])
        
        pred10 = pred[:,0:inputSequenceLength,:,:]
        
        #construct pose offset array for pose consistancy calculation
        firstOne = real10[:,inputSequenceLength-1:inputSequenceLength,:]
        firstOne = tf.reshape(firstOne, [batchSize, 1 , jts, 3])
        
        last9   = pred[:,0:inputSequenceLength-1:,:]
        offsetSeq = tf.concat([firstOne, last9], axis=1)
        poseLoss = tf.sqrt(tf.reduce_sum(tf.square(pred10-offsetSeq)))    
        poseLoss = tf.maximum(0.0001, poseLoss) / (16*10)
     
#        posLoss = tf.reduce_sum(tf.square(real20-pred20)) / (16 * 20)
#        posLoss = tf.reduce_mean(tf.square(real20-pred20)) # MSE
        posLoss = tf.sqrt(tf.reduce_sum(tf.square(real10-pred10)))    #l2 norm
        posLoss = tf.maximum(0.0001, posLoss) / (16*10)

#        posLoss = keras.losses.mse(real20, pred20)
        
        bLoss = nn.bone_loss(real10, pred10, body_inf) / (16 * 10)  #batchSize and sequence length
#        bLoss = nn.bone_loss(offsetSeq, pred10, body_inf) / (16 * 10)  #batchSize and sequence length
        
        try:
            tf.debugging.check_numerics(bLoss, "Bonie")
        except Exception as e:
            assert "Checking for NaN " in e.message

        return posLoss, bLoss, poseLoss

    inputSequenceLength = 10
    output_sequence_length = 10
    sequenceLength = inputSequenceLength + output_sequence_length
    latentDims = 128
    JOINTCNT, AXISIS = dataGeneratorTrain.element_shape

    critic2d3d = nnet.NNCritic((sequenceLength, JOINTCNT*3))
    critic2d3d.summary()

    d2d3Generator = nnet.D2TOD3Generator10x10(1024, (inputSequenceLength, JOINTCNT*2), (latentDims))
    d2d3Generator.summary()    
    
    d2d3GAN10x10 = D2D3Gan(critic2d3d, d2d3Generator, hpgan.discriminator, nn.generate_random, criticSteps=10, genSteps=2, latentDims=latentDims, upliftOnly=True) 

    d2d3GAN10x10.setLengths(batchSize, inputSequenceLength, output_sequence_length)
    d2d3GAN10x10.setConstants(body_inf)
    
    d2LearnRate = 5e-5 # learning rate
    d2d3G_Optimiser = keras.optimizers.Adam(learning_rate=d2LearnRate, beta_1=decayRate)
    d2d3C_Optimiser = keras.optimizers.Adam(learning_rate=criticLR, beta_1=decayRate)
    d2d3GAN10x10.compile(d2d3G_Optimiser, d2d3G_Optimiser, d2d3generatorLoss10, criticLoss)
    
    # change data for 20 lenght sequences
    pelvis3Dtrain_n = pelvis3Dtrain_n[:,:sequenceLength,:,:]
    pelvisVal2D_n = pelvisVal2D_n[:,:sequenceLength,:,:]
    pelvisVal3D_n = pelvisVal3D_n[:,:sequenceLength,:,:]
    
    printZData = nn.generate_random(z_rand_type, z_rand_params, shape=[11, latentDims]) #get a new noise vector for printing
    printBatch = pelvis3Dtrain_n[0:11,:,:]  #3D ground truth for the full sequence, pelvis centered for 2D representation and normalised
    printPelvis = pelvisPosTT[0:11,:sequenceLength,:,:]
    printActIDs = activityIDs[0:11]
    printSubIDs = subjectIDs[0:11]
    printPast  = printBatch[:, 0:inputSequenceLength, :, 0:2]  #driver sequence

    d3GroundT = inputBatch[0:11,:,:,:] #3D ground truth of 3D data, for sequence image generation only
    
    d2d3Mon = d2d3Monitor(d3GroundT, printBatch, printPelvis, printPast, printZData, printActIDs, printSubIDs, output_2D_folder, data_preprocessing, d2stats, d2d3GAN10x10, batchSize=batchSize, logging=logging)

#    history = d2d3GAN10x10.fit(pelvis3Dtrain_n, validation_data=(pelvisVal2D_n, pelvisVal3D_n, y1), batch_size=batchSize, epochs=1, callbacks=[d2d3Mon,tbCallBack], shuffle=True)

    #plot training statistics
#    myplt.trainingPlotter2_2D(history.history, epochs, save=True, show=True)

#%% Cell 2d3d 10x10 results
    #run tests, eculidian distances and MMD and print test generations
    logging.info("--------- Starting 10->10 testing, noise added to states ------------")

    pelvisTest3D_nn = pelvisTest3D_n[:,:sequenceLength,:,:]
    pelvisPos = pelvisPos[:,:sequenceLength,:,:]
    tests(pelvisTest3D_nn, testSubjectIDs, testActivityIDs, d2d3GAN10x10, data_preprocessing, normaliser2d=d2stats, d3GroundT=testBatch, pelvisData=pelvisPos)
    
    positionMetrics(pelvisTest3D_nn, d2d3GAN10x10, testSubjectIDs, testActivityIDs, d2stats)
    del pelvisTest3D_nn
    
    errorP1, errorP2 = mpjperror(pelvisTest3D_n, d2d3GAN10x10, d2stats, inputSequenceLength, 2, single=True) #for 2d to 3d lift
    logging.info("MPJPE error: {}".format(errorP1))
    
    logging.info("--------- Completed 10->10 testing, noise added to states ------------")


#%% Cell 2d3d with noise added to the z dimension.

#reset all the data after droping the future poses for 10x10 training above

    inputSequenceLength = 10
    output_sequence_length = 20
    sequenceLength = inputSequenceLength + output_sequence_length
    latentDims = 128
    JOINTCNT, AXISIS = dataGeneratorTrain.element_shape
    
    #training, validation and test data, unnormalise and recentre on the pelvis
    #unnormalise
    normalInBatch = data_preprocessing.unnormalize(inputBatch)   
    normalValBtch = data_preprocessing.unnormalize(validateBatch)
    normalTestBtch = data_preprocessing.unnormalize(testBatch)

    # recenter all batches on the pelvis
    pelvis = JointType(0)

    pelvisPosTT  = normalInBatch[:,:,pelvis.value:pelvis.value+1]
    pelvisTrain3D = normalInBatch - pelvisPosTT
    #change the pelvis 0.0 value to a small value, to enable learning on 
    pelvisTrain3D = np.where(pelvisTrain3D == 0.0, 0.0001, pelvisTrain3D)
    pelvis2Dtrain = pelvisTrain3D[:,:,:,0:2]
    
    pelvisPos = normalValBtch[:,:,pelvis.value:pelvis.value+1]
    pelvisVal3D = normalValBtch - pelvisPos
    #change the pelvis 0.0 value to a small value, to enable learning on 
    pelvisVal3D = np.where(pelvisVal3D == 0.0, 0.0001, pelvisVal3D)
    pelvisVal2D = pelvisVal3D[:,:,:,0:2]
    
    pelvisPos = normalTestBtch[:,:,pelvis.value:pelvis.value+1]
    pelvisTest3D = normalTestBtch - pelvisPos
    #change the pelvis 0.0 value to a small value, to enable learning on 
    pelvisTest3D = np.where(pelvisTest3D == 0.0, 0.0001, pelvisTest3D)
    pelvisTest2D = pelvisTest3D[:,:,:,0:2]
   
    #compute stats for the full dataset
    d2stats = nnet.Stats2D(np.concatenate((pelvisTrain3D, pelvisVal3D, pelvisTest3D)), meanType="global_linear")
    
 #   skeleton2D.draw_to_file(normalInBatch[0:5,:,:,:], subjectIDs[0], os.path.join(args.output_folder, "2D_representation/train_3D_GT{}_S{}_A{}.png".format(0, subjectIDs[0:5], activityIDs[0:5])))
 #   skeleton2D.draw2d_to_file(pelvisTrain3D[0:5,:,:,0:2], subjectIDs[0], os.path.join(args.output_folder, "2D_representation/train_2D_GT{}_S{}_A{}.png".format(0, subjectIDs[0:5], activityIDs[0:5])))
    myplt.plot2DSkel(body_inf, pelvisTrain3D[0][0])    
    myplt.plot3DSkel(body_inf, pelvisTrain3D[0][0], "Original 3D Pose")
    
    #normalise
    pelvis3Dtrain_n = d2stats.normalise(pelvisTrain3D)
    pelvis2Dtrain_n = pelvis2Dtrain[:,:,:,0:2]
    
    pelvisVal3D_n = d2stats.normalise(pelvisVal3D)
    pelvisVal2D_n = pelvisVal3D_n[:,:,:,0:2]
            
    pelvisTest3D_n = d2stats.normalise(pelvisTest3D)
    pelvisTest2D_n = pelvisTest3D_n[:,:,:,0:2]
    
#    y1 = np.ones((testRecCount, 1), dtype=np.int32)
    
 #   skeleton2D.draw_to_file(normalValBtch[0:5,:,:,:], valSubjectIDs[0], os.path.join(args.output_folder, "2D_representation/val_3D_GT{}_S{}_A{}.png".format(0, valSubjectIDs[0:5], valActivityIDs[0:5])))
 #   skeleton2D.draw2d_to_file(pelvisVal3D[0:5,:,:,0:2], valSubjectIDs[0], os.path.join(args.output_folder, "2D_representation/val_2D_GT{}_S{}_A{}.png".format(0, valSubjectIDs[0:5], valActivityIDs[0:5])))

 #   skeleton2D.draw_to_file(normalTestBtch[0:5,:,:,:], testSubjectIDs[0], os.path.join(args.output_folder, "2D_representation/test_3D_GT{}_S{}_A{}.png".format(0, testSubjectIDs[0:5], testActivityIDs[0:5])))
 #   skeleton2D.draw2d_to_file(pelvisTest3D[0:5,:,:,0:2], testSubjectIDs[0], os.path.join(args.output_folder, "2D_representation/test_2D_GT{}_S{}_A{}.png".format(0, testSubjectIDs[0:5], testActivityIDs[0:5])))
    
    #free up memory
    del normalInBatch
    del normalValBtch
    del normalTestBtch

    critic2d3d_ZD = nnet.NNCritic((sequenceLength, JOINTCNT*3))
    critic2d3d_ZD.summary()

    d2d3Gen_ZD = nnet.D2TOD3GeneratorZ(1024, (inputSequenceLength, JOINTCNT*2), (inputSequenceLength, JOINTCNT, 1), (latentDims))
    d2d3Gen_ZD.summary()    
    
    d2d3GAN_ZD = D2D3Gan(critic2d3d_ZD, d2d3Gen_ZD, hpgan.discriminator, nn.generate_random, criticSteps=10, genSteps=2, latentDims=latentDims) 
    d2d3GAN_ZD.setLengths(batchSize, inputSequenceLength, output_sequence_length)
    d2d3GAN_ZD.setConstants(body_inf) #, z_rand_params = {'low':-1.0, 'high':1.0, 'mean':0.0, 'std':0.2})
    
    d2LearnRate = 5e-5 # learning rate
    d2d3G_Optimiser = keras.optimizers.Adam(learning_rate=d2LearnRate, beta_1=decayRate)
    d2d3C_Optimiser = keras.optimizers.Adam(learning_rate=criticLR, beta_1=decayRate)
    d2d3GAN_ZD.compile(d2d3G_Optimiser, d2d3G_Optimiser, d2d3generatorLoss, criticLoss)

    #setup here the end of epoch print of the training status,  ## change to training data for production
    printZData = nn.generate_random(z_rand_type, z_rand_params, shape=[11, latentDims]) #get a new noise vector for printing
    printBatch = pelvis3Dtrain_n[0:11,:,:]  #3D ground truth for the full sequence, pelvis centered for 2D representation and normalised
    printPelvis = pelvisPosTT[0:11,:,:]
    printActIDs = activityIDs[0:11]
    printSubIDs = subjectIDs[0:11]
    printPast  = printBatch[:, 0:inputSequenceLength, :, 0:2]  #driver sequence

    d3GroundT = inputBatch[0:11,:,:,:] #3D ground truth of 3D data, for sequence image generation only
    
    d2d3Mon = d2d3Monitor(d3GroundT, printBatch, printPelvis, printPast, printZData, printActIDs, printSubIDs, output_2D_folder, data_preprocessing, d2stats, d2d3GAN, batchSize=batchSize, logging=logging)

#    skeletonData, probs = nnet.generateSkeltons2D(printBatch, printPast, d3GroundT, printZData, d2d3GAN_ZD, data_preprocessing, d2stats, batchIndex=0)

#    history = d2d3GAN_ZD.fit(pelvis3Dtrain_n, validation_data=(pelvisVal2D_n, pelvisVal3D_n, y1), batch_size=batchSize, epochs=125, callbacks=[d2d3Mon,tbCallBack], shuffle=True)
     
    #plot training statistics
#    myplt.trainingPlotter2_2D(history.history, epochs, save=True, show=True)
    
#%% Cell 2d3d results noise added to z dimension
    logging.info("--------- Starting 10->20 testing, noise added to z dimension ------------")

    #run tests, eculidian distances and MMD and print test generations
    tests(pelvisTest3D_n, testSubjectIDs, testActivityIDs, d2d3GAN_ZD, data_preprocessing, normaliser2d=d2stats, d3GroundT=testBatch, pelvisData=pelvisPos)
 #   positionMetrics(pelvisTest3D_n, d2d3GAN_ZD, testSubjectIDs, testActivityIDs, d2stats)
    
#    orginalTestBatch = data_preprocessing.unnormalize(testBatch)
    errorP1, errorP2 = mpjperror(pelvisTest3D_n, d2d3GAN_ZD, d2stats, inputSequenceLength, 2, pelvis=pelvisPos[:,inputSequenceLength:,:,:]) #for 2d to 3d lift
    if dataset == 'nturgbd':
        errorP1 *= 1000.0
        errorP2 *= 1000.0
    myplt.mpjpePrinter(errorP1, errorP2, logging, testSubjectIDs, testActivityIDs)
    
    logging.info("--------- Completed 10->20 testing, noise added to z dimension ------------")
#    del orginalTestBatch

#%% Adding Noise to test data
    
    logging.info("---- error by activity summary 10->20----- with noise injection")
    
    noise = getNoise(pelvisTest3D_n.shape) * 0.05
    withNoise = pelvisTest3D_n + noise
    errorP1, errorP2 = mpjperror(withNoise, d2d3GAN_ZD, d2stats, inputSequenceLength, 2, pelvis=pelvisPos[:,inputSequenceLength:,:,:])
    if dataset == 'nturgbd':
        errorP1 *= 1000.0
        errorP2 *= 1000.0
    myplt.mpjpePrinter(errorP1, errorP2, logging, testSubjectIDs, testActivityIDs)
    
    logging.info("--------- Completed 10->20 testing, noise added to z dimension & noise injection------------")
    
    del noise
    del withNoise

#%% Cell - 2D to 3D lift with 10 - 10 approach, single pose estimation with noise in z dimension

    #this loss function compares the driver sequence (10) with the first 10 of the generated sequence
    def d2d3generatorLoss10(real, pred, body_inf):
        
        jts = pred.shape[2]
        pred = tf.reshape(pred, [batchSize, output_sequence_length, jts, 3])

        real10 = real[:,0:inputSequenceLength,:] #get the first 10 ground truth sequences (driver sequence)
        real10 = tf.reshape(real10, [batchSize, inputSequenceLength, jts, 3])
        
        pred10 = pred[:,0:inputSequenceLength,:,:]
        
        #construct pose offset array for pose consistancy calculation
        firstOne = real10[:,inputSequenceLength-1:inputSequenceLength,:]
        firstOne = tf.reshape(firstOne, [batchSize, 1 , jts, 3])
        
        last9   = pred[:,0:inputSequenceLength-1:,:]
        offsetSeq = tf.concat([firstOne, last9], axis=1)
        poseLoss = tf.sqrt(tf.reduce_sum(tf.square(pred10-offsetSeq)))    
        poseLoss = tf.maximum(0.0001, poseLoss) / (16*10)
     
#        posLoss = tf.reduce_sum(tf.square(real20-pred20)) / (16 * 20)
#        posLoss = tf.reduce_mean(tf.square(real20-pred20)) # MSE
        posLoss = tf.sqrt(tf.reduce_sum(tf.square(real10-pred10)))    #l2 norm
        posLoss = tf.maximum(0.0001, posLoss) / (16*10)
        
        bLoss = nn.bone_loss(real10, pred10, body_inf) / (16 * 10)  #batchSize and sequence length
#        bLoss = nn.bone_loss(offsetSeq, pred10, body_inf) / (16 * 10)  #batchSize and sequence length
        
        try:
            tf.debugging.check_numerics(bLoss, "Bonie")
        except Exception as e:
            assert "Checking for NaN " in e.message

        return posLoss, bLoss, poseLoss

    inputSequenceLength = 10
    output_sequence_length = 10
    sequenceLength = inputSequenceLength + output_sequence_length
    latentDims = 128

    critic2d3d = nnet.NNCritic((sequenceLength, JOINTCNT*3))
    critic2d3d.summary()

    d2d3Generator = nnet.D2TOD3Generator10x10_Z(1024, (inputSequenceLength, JOINTCNT*2), (inputSequenceLength, JOINTCNT, 1), (latentDims))
    d2d3Generator.summary()    
    
    d2d3GAN10x10 = D2D3Gan(critic2d3d, d2d3Generator, hpgan.discriminator, nn.generate_random, criticSteps=10, genSteps=2, latentDims=latentDims, upliftOnly=True) 

    d2d3GAN10x10.setLengths(batchSize, inputSequenceLength, output_sequence_length)
    d2d3GAN10x10.setConstants(body_inf, z_rand_params = {'low':-1.0, 'high':1.0, 'mean':0.0, 'std':0.2})
    
    d2LearnRate = 5e-5 # learning rate
    d2d3G_Optimiser = keras.optimizers.Adam(learning_rate=d2LearnRate, beta_1=decayRate)
    d2d3C_Optimiser = keras.optimizers.Adam(learning_rate=criticLR, beta_1=decayRate)
    d2d3GAN10x10.compile(d2d3G_Optimiser, d2d3G_Optimiser, d2d3generatorLoss10, criticLoss)
    
    # change data for 20 length sequences
    pelvis3Dtrain_n = pelvis3Dtrain_n[:,:sequenceLength,:,:]
    pelvisVal2D_n = pelvisVal2D_n[:,:sequenceLength,:,:]
    pelvisVal3D_n = pelvisVal3D_n[:,:sequenceLength,:,:]
    
    printZData = nn.generate_random(z_rand_type, z_rand_params, shape=[11, latentDims]) #get a new noise vector for printing
    printBatch = pelvis3Dtrain_n[0:11,:,:]  #3D ground truth for the full sequence, pelvis centered for 2D representation and normalised
    printPelvis = pelvisPosTT[0:11,:sequenceLength,:,:]
    printActIDs = activityIDs[0:11]
    printSubIDs = subjectIDs[0:11]
    printPast  = printBatch[:, 0:inputSequenceLength, :, 0:2]  #driver sequence

    d3GroundT = inputBatch[0:11,:,:,:] #3D ground truth of 3D data, for sequence image generation only
    
    d2d3Mon = d2d3Monitor(d3GroundT, printBatch, printPelvis, printPast, printZData, printActIDs, printSubIDs, output_2D_folder, data_preprocessing, d2stats, d2d3GAN10x10, batchSize=batchSize, logging=logging)

    history = d2d3GAN10x10.fit(pelvis3Dtrain_n, validation_data=(pelvisVal2D_n, pelvisVal3D_n, y1), batch_size=batchSize, epochs=125, callbacks=[d2d3Mon,tbCallBack], shuffle=True)

    #plot training statistics
    myplt.trainingPlotter2_2D(history.history, epochs, save=True, show=True)
    
#%% Cell 2d3d 10x10 results
    #run tests, eculidian distances and MMD and print test generations
    logging.info("--------- Starting 10x10 testing, noise added to z dimension ------------")

    pelvisTest3D_nn = pelvisTest3D_n[:,:sequenceLength,:,:]
    pelvisPos = pelvisPos[:,:sequenceLength,:,:]
    tests(pelvisTest3D_nn, testSubjectIDs, testActivityIDs, d2d3GAN10x10, data_preprocessing, normaliser2d=d2stats, d3GroundT=testBatch, pelvisData=pelvisPos)
    
 #   positionMetrics(pelvisTest3D_nn, d2d3GAN10x10, testSubjectIDs, testActivityIDs, d2stats)
    del pelvisTest3D_nn
        
    errorP1, errorP2 = mpjperror(pelvisTest3D_n, d2d3GAN10x10, d2stats, inputSequenceLength, 2, single=True, pelvis=pelvisPos[:,:inputSequenceLength,:,:]) #for 2d to 3d lift

    myplt.mpjpePrinter(errorP1, errorP2, logging, testSubjectIDs, testActivityIDs)

    logging.info("--------- Completed 10x10 testing, noise added to z dimension ------------")


#%% Adding Noise to test data
    
    logging.info("---- error by activity summary ----- with noise injection")
    
    noise = getNoise(pelvisTest3D_n.shape) * 0.05
    withNoise = pelvisTest3D_n + noise
    errorP1, errorP2 = mpjperror(withNoise, d2d3GAN10x10, d2stats, inputSequenceLength, 2, single=True, pelvis=pelvisPos[:,:inputSequenceLength,:,:])
    
    myplt.mpjpePrinter(errorP1, errorP2, logging, testSubjectIDs, testActivityIDs)
    
    logging.info("--------- Completed 10x10 testing, noise added to z dimension & noise injection------------")
    
    del noise
    del withNoise

#%% Cell 7

#print out a sample of the training and test sequences to check the printing is working
"""
dataGeneratorTest.reset()
dataGeneratorTrain.reset()

sampleTrain, targets, rntBatchSize, activityIDs, subjectIDs = dataGeneratorTrain.next_minibatch(256)

for i in range(150,175):
    someSkels  = sampleTrain[i]
    activityID = activityIDs[i]
    subjectID  = subjectIDs[i]
    #un normalise the skeletons
#    unNormSkels = normaliser.mean_std_unnormalize(someSkels, std_factor=2.0)
    unNormSkels = data_preprocessing.unnormalize(someSkels)

    skeleton2D.draw_to_file([unNormSkels], subjectID, os.path.join(args.output_folder, "trainSample_S{}_A{}.png".format(subjectID, activityID)))

sampleTest, targets, rntBatchSize, activityIDs, subjectIDs  = dataGeneratorTest.next_minibatch(256)

for i in range(150,175):
    someSkels  = sampleTest[i]
    activityID = activityIDs[i]
    subjectID  = subjectIDs[i]
    #un normalise the skeletons
#    unNormSkels = normaliser.mean_std_unnormalize(someSkels, std_factor=2.0)
    unNormSkels = data_preprocessing.unnormalize(someSkels)

    skeleton2D.draw_to_file([unNormSkels], subjectID, os.path.join(args.output_folder, "testSample_S{}_A{}.png".format(subjectID, activityID)))
              
"""


   
    
    
    
