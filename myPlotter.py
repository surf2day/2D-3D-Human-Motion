#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 09:30:27 2021

@author: cbunn
"""

# plotting functions for various reporting
import matplotlib.pyplot as plt
import numpy as np

def trainingPlotter(history, epoch, save=True, show=False):
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    ax1.grid()
    ax2.grid()
    ax3.grid()
    
    gLoss = history["2d_ganLoss"]
    val_gLoss = history["2d_val_ganLoss"]
    
    criticLoss = history["2d_criticLoss"]
    val_criticLoss = history["2d_val_criticLoss"]
    
    dLoss = history["dLoss"]
    val_dLoss = history["val_dLoss"]
    
    bLoss = history["bLoss"]
    val_bLoss = history["val_bLoss"]
    
    pgLoss = history["pgLoss"]
    val_pgLoss = history["val_pgLoss"]
    
    #generator loss
    ax1.plot(gLoss)
    ax1.plot(val_gLoss)
    ax1.set_title("Generator Loss")
    ax1.set(ylabel="Loss")
    ax1.set(xlabel="Epoch")
    ax1.legend(["training", "validation"])
    
    #Critic
    ax2.plot(criticLoss)
    ax2.plot(val_criticLoss)
    ax2.set_title("Critic Loss")
    ax2.set(ylabel="Loss")
    ax2.set(xlabel="Epoch")
    ax2.legend(["training", "validation"])
    
    #discriminator
    ax3.plot(dLoss)
    ax3.plot(val_dLoss)
    ax3.set_title("Discriminator Loss")
    ax3.set(ylabel="Loss")
    ax3.set(xlabel="Epoch")
    ax3.legend(["training", "validation"])
    
    #Bone loss
    ax3.plot(bLoss)
    ax3.plot(val_bLoss)
    ax3.set_title("Bone Loss")
    ax3.set(ylabel="Loss")
    ax3.set(xlabel="Epoch")
    ax3.legend(["training", "validation"])
    
    #pose consistancy
    ax3.plot(pgLoss)
    ax3.plot(val_pgLoss)
    ax3.set_title("Discriminator Loss")
    ax3.set(ylabel="Loss")
    ax3.set(xlabel="Epoch")
    ax3.legend(["training", "validation"])
    
    fig.tight_layout()
    if show:
        plt.show()    
    if save:
        fig.savefig("./results/output/plots/trainingPlot_{}.png".format(epoch), format='png')
        
def trainingPlotter2(history, epoch, save=True, show=True):
    
    gLoss = history["ganLoss"]
    val_gLoss = history["val_ganLoss"]
    
    criticLoss     = history["criticLoss"]
    val_criticLoss = history["val_criticLoss"]
    
    dLoss     = history["dLoss"]
    val_dLoss = history["val_dLoss"]
    
    bLoss    = history["boneLoss"]
    valBloss = history["val_boneLoss"]
    
    poseLoss    = history["poseConsist"]
    valPoseLoss = history["val_poseConsist"]
        
    #generator loss
    plt.plot(gLoss, label="Training")
    plt.plot(val_gLoss, label="Validation")
    plt.title("Generator Loss")
    plt.legend(loc="upper left")
    plt.ylabel(ylabel="Loss")
    plt.xlabel(xlabel="Epoch")
    plt.legend()
    if save:
        plt.savefig("./results/output/plots/trainingPlot_generator_{}.png".format(epoch), format='png')
    if show:
        plt.show()    
    
    #Critic
    plt.plot(criticLoss, label="Training")
    plt.plot(val_criticLoss, label="Validation")
    plt.title("Critic Loss")
    plt.ylabel(ylabel="Loss")
    plt.xlabel(xlabel="Epoch")
    plt.legend()   
    if save:
        plt.savefig("./results/output/plots/trainingPlot_critic_{}.png".format(epoch), format='png')    
    if show:
        plt.show() 
    
    #discriminator
    plt.plot(dLoss, label="Training")
    plt.plot(val_dLoss, label="Validation")
    plt.title("Discriminator Loss")
    plt.ylabel(ylabel="Loss")
    plt.xlabel(xlabel="Epoch")
    plt.legend()
    if save:
        plt.savefig("./results/output/plots/trainingPlot_discriminator_{}.png".format(epoch), format='png')    
    if show:
        plt.show()    
        
    #Bone loss
    plt.plot(bLoss, label="Training")
    plt.plot(valBloss, label="Validation")
    plt.title("Bone Loss")
    plt.ylabel(ylabel="Loss")
    plt.xlabel(xlabel="Epoch")
    plt.legend()
    if save:
        plt.savefig("./results/output/plots/trainingPlot_BoneLoss_{}.png".format(epoch), format='png')    
    if show:
        plt.show()
        
    #Pose Loss
    plt.plot(poseLoss, label="Training")
    plt.plot(valPoseLoss, label="Validation")
    plt.title("Pose conistancy Loss")
    plt.ylabel(ylabel="Loss")
    plt.xlabel(xlabel="Epoch")
    plt.legend()
    if save:
        plt.savefig("./results/output/plots/trainingPlot_PoseLoss_{}.png".format(epoch), format='png')    
    if show:
        plt.show()    
        
        
        
def trainingPlotter2_2D(history, epoch, save=True, show=True):
    
    gLoss = history["2d_ganLoss"]
    val_gLoss = history["val_2d_ganLoss"]
    
    criticLoss     = history["2d_criticLoss"]
    val_criticLoss = history["val_2d_criticLoss"]
    
    pLoss     = history["2d_positionLoss"]
    val_pLoss = history["val_2d_positionLoss"]
    
    bLoss    = history["2d_boneLoss"]
    valBloss = history["val_2d_boneLoss"]
    
    poseLoss    = history["2d_poseLoss"]
    valPoseLoss = history["val_2d_poseLoss"]
        
    #generator loss
    plt.plot(gLoss, label="Training")
    plt.plot(val_gLoss, label="Validation")
    plt.title("GAN Loss")
    plt.legend(loc="upper left")
    plt.ylabel(ylabel="Loss")
    plt.xlabel(xlabel="Epoch")
    plt.legend()
    if save:
        plt.savefig("./results/output/plots/2D_trainingPlot_gan_{}.png".format(epoch), format='png')
    if show:
        plt.show()    
    
    #Critic
    plt.plot(criticLoss, label="Training")
    plt.plot(val_criticLoss, label="Validation")
    plt.title("Critic Loss")
    plt.ylabel(ylabel="Loss")
    plt.xlabel(xlabel="Epoch")
    plt.legend()   
    if save:
        plt.savefig("./results/output/plots/2D_trainingPlot_critic_{}.png".format(epoch), format='png')    
    if show:
        plt.show() 
    
    #discriminator
    plt.plot(pLoss, label="Training")
    plt.plot(val_pLoss, label="Validation")
    plt.title("Position Loss")
    plt.ylabel(ylabel="Loss")
    plt.xlabel(xlabel="Epoch")
    plt.legend()
    if save:
        plt.savefig("./results/output/plots/2D_trainingPlot_position_{}.png".format(epoch), format='png')    
    if show:
        plt.show()    
        
    #Bone loss
    plt.plot(bLoss, label="Training")
    plt.plot(valBloss, label="Validation")
    plt.title("Bone Loss")
    plt.ylabel(ylabel="Loss")
    plt.xlabel(xlabel="Epoch")
    plt.legend()
    if save:
        plt.savefig("./results/output/plots/2D_trainingPlot_BoneLoss_{}.png".format(epoch), format='png')    
    if show:
        plt.show()
        
    #Pose Loss
    plt.plot(poseLoss, label="Training")
    plt.plot(valPoseLoss, label="Validation")
    plt.title("Pose conistancy Loss")
    plt.ylabel(ylabel="Loss")
    plt.xlabel(xlabel="Epoch")
    plt.legend()
    if save:
        plt.savefig("./results/output/plots/2D_trainingPlot_PoseLoss_{}.png".format(epoch), format='png')    
    if show:
        plt.show()
        
        
#plot the x, y, z positions for a sequence of real and fake 
def plotSequence(realSeq, fakeSeq, index, title):
    
    #ground truth
    head = realSeq[index]
    seqLen, _, _ = head.shape
    headx = np.reshape(head[:,:, 0:1], (seqLen))
    heady = np.reshape(head[:,:, 1:2], (seqLen))
    headz = np.reshape(head[:,:, 2:3], (seqLen))
    
    #generated
    head = fakeSeq[index]
    seqLen, _, _ = head.shape
    headxF = np.reshape(head[:,:, 0:1], (seqLen))
    headyF = np.reshape(head[:,:, 1:2], (seqLen))
    headzF = np.reshape(head[:,:, 2:3], (seqLen))
    
    plt.plot(headx, label="real x", color="red")
    plt.plot(heady, label="real y", color="green")
    plt.plot(headz, label="real z", color="blue")
    
    plt.plot(headxF, label="fake x", color="red", linestyle="dashed")
    plt.plot(headyF, label="fake y", color="green", linestyle="dashed")
    plt.plot(headzF, label="fake z", color="blue", linestyle="dashed")
    
    plt.title(title)
    plt.ylabel(ylabel="Value")
    plt.xlabel(xlabel="Pose")
    plt.legend()
    plt.show()

def testingPlotter(boneLoss, consistancyLoss, numSample, save=True, show=False):
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
    if show:
        plt.show()    
    if save:
        fig.savefig("./results/output/plots/testingPlot_{}.png".format(numSample), format='png')
        
#plot a single 2D skeleton
def plot2DSkel(bodyInfo, skel):
     colour = {"center":"red", "left":"green", "right":"blue"}
     for (end1, end2, tag) in bodyInfo.bones:
         pt1 = skel[end1]
         pt2 = skel[end2]
         plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=colour[tag])
     plt.show()        
        
#plot a single 3D skeleton
def plot3DSkel(bodyInfo, skel, title):
         colour = {"center":"red", "left":"green", "right":"blue"}      
         fig = plt.figure()
         ax = plt.axes(projection ='3d')
         ax.set_title = title
          
         for (end1, end2, tag) in bodyInfo.bones:
             pt1 = skel[end1]
             pt2 = skel[end2]
             x = [pt1[0], pt2[0]]
             y = [pt1[1], pt2[1]]
             z = [pt1[2], pt2[2]]
#             ax.plot3D(x, y, z, color="red")
             ax.plot3D(x, z, y, color=colour[tag])
#             ax.plot3D(z, x, y, color="blue")
         plt.show() 
         print()       
       
#results printer 
def mpjpePrinter(errorP1, errorP2, logging, subjects, activities):
        
    fmtMetrc = " " + " ".join(["{:.1f}"]*len(errorP1[0]))
    for i in range(len(errorP1)):
        logging.info("------------------------")
        logging.info("--MPJPE P1 10-->10: Subject_{}, activity_{} value_{:.1f} epoch_{}".format(subjects[i], activities[i], np.mean(errorP1[i]), i))
        logging.info("--MPJPE P1 10-->10: Full sequence: {}".format(fmtMetrc.format(*errorP1[i])))
        logging.info("--MPJPE P2 10-->10: Subject_{}, activity_{} value_{:.1f} epoch_{}".format(subjects[i], activities[i], np.mean(errorP2[i]), i))
        logging.info("--MPJPE P2 10-->10: Full sequence: {}".format(fmtMetrc.format(*errorP2[i])))
    logging.info("---- error by activity summary -----")

    for i in range(1,np.max(activities)+1):
        ep1 = np.mean(np.array(errorP1[np.where(activities == i)]))
        ep2 = np.mean(np.array(errorP2[np.where(activities == i)]))
        logging.info("MPJPE P1 summary for activity_{} = {:.1f}".format(i, ep1))
        logging.info("MPJPE P2 summary for activity_{} = {:.1f}".format(i, ep2))
        logging.info("---- end error by activity summary -----")
    
    logging.info("MPJPE   = {}".format(np.mean(errorP1)))
    logging.info("P-MPJPE = {}".format(np.mean(errorP2)))
    logging.info("Worst = {}".format(fmtMetrc.format(*np.argsort(np.mean(errorP1, axis=1))[-10:])))
    logging.info("Worst = {}".format(fmtMetrc.format(*np.sort(np.mean(errorP1, axis=1))[-10:])))
    logging.info("Best  = {}".format(fmtMetrc.format(*np.argsort(np.mean(errorP1, axis=1))[:10])))
    logging.info("Best  = {}".format(fmtMetrc.format(*np.sort(np.mean(errorP1, axis=1))[:10])))
    
    
    
    
    