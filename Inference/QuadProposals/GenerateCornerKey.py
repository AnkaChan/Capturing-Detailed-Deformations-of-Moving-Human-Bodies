import os
import cv2
import glob
import numpy as np
import h5py

from .ReadPatternFile import *
from .CNNRecogModel import  *

def generateCornerKey(patternFile, subImgH5File, model, useSoftMaxCutOff=True):
    unitard = Unitard()
    unitard.readPatternFile(patternFile)
    imgList = []
    nameList = []
    print("number of corners", len(unitard.corners))
    print("number of patterns", len(unitard.patterns))

    h5File = h5py.File(subImgH5File, 'a')
    data = h5File['data']

    imgArray = np.array(data)
    imgArray = imgArray.astype(np.float32)
    print("imgSet Shape: ",imgArray.shape)

    model.setInputImgSet(imgArray)
    if (useSoftMaxCutOff):
        outputs = model.predictSoftMaxCut()
    else:
        outputs = model.predict()

    codes = []
    labels = []
    for i in range(len(outputs)):
        code = outputs[i]
        codes.append(np.string_(code))

        if code == "\0\0":
            labels.append(np.int8(0))
        else:
            labels.append(np.int8(1))

    labelSet = h5File.require_dataset("label", (data.shape[0],), dtype=np.uint8)
    labelSet[...] = labels
    codeSet = h5File.require_dataset("code", (data.shape[0],), dtype='S2')
    codeSet[...] = codes

    return codes, labels

def generateCornerKeyNoWriteBack(patternFile, subImgH5File, model):
    unitard = Unitard()
    unitard.readPatternFile(patternFile)
    imgList = []
    nameList = []
    print("number of corners", len(unitard.corners))
    print("number of patterns", len(unitard.patterns))

    h5File = h5py.File(subImgH5File, 'a')
    data = h5File['data']

    imgArray = np.array(data)
    imgArray = imgArray.astype(np.float32)
    print("imgSet Shape: ",imgArray.shape)

    model.setInputImgSet(imgArray)
    outputs = model.predict()

    codes = []
    labels = []
    for i in range(len(outputs)):
        code = outputs[i]
        codes.append(np.string_(code))

        if code == "\0\0":
            labels.append(np.int8(0))
            print(-1)
        else:
            labels.append(np.int8(1))
            print(code)


def generateCornerKeySequence(patternsPath, weightPath = './CNN_2char_bn_unified.ckpt', useSoftMaxCutOff=False,
                              softMaxCutOffVal = 0.8,  seqCodesOut = None, seqLabelsOut = None, model = None):
    if model == None:
        model = CNNRecogModel('BN')
        model.loadWeights_BN(weightPath)
    model.softMaxCutval = softMaxCutOffVal
    patternFiles = glob.glob(patternsPath + "/*.txt")
    patternFiles.sort()
    # for i in range(min(100, len(patternFiles))):
    for i in range(len(patternFiles)):
        patternFile = patternFiles[i]
        print("Processing: ", patternFile)

        subImgH5File = patternFile.replace(".txt", ".h5")

        codes, labels = generateCornerKey(patternFile, subImgH5File, model, useSoftMaxCutOff)
        if seqCodesOut != None:
            seqCodesOut.append(codes)
        if seqLabelsOut != None:
            seqLabelsOut.append(labels)

def generateCornerKeySequenceNoWriteBack(patternsPath, weightPath = './CNN_2char_bn_unified.ckpt'):

    model = CNNRecogModel('BN')
    model.loadWeights_BN(weightPath)
    patternFiles = glob.glob(patternsPath + "/*.txt")
    patternFiles.sort()
    # for i in range(min(100, len(patternFiles))):
    for i in range(len(patternFiles)):
        patternFile = patternFiles[i]
        print("Processing: ", patternFile)

        subImgH5File = patternFile.replace(".txt", ".h5")
        generateCornerKeyNoWriteBack(patternFile, subImgH5File, model)


#if __name__ == "__main__":
#    model = CNNRecogModel('BN')
#    model.loadWeights_BN()
#
#    #patternsPath = "D:\\Data\\Mocap\\Turn\\output\\Left\\Patterns"
#    #patternsPath = "D:\\Data\\Mocap\\Turn\\output\\Right\\Patterns"
#    patternsPath = "D:\\GDrive\\mocap\\2018_12_03_ForUniqueCornerId\\ForUIDExtranction\PatternsBN"
#
#    labelsPath = patternsPath + "/Labels"
#    patternFiles = glob.glob(patternsPath + "/*.txt")
#    patternFiles.sort()
#
#    os.makedirs(labelsPath, exist_ok=True)
#    #for i in range(min(100, len(patternFiles))):
#    for i in range(len(patternFiles)):
#        patternFile = patternFiles[i]
#        print("Processing: ", patternFile)
#        _, f = os.path.split(patternFile)
#        fileName, _ = os.path.splitext(f)
#        fileName = fileName.replace("Patterns", "")
#        labelPathForThisImg = labelsPath + "/" + fileName
#        os.makedirs(labelPathForThisImg, exist_ok=True)
#        generateCornerKey(patternFile, patternsPath#, model)


