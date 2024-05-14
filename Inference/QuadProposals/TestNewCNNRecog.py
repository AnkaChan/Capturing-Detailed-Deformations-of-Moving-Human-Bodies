import CNNRecogModel
import GenerateCornerKey
import cv2
import numpy as np

if __name__ == "__main__":
    model = CNNRecogModel.CNNRecogModel()
    model.loadWeights_BN("D:\GDrive\mocap/2019_01_30_new_CNN_v02\CNN_2char_bn_unified.ckpt")

    GenerateCornerKey.generateCornerKey("./TestData/1/pattern.txt", "./TestData/1/pattern.h5", model)

