import cv2
import os
import h5py
import numpy as np

class Key:
    def __init__(self):
        self.code = ""
        self.index = 0

class Corners:
    def __init__(self, i = 0, pts = [], keys = []):
        self.coordinates = pts
        self.keys = keys
        self.id = i

class Pattern:
    def __init__(self, code = "", corners = [], cornersCorrectOrder = [], id = 0, orientation = 0, valid = False):
        self.code = code
        self.corners = corners
        self.cornersCorrectOrder = cornersCorrectOrder
        self.id = id
        self.orientation = orientation
        self.valid = valid


class Unitard:
    def __init__(self):
        self.name = ""
        self.corners = []
        self.patterns = []
        self.zerosPadN = 5
        self.useGrayIimg = True
        self.subImgData = None

    def readPatternFile(self, patternFileFath):
        path, f = os.path.split(patternFileFath)
        fileName, ext = os.path.splitext(f)
        self.name = fileName

        with open(patternFileFath) as fo:

            while(True):
                line = fo.readline()
                if not line:
                    break
                if(line[0] == 'C'):
                    tokens = line.split(" ")
                    c = Corners(i = len(self.corners), pts = [float(tokens[2]), float(tokens[3])])
                    #c.coordinates[0] = float(tokens[2])
                    #c.coordinates[1] = float(tokens[3])
                    #c.id = len(self.corners)
                    self.corners.append(c)
                elif(line[0] == 'P'):
                    tokens = line.split(" ")
                    p = Pattern()
                    p.id = len(self.patterns)
                    p.corners = [int(tokens[2]), int(tokens[3]), int(tokens[4]), int(tokens[5])]
                    self.patterns.append(p)

    def readH5Codes(self, h5FileFath):
        h5File = h5py.File(h5FileFath, 'r')
        data = h5File['data']
        labelSet = h5File["label"]
        labels = np.array(labelSet)
        codeSet = h5File["code"]
        codes = np.array(codeSet)

        all_orientation = h5File.attrs.get("all_orientation", True)

        self.subImgData = np.array(data)

        for i in range(len(self.patterns)):
            p = self.patterns[i]
            orientationId = -1
            if all_orientation:
                for iOrientation in range(4):
                    cO = codes[4*i + iOrientation]
                    if cO != b'':
                        orientationId = iOrientation
                        break
            else:
                cO = codes[i]
                if cO != b'':
                    orientationId = 0
                    

            if orientationId != -1:
                p.code = cO
                p.orientation = orientationId
                p.valid = True
                p.cornersCorrectOrder = [p.corners[id % 4] for id in range(orientationId, 4 + orientationId) ]
                #print(orientationId)
                #print(p.corners)
                #print(p.cornersCorrectOrder)

    def readRecogResult(self, patternFilePath, h5FilePath):
        self.readPatternFile(patternFilePath)
        self.readH5Codes(h5FilePath)

    def savePatternFile(self, patternFilePath):
        with open(patternFilePath, 'wt') as pf:
            for c in self.corners:
                pf.write('C ' + str(c.id) + ' ' + str(c.coordinates[0]) + ' ' + str(c.coordinates[1]) +'\n')
            for p in self.patterns:
                pf.write('P ' + str(p.id) + ' ' + ' '.join([str(i) for i in p.corners]) +'\n')
            pf.close()
    def saveH5Data(self, h5FilePath):
        with h5py.File(h5FilePath, 'w') as h5f:
                if self.subImgData is not None:
                    dataSet = h5f.require_dataset("data", self.subImgData.shape, dtype=np.float32)
                    dataSet[...] = self.subImgData
                h5f.attrs.create('all_orientation', data = 0, dtype=np.uint32)
                labelSet = h5f.require_dataset("label", (len(self.patterns),), dtype=np.uint8)
                labelSet[...] = [np.int8(1) for i in range(len(self.patterns))]
                codeSet = h5f.require_dataset("code", (len(self.patterns),), dtype='S2')
                codeSet[...] = [np.string_(p.code) for p in self.patterns]


