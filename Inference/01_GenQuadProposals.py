import os
QuadProposalCNNDir = r'QuadProposals/'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class PreprocessConfig:
    def __init__(self):
        self.CNMWeightsPath = './CornerKeys/2019_01_31_CNN_v03/CNN_2char_bn_unified.ckpt'
        self.method = 'component_search' # either 'component_search' or 'quad_proposal'

        self.qdCNNCfg = CNNQuadDetector.CNNQuadDetectorCfg()

        self.skipStep = 1
        self.skipPatternExtraction = False
        self.outputPatternRecogInfo = False
        self.outputPatternRecogInfoExt = 'png'
        self.outputPatternExtractionInfo = False
        self.outputBinaryImg = False
        self.outputCornersRecogInfo = False
        self.select = []
        self.startShift = 0
        self.useSoftMaxCotOff = False
        self.softMaxCutOffVal = 0.5
        self.extName = "pgm"
        self.outputPatternPixels = False
        self.outputCorners = False
        self.numProcess = 6
        # White Pattern Extranction Config
        self.minBlackComponentsSize = 180
        self.minWhiteComponentsSize = 15
        self.maxWhiteComponentsSize = 3000
        self.applyPatternFilter = True
        self.maxEdgeRatio = 2.8
        self.maxAngle = 0.75 * 3.1415926
        self.minAngle = 0.1 * 3.1415926
        # Erosion size control
        self.erosionSize = 2
        # Corner detection control
        self.findCornerOnBinarizedImg = True

        if self.findCornerOnBinarizedImg:
            self.minCornerDistance = 5
            self.blockSize = 5    # block size for goodFeaturesToTrack
            self.cornerQualityLevel = 0.2
            self.subPixRefineWindowSize = 5
        else:
            self.minCornerDistance = 5
            self.blockSize = 5    # block size for goodFeaturesToTrack
            self.cornerQualityLevel = 0.03
            self.subPixRefineWindowSize = 2

        self.pattern = ""
        self.patternPix = ""
        self.allcorner = ""
        self.patterncorner = ""
        self.binary = ""
        self.verbose = False

import sys
sys.path.insert(0, QuadProposalCNNDir)

from QuadProposals import CNNQuadDetector


if __name__ == '__main__':
    # inFile = r'Data/I04799.pgm'
    inFile = r'Data/04165.pgm'
    # inFile = r'Data/04559.pgm'

    os.makedirs('output', exist_ok=True)
    outPFile = r'output/OutQuadProposal.txt'
    outH5File = r'output/OutQuadProposal.h5'
    predictionFile = 'output/predictionFile.json'

    cfg = PreprocessConfig()
    cfg.outputPatternRecogInfo = 'output/Results.pdf'
    cfg.qdCNNCfg.cornerdet_sess = './_Nets/Cornerdet/20200105_13h57m_epoch_60.ckpt'
    cfg.qdCNNCfg.rejector_sess = './_Nets/Rejector/200117_rejector_321.ckpt'
    cfg.qdCNNCfg.recognizer_sess = './_Nets/Recognizer/CNN_108_gen12_auto00001.ckpt'
    cfg.qdCNNCfg.CIDFile = 'QuadProposals/CID/CID_list.txt'

    CNNModel = CNNQuadDetector.CNNQuadDetector(cfg.qdCNNCfg)
    CNNModel.restoreCNNSess()
    CNNQuadDetector.processSequenceQuadProposalSingleFile(inFile, outPFile, outH5File,
                                                          qDetector=CNNModel, config=cfg,
                                                          predictionFile=predictionFile)


