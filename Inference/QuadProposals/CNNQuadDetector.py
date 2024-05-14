# this version accepts all images, both good and bad, the bad ones have labels -1, -1
# this notebook has just quick CNN training

PatternFileIODir = r'../../04_PipeLine/GenerateModelSequenceMesh7/CornerKeys/'

import sys
sys.path.insert(0, PatternFileIODir)

import ReadPatternFile

import tensorflow as tf
import numpy as np
import json
#%matplotlib notebook
import matplotlib.pyplot as plt
import time
import cv2
import math
import itertools
import quadprops as qps
import imganalysis as ia
import importlib
import sys
import os

from pathlib import Path
import glob
from multiprocessing import Pool

class QGConfig:
    def __init__(self):
        self.numProcess = 8
        self.subImgW = 104
        self.subImgH = 104
        self.sugImgPadSize = 20
        self.interpMethod = 0
        self.darkThreshold = 35

def genwWarps(args):
    qv_img, img = args
    wimg = qps.warped_subimage(img, qv_img)
    return wimg

class CNNQuadDetectorCfg:
    def __init__(self):
        self.qgCfg = QGConfig()
        self.cornerdet_sess = './CornerDectector2/20200105_13h57m.ckpt'
        #self.cornerdet_meta = self.cornerdet_sess + '.meta'
        self.rejector_sess = './Rejector2/20200114_14h07m.ckpt'
        #self.rejector_meta = self.rejector_sess + '.meta'
        # self.recognizer_sess = './nets_recognizer/recognizer_83_renamed.ckpt'
        self.recognizer_sess = './nets_recognizer/CNN_100_gen7_and_synth.ckpt'
        #self.recognizer_meta = self.recognizer_sess + '.meta'
        self.CIDFile = 'QuadProposals/CID/CID_list.txt'
        self.num_processes = 1
        self.corners_PDF = False
        self.corners_PDF_file = ''
        self.quads_PDF = False
        self.quads_PDF_file = ''
        self.quad_props_PDF = False
        self.quad_props_PDF_file = ''
        self.dump_suspect_cnt = False
        self.saveSubImgs = True
        self.showTimeConsumption = False
        self.preRejectCriteria =  {
            'numDarkRange':(0, 1650),
            'stdDevRange':(0, 65),
            'intensityRange':(32, 175)
        }


class CNNQuadDetector:
    def __init__(self, cfg=CNNQuadDetectorCfg()):
        self.cfg = cfg
        self.img = None
        self.imgFile = None
        self.sess_cornerdet = None
        self.sess_rejector = None
        self.sess_recognizer = None
        self.logf = sys.stdout
        self.corners = None 
        self.confids = None
        self.quadProposals = None
        self.img_list = []
        self.qv_list = [] # (proposed) quad vertices
        self.qi_list = [] # (proposed) quad indices
        self.suspect_cnt = 0
    def restoreCNNSess(self):
        tf.reset_default_graph()

        saver = tf.train.import_meta_graph(self.cfg.cornerdet_sess + '.meta', import_scope="cornerdet")
        self.sess_cornerdet = tf.Session()
        saver.restore(self.sess_cornerdet, self.cfg.cornerdet_sess)

        saver2 = tf.train.import_meta_graph(self.cfg.rejector_sess + '.meta', import_scope="rejector")
        self.sess_rejector = tf.Session()
        saver2.restore(self.sess_rejector, self.cfg.rejector_sess)

        saver3 = tf.train.import_meta_graph(self.cfg.recognizer_sess + '.meta', import_scope="recognizer")
        self.sess_recognizer = tf.Session()
        saver3.restore(self.sess_recognizer, self.cfg.recognizer_sess)

    def dump_err_imgs(self, code_strs, err_codes, accept_imgs, suspect_cnt):
        for i, code in enumerate(code_strs):
            if code in err_codes:
                fname = "out/err/%06i" % suspect_cnt
                cv2.imwrite(fname + ".pgm", accept_imgs[i,:,:,0])
                with open(fname + ".txt", "w") as f:
                    print(code, file=f)
                suspect_cnt += 1
        return suspect_cnt

    def vis_data(self, imgs, labels = None, offset = 0):
        plt.figure(figsize=(16,12))
        for i in range(8*11):
            plt.subplot(8,11,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            if offset + i < imgs.shape[0]:
                plt.imshow(imgs[offset + i, :, :, 0], cmap=plt.cm.gray)
                if labels is not None:
                    plt.xlabel("%s" % (labels[offset + i]))

    def preAccept(self, qStatistics, cfg):
        preAccepted_ids = []
        preRejected_ids = []

        for i in range(qStatistics.shape[0]):
            numDark = qStatistics[i, 2]
            stdDev = qStatistics[i, 1]
            avgIntensity = qStatistics[i, 0]
            if numDark < cfg['numDarkRange'][0] or numDark > cfg['numDarkRange'][1]:
                preRejected_ids.append(i)
            elif stdDev < cfg['stdDevRange'][0] or stdDev > cfg['stdDevRange'][1]:
                preRejected_ids.append(i)
            elif avgIntensity < cfg['intensityRange'][0] or avgIntensity > cfg['intensityRange'][1]:
                preRejected_ids.append(i)
            else:
                preAccepted_ids.append(i)
        return np.array(preAccepted_ids), np.array(preRejected_ids)

    def process2(self):
        st_time = time.time()
        img = self.img
        print(img.shape, file=self.logf)

        # ******************************************************
        # CNN Corner Detection
        start = time.clock()
        crops, i_list, j_list = qps.gen_crops(img)

        (d4a, d4b) = ia.cornerdet_inference(self.sess_cornerdet, crops)
        if self.cfg.showTimeConsumption:
            print('Time consumption in corner detection CNN', time.clock() - start, file=self.logf)

        assert (d4a.shape[0] == d4b.shape[0] and len(i_list) == d4a.shape[0] and len(j_list) == d4a.shape[0])

        start = time.clock()
        self.corners, self.confids = ia.extract_corners_confids(d4a, d4b, i_list, j_list)

        self.corners, self.confids = qps.cluster_points_with_confidences2(self.corners, self.confids)
        if self.cfg.showTimeConsumption:
            print('Time consumption in corner clustering', time.clock() - start, file=self.logf)

        start = time.clock()
        self.corners = ia.refine_corners2(self.sess_cornerdet, self.corners, img)
        if self.cfg.showTimeConsumption:
            print('Time consumption in corner refining', time.clock() - start, file=self.logf)


        if self.cfg.corners_PDF:
            fig, ax = plt.subplots()
            ax.imshow(img, vmin=0, vmax=255, interpolation='nearest')
            ax.axis('off')
            ax.plot(self.corners[:, 0], self.corners[:, 1], 'x', color='red', markeredgewidth=0.06, markersize=1)
            fig.savefig(self.cfg.corners_PDF_file, dpi=2000, transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close()

        # ******************************************************
        # Generating Quad Proposal
        start = time.clock()
        # min_dist, _, _ = qps.get_min_pair(self.corners)

        hull_list = qps.quad_proposals(self.corners, min_area=0, min_edge_len=12)
        self.quadProposals = qps.hull_list_to_qp(hull_list)
        if self.cfg.showTimeConsumption:
            print('Time consumption in generating quad proposal hull list', time.clock() - start)

        start = time.clock()
        # print("min_dist = %.2f, num_qps = %i" % (min_dist, len(self.quadProposals)), file=self.logf)
        # if min_dist < 5:
        #     print("Warning: min_dist too low", file=self.logf)
        #     self.logf.flush()

        if len(self.quadProposals) == 0:
            print("empty qp, quitting", file=self.logf)
            # self.logf.flush()
            return

        self.qv_list = []  # (proposed) quad vertices
        self.qi_list = []  # (proposed) quad indices

        for _, item in enumerate(self.quadProposals):
            qv_img = self.corners[item, :]
            self.qi_list.append(item)
            self.qv_list.append(qv_img)

        self.rList = QuadGeneratorCPP.genQP(self.imgFile, [qvs.tolist() for qvs in self.qv_list], self.cfg.qgCfg)

        self.img_list = self.rList[0][:,:,:,np.newaxis]
        self.quadStatistics = self.rList[1]

        qi_arr = np.array(self.qi_list)
        qv_arr = np.array(self.qv_list)
        qp_imgs = self.img_list
        if self.cfg.showTimeConsumption:
            print('Time consumption in warping quad proposal images', time.clock() - start)
        # sys.stdout.flush()

        if self.cfg.quad_props_PDF:
            fig, ax = plt.subplots()
            ax.imshow(img, vmin=0, vmax=255, interpolation='nearest')
            for i in range(qv_arr.shape[0]):
                qpoints = qv_arr[i, :, :]
                qps.draw_quad(ax, qpoints)
            fig.savefig(self.cfg.quad_props_PDF_file, dpi=2000)
            plt.close()
        # ******************************************************
        # Reting Quad Proposal
        start = time.clock()

        self.preAcceptedIds, self.preRejectedIds = self.preAccept(self.quadStatistics, self.cfg.preRejectCriteria)

        if len(self.preAcceptedIds) != 0:
            preAcceptedImgs = qp_imgs[self.preAcceptedIds, :,:,:]

            rejector_logits = ia.run_rejector(self.sess_rejector, preAcceptedImgs)
            qp_accepted = rejector_logits > 0
            self.accept_indices_inPreAccept = np.flatnonzero(qp_accepted)
            self.accept_indices = self.preAcceptedIds[self.accept_indices_inPreAccept]

            self.accept_imgs = qp_imgs[self.accept_indices, :, :, :]
            self.accept_qv = qv_arr[self.accept_indices, :, :]
            self.accept_qi = qi_arr[self.accept_indices, :]

            self.code_strs, self.recog_dens4a, self.recog_dens4b \
                = ia.run_recognizer_with_prediction(self.sess_recognizer, self.accept_imgs)
        else:
            self.accept_indices = np.array([])

            self.accept_imgs = np.array([])
            self.accept_qv = np.array([])
            self.accept_qi = np.array([])

            self.code_strs, self.recog_dens4a, self.recog_dens4b \
                = [], [], []
            print("All quads has been rejected for: ", self.imgFile)

        # The checking basically does nothing other than giving warnings, so I deleted it

        # print(time.time() - st_time, file=self.logf)
        # assert (self.accept_qi.shape[0] == len(self.code_strs))
        # self.err_codes = set()
        # ia.check_code_duplicates(self.code_strs, self.logf, self.err_codes)
        # self.valid_codes = ia.valid_codes_set(self.cfg.CIDFile)
        # ia.check_CIDs(self.code_strs, self.accept_qi, self.valid_codes, self.logf, self.err_codes)
        # print(err_codes, file=self.logf)
        # if self.cfg.dump_suspect_cnt:
        #     suspect_cnt = self.dump_err_imgs(code_strs, err_codes, self.accept_imgs, self.suspect_cnt)
        if self.cfg.showTimeConsumption:
            print('Time consumption in rejecting quad proposal and recognition', time.clock() - start)

        if self.cfg.quads_PDF:
            fig, ax = plt.subplots()
            ax.imshow(img, vmin=0, vmax=255, interpolation='nearest')
            for i in range(len(self.code_strs)):
                qpoints = self.accept_qv[i, :, :]
                qps.draw_quad(ax, qpoints)
                ax.text(np.mean(qpoints[:, 0]), np.mean(qpoints[:, 1]), self.code_strs[i], \
                        verticalalignment='center', horizontalalignment='center', fontsize=1,
                        fontweight='ultralight', color='yellow', alpha=0.75)
            fig.savefig(self.cfg.quads_PDF_file, dpi=2000)
            fig.clf()
            ax.cla()
            plt.close('all')
            del fig
            del ax

    def process(self):
        img = self.img
        print(img.shape, file=self.logf)

        #******************************************************
        #CNN Corner Detection
        crops, i_list, j_list = qps.gen_crops(img)
        (d4a, d4b) = ia.cornerdet_inference(self.sess_cornerdet, crops)
        assert(d4a.shape[0] == d4b.shape[0] and len(i_list) == d4a.shape[0] and len(j_list) == d4a.shape[0])

        self.corners, self.confids = ia.extract_corners_confids(d4a, d4b, i_list, j_list)
        self.corners, self.confids = qps.cluster_points_with_confidences( self.corners, self.confids)

        self.corners = ia.refine_corners(self.sess_cornerdet, self.corners, img)

        if self.cfg.corners_PDF:
            fig, ax = plt.subplots()
            ax.imshow(img, vmin=0, vmax=255, interpolation = 'nearest')
            ax.axis('off')
            ax.plot(self.corners[:,0], self.corners[:,1], 'x', color='red', markeredgewidth = 0.06, markersize=1)    
            fig.savefig(self.cfg.corners_PDF_file, dpi = 2000,  transparent = True, bbox_inches = 'tight', pad_inches = 0)
            plt.close()

        #******************************************************
        #Generating Quad Proposal
        start = time.clock()
        # min_dist, _, _ = qps.get_min_pair(self.corners)
    
        hull_list = qps.quad_proposals(self.corners, min_area = 0, min_edge_len=12)
        self.quadProposals = qps.hull_list_to_qp(hull_list)

        start = time.clock()
        # print("min_dist = %.2f, num_qps = %i" % (min_dist, len(self.quadProposals)), file=self.logf)
        # if min_dist < 5:
        #     print("Warning: min_dist too low", file=self.logf)
        #     self.logf.flush()
        
        if len(self.quadProposals) == 0:
            print("empty qp, quitting", file=self.logf)
            self.logf.flush()
            return
    
        self.img_list = []
        self.qv_list = [] # (proposed) quad vertices
        self.qi_list = [] # (proposed) quad indices

        if self.cfg.num_processes == 1:
            for _, item in enumerate(self.quadProposals):
                qv_img = self.corners[item, :]
                wimg = qps.warped_subimage(img, qv_img)

                self.qi_list.append(item)
                self.img_list.append(wimg)
                self.qv_list.append(qv_img)
        else:
            print("Multi processing is not available!")
            exit()
        #     conv_args = []
        #     print("Gen warped subimages by multi processing")
        #     # sys.stdout.flush()
        #
        #     for _, item in enumerate(self.quadProposals):
        #         self.qi_list.append(item)
        #         qv_img = self.corners[item, :]
        #         self.qv_list.append(qv_img)
        #
        #         conv_args.append(
        #             (qv_img, img))
        #     # # print(conv_args)
        #     # start = time.time()
        #     with Pool(self.cfg.num_processes) as p:
        #         self.img_list = p.map(genwWarps, conv_args)

        qi_arr = np.array(self.qi_list)
        qv_arr = np.array(self.qv_list)
        qp_imgs = np.array(self.img_list)
        # sys.stdout.flush()
        
        if self.cfg.quad_props_PDF:
            fig, ax = plt.subplots()
            ax.imshow(img, vmin=0, vmax=255, interpolation = 'nearest')
            for i in range(qv_arr.shape[0]):
                qpoints = qv_arr[i, :, :]            
                qps.draw_quad(ax, qpoints)
            fig.savefig(self.cfg.quad_props_PDF_file, dpi = 2000)        
            plt.close()
        #******************************************************
        #Reting Quad Proposal
        start = time.clock()
        rejector_logits = ia.run_rejector(self.sess_rejector, qp_imgs)
        qp_accepted = rejector_logits > 0
        self.accept_indices = np.flatnonzero(qp_accepted)
        self.accept_imgs = qp_imgs[self.accept_indices, :, :, :]
        self.accept_qv = qv_arr[self.accept_indices, :, :]
        self.accept_qi = qi_arr[self.accept_indices, :]

        self.code_strs, self.recog_dens4a, self.recog_dens4b \
            = ia.run_recognizer_with_prediction(self.sess_recognizer, self.accept_imgs)
        assert(self.accept_qi.shape[0] == len(self.code_strs))
        self.err_codes = set()
        # ia.check_code_duplicates(self.code_strs, self.logf, self.err_codes)
        self.valid_codes = ia.valid_codes_set(self.cfg.CIDFile)
        # ia.check_CIDs(self.code_strs, self.accept_qi, self.valid_codes, self.logf, self.err_codes)
        #print(err_codes, file=self.logf)
        # if self.cfg.dump_suspect_cnt:
        #     suspect_cnt = self.dump_err_imgs(code_strs, err_codes, self.accept_imgs, self.suspect_cnt)

        if self.cfg.quads_PDF:
            fig, ax = plt.subplots()
            ax.imshow(img, vmin=0, vmax=255, interpolation = 'nearest')
            for i in range(len(self.code_strs)):
                qpoints = self.accept_qv[i, :, :]            
                qps.draw_quad(ax, qpoints)
                ax.text(np.mean(qpoints[:,0]), np.mean(qpoints[:,1]), self.code_strs[i], \
                        verticalalignment='center', horizontalalignment='center', fontsize=1, 
                        fontweight='ultralight', color='yellow', alpha=0.75)
            fig.savefig(self.cfg.quads_PDF_file, dpi = 2000)
            fig.clf()
            ax.cla()
            plt.close('all')
            del fig
            del ax

    def resultsToUtd(self, allQuadProposals = False):
        utd = ReadPatternFile.Unitard()
        utd.corners = [ReadPatternFile.Corners(i, pts) for i, pts in enumerate(self.corners)]
        if not allQuadProposals:
            utd.patterns = [
                ReadPatternFile.Pattern(corners = corners, cornersCorrectOrder = corners, valid = True, code=code, id = i)
                for i, corners, code in zip(range(len(self.accept_qi)), self.accept_qi, self.code_strs)        
            ]
        else:
            pass
        if self.cfg.saveSubImgs:
            utd.subImgData = np.squeeze(self.accept_imgs)
        else:
            utd.subImgData = np.zeros((self.accept_imgs.shape[0],1,1), dtype=self.accept_imgs.dtype)
        return utd

class ProcessSqQuadPoposalCfg:
    def __init__(self):
        self.qdCNNCfg = CNNQuadDetectorCfg()
        self.skipStep = 1
        self.select = []
        self.extName = "pgm"
        self.outputPatternRecogInfo = False
        self.outputCornersRecogInfo = False
        self.saveSubImgs = True

def getIterRange(imgFiles, config):
    if len(config.select) == 0:
        itRange = range(int(len(imgFiles) / config.skipStep))
    else:
        if (config.select[1] > len(imgFiles)):
            config.select[1] = len(imgFiles)
        itRange = [config.select[0] + config.startShift + i * config.skipStep for i in range(int((config.select[1] - config.startShift - config.select[0])/config.skipStep))]
    return itRange

def processSequenceQuadProposalSingleFile(inFile, patternFile, h5File, qDetector = None, config = ProcessSqQuadPoposalCfg(), predictionFile=None):

    if qDetector is None:
        qDetector = CNNQuadDetector()
        qDetector.restoreCNNSess()

    qDetector.cfg = config.qdCNNCfg
    qDetector.imgFile = inFile

    print("Processing: ", inFile)
    imgFilePath = Path(inFile)
    patternPath = Path(patternFile)
    # path = str(imgFilePath.parent)
    if config.outputPatternRecogInfo:
        config.qdCNNCfg.quads_PDF = True
        config.qdCNNCfg.quads_PDF_file = str(patternPath.parent) + "/" + patternPath.stem + "Quads.pdf"

    if config.outputCornersRecogInfo:
        config.qdCNNCfg.corners_PDF = True
        config.qdCNNCfg.corners_PDF_file = str(patternPath.parent) + "/" + patternPath.stem + ".png"

    qDetector.cfg = config.qdCNNCfg
    qDetector.img = cv2.imread(inFile)
    # qDetector.process2()
    qDetector.process()
    utd = qDetector.resultsToUtd()
    utd.savePatternFile(patternFile)
    utd.saveH5Data(h5File)

    if predictionFile is not None:
        json.dump({
            'corners':qDetector.corners.tolist(),
            'code':qDetector.code_strs,
            'accept_qi':qDetector.accept_qi.tolist(),
            'accept_qv':qDetector.accept_qv.tolist(),
            'recog_dens4a':qDetector.recog_dens4a.tolist(),
            'recog_dens4b':qDetector.recog_dens4b.tolist()
        }, open(predictionFile, 'w'), indent=2)


def processSequenceQuadProposal(inFolder, patternPath, qDetector = None, config = ProcessSqQuadPoposalCfg()):
    os.makedirs(patternPath, exist_ok=True)
    if qDetector is None:
        qDetector = CNNQuadDetector()
        qDetector.restoreCNNSess()

    qDetector.cfg = config.qdCNNCfg

    imgFiles = glob.glob(inFolder + "/*." + config.extName)
    imgFiles.sort()

    itRange = getIterRange(imgFiles,config)

    for i in itRange:
        fileId = i
        fileName = imgFiles[fileId]
        print("Processing: ", fileName)
        imgFilePath = Path(fileName)
        #path = str(imgFilePath.parent)
        name = imgFilePath.stem
        patternH5File = patternPath + "/" + name + ".h5"
        patternFile = patternPath + "/" + name + ".txt"
        if config.outputPatternRecogInfo:
            config.qdCNNCfg.quads_PDF = True
            config.qdCNNCfg.quads_PDF_file = patternPath + "/" + name + ".pdf"

        if config.outputCornersRecogInfo:
            config.qdCNNCfg.corners_PDF = True
            config.qdCNNCfg.corners_PDF_file = patternPath + "/" + name + "Corners" + ".png"
        
        qDetector.cfg = config.qdCNNCfg
        qDetector.img = cv2.imread(fileName)
        qDetector.process()
        utd = qDetector.resultsToUtd()
        utd.savePatternFile(patternFile)
        utd.saveH5Data(patternH5File)






