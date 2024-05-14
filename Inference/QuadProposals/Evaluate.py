# this version accepts all images, both good and bad, the bad ones have labels -1, -1
# this notebook has just quick CNN training
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

if __name__ == '__main__':

    importlib.reload(qps)
    importlib.reload(ia)

    tf.reset_default_graph()

    saver = tf.train.import_meta_graph('./nets/28_renamed.ckpt.meta', import_scope="cornerdet")
    sess_cornerdet = tf.Session()
    saver.restore(sess_cornerdet, './nets/28_renamed.ckpt')

    #rejector_name = './nets_rejector/67_rejector_better_converged_04.ckpt'
    rejector_name = './nets_rejector/75_rejector_v04.ckpt'
    saver2 = tf.train.import_meta_graph(rejector_name + '.meta', import_scope="rejector")
    sess_rejector = tf.Session()
    saver2.restore(sess_rejector, rejector_name)

    recognizer_fname = "./nets_recognizer/recognizer_83_renamed.ckpt"
    saver3 = tf.train.import_meta_graph(recognizer_fname + ".meta", import_scope="recognizer")
    sess_recognizer = tf.Session()
    saver3.restore(sess_recognizer, recognizer_fname)

    def dump_err_imgs(code_strs, err_codes, suspect_cnt):
        suspect_cnt = 0
        for i, code in enumerate(code_strs):
            if code in err_codes:
                fname = "out/err/%06i" % suspect_cnt
                cv2.imwrite(fname + ".pgm", accept_imgs[i,:,:,0])
                with open(fname + ".txt", "w") as f:
                    print(code, file=f)
                suspect_cnt += 1
        return suspect_cnt

    folder_name = r"F:\WorkingCopy2\2019_04_16_8CamsCapture\Converted\D" + '\\'
    #out_folder_name = r"e:\2019_06_18_out\A" + "\\"
    out_folder_name = r"out/"

    os.makedirs(out_folder_name, exist_ok=True)
    #logf = open("out/log028_A.txt", "w")
    logf = sys.stdout

    valid_codes = ia.valid_codes_set('CID/CID_list.txt')
    corners_PDF = True
    quads_PDF = True
    quad_props_PDF = False
    suspect_cnt = 0
    dbg_img = 2627
    #for img_num in range(1350, 3500):
    for img_num in range(dbg_img, dbg_img + 1):
        fname = "%s%05i.pgm" % (folder_name, img_num)
        print(file=logf)
        print(fname, file=logf)

        st_time = time.time()
        img = cv2.imread(fname)    
        print(img.shape, file=logf)

        #******************************************************
        #CNN Corner Detection
        start = time.process_time()
        crops, i_list, j_list = qps.gen_crops(img)
  
        (d4a, d4b) = ia.cornerdet_inference(sess_cornerdet, crops)
        assert(d4a.shape[0] == d4b.shape[0] and len(i_list) == d4a.shape[0] and len(j_list) == d4a.shape[0])

        corners, confids = ia.extract_corners_confids(d4a, d4b, i_list, j_list)
        corners, confids = qps.cluster_points_with_confidences(corners, confids)
        print('Time consumption in corner detection', time.process_time() - start)

        if corners_PDF:
            fig, ax = plt.subplots()
            ax.imshow(img, vmin=0, vmax=255, interpolation = 'nearest')
            ax.plot(corners[:,0], corners[:,1], 'x', color='red', markeredgewidth = 0.06, markersize=1)    
            fig.savefig(out_folder_name + "%05i_corners.pdf" % img_num, dpi = 2000)
            plt.close()

        #******************************************************
        #Generating Quad Proposal
        start = time.process_time()
        min_dist, _, _ = qps.get_min_pair(corners)    
    
        hull_list = qps.quad_proposals(corners, min_area = 0, min_edge_len=12)
        qp = qps.hull_list_to_qp(hull_list)
    
        print("min_dist = %.2f, num_qps = %i" % (min_dist, len(qp)), file=logf)
        if min_dist < 5:
            print("Warning: min_dist too low", file=logf)
        logf.flush()
        
        if len(qp) == 0:
            print("empty qp, quitting", file=logf)
            logf.flush()
            continue
    
        img_list = []
        qv_list = [] # (proposed) quad vertices
        qi_list = [] # (proposed) quad indices
        for _, item in enumerate(qp):
            qv_img = corners[item, :]    
            wimg = qps.warped_subimage(img, qv_img)        

            qi_list.append(item)
            img_list.append(wimg)
            qv_list.append(qv_img)

        qi_arr = np.array(qi_list)
        qv_arr = np.array(qv_list)
        qp_imgs = np.array(img_list)
        print('Time consumption in generating quad proposal', time.process_time() - start)
        
        if quad_props_PDF:
            fig, ax = plt.subplots()
            ax.imshow(img, vmin=0, vmax=255, interpolation = 'nearest')
            for i in range(qv_arr.shape[0]):
                qpoints = qv_arr[i, :, :]            
                qps.draw_quad(ax, qpoints)
            fig.savefig(out_folder_name + "%05i_quad_props.pdf" % img_num, dpi = 2000)        
            plt.close()
        #******************************************************
        #Reting Quad Proposal
        start = time.process_time()
        rejector_logits = ia.run_rejector(sess_rejector, qp_imgs)
        qp_accepted = rejector_logits > 0
        accept_indices = np.flatnonzero(qp_accepted)
        accept_imgs = qp_imgs[accept_indices, :, :, :]
        accept_qv = qv_arr[accept_indices, :, :]
        accept_qi = qi_arr[accept_indices, :]
    
        code_strs = ia.run_recognizer(sess_recognizer, accept_imgs)
        print(time.time() - st_time, file=logf)
        assert(accept_qi.shape[0] == len(code_strs))
        err_codes = set()
        ia.check_code_duplicates(code_strs, logf, err_codes)
        ia.check_CIDs(code_strs, accept_qi, valid_codes, logf, err_codes)
        #print(err_codes, file=logf)
        suspect_cnt = dump_err_imgs(code_strs, err_codes, suspect_cnt)
        print('Time consumption in rejecting quad proposal', time.process_time() - start)
    
        if quads_PDF:
            fig, ax = plt.subplots()
            ax.imshow(img, vmin=0, vmax=255, interpolation = 'nearest')
            for i in range(len(code_strs)):
                qpoints = accept_qv[i, :, :]            
                qps.draw_quad(ax, qpoints)
                ax.text(np.mean(qpoints[:,0]), np.mean(qpoints[:,1]), code_strs[i], \
                        verticalalignment='center', horizontalalignment='center', fontsize=1, 
                        fontweight='ultralight', color='yellow', alpha=0.75)
            fig.savefig(out_folder_name + "%05i_quads.pdf" % img_num, dpi = 2000)
            fig.clf()
            ax.cla()
            plt.close('all')
            del fig
            del ax
    
        logf.flush()

        def vis_data(imgs, labels = None, offset = 0):
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