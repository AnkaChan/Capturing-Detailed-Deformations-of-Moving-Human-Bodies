import numpy as np
import time

# dic = "1234567ABCDEFGJKLMPQRTUVY" -- from Anka
suit_dict = {'1':0, '2':1, '3':2, '4':3, '5':4, '6':5, '7':6, 'A':7, 'B':8, 'C':9, 'D':10, 'E':11, 'F':12, 'G':13, 'J':14, 
           'K':15, 'L':16, 'M':17, 'P':18, 'Q':19, 'R':20, 'T':21, 'U':22, 'V':23, 'Y':24}
inv_suit_dict = {v: k for k, v in suit_dict.items()}

def cornerdet_inference(sess_cornerdet, crops, batch_size = 40000):
    """ runs corner detector on a bunch of crops, split into batches of size "batch_size",
    then puts the results back together
    """    
    #st_time = time.time()
    d4a = np.zeros((crops.shape[0], 1))
    d4b = np.zeros((crops.shape[0], 2))    
    for b in range(0, crops.shape[0], batch_size):
        b_end = min(b + batch_size, crops.shape[0])

        # inference_dict = {"cornerdet/cornerdet_imgs_ph:0":crops[b:b_end, :, :, :],
        #                   "cornerdet/cornerdet_pkeep_ph:0":1.0}
        # [d4a[b:b_end, :], d4b[b:b_end, :]] = sess_cornerdet.run(["cornerdet/cornerdet_out0:0", "cornerdet/cornerdet_out1:0"], inference_dict)

        inference_dict = {"cornerdet/imgs_ph:0":crops[b:b_end, :, :, :],
                          "cornerdet/pkeep_ph:0":1.0}
        [d4a[b:b_end, :], d4b[b:b_end, :]] = sess_cornerdet.run(["cornerdet/dens4a:0", "cornerdet/dens4b:0"], inference_dict)
    #print("cornerdet_inference:", time.time() - st_time)
    return (d4a, d4b)

def refine_corners(sess_cornerdet, corners_orig, img, crop_size = 20, margin_size = 6):
    """ corners_orig: num_corners x 2 ... original corner coordinates
        img: 2160 x 4000 x 3 ... greyscale image
        returns: num_corners x 2 ... refined 
    """
    # print("Refine Corner")
    rcorns = np.round(corners_orig).astype(np.int32)
    shifts_arr = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]])

    new_corner_list = []
    for cidx in range(rcorns.shape[0]):
        simg_list = []
        i_list = []
        j_list = []

        closetToBoundary = False
        for k in range(shifts_arr.shape[0]):
            im = rcorns[cidx,1] - int(crop_size/2) + shifts_arr[k, 1]
            jm = rcorns[cidx,0] - int(crop_size/2) + shifts_arr[k, 0]

            if im<0 or im+crop_size >= img.shape[0] or jm<0 or jm>=img.shape[1]:
                closetToBoundary = True
                break
            simg = img[im : im + crop_size, jm : jm + crop_size, 0]                
            simg_list.append(simg)
            i_list.append(im + margin_size)
            j_list.append(jm + margin_size)
        if closetToBoundary:
            continue
        simg_arr = np.array(simg_list).reshape(9, 20, 20, 1)
        (s_d4a, s_d4b) = cornerdet_inference(sess_cornerdet, simg_arr)

        s_corner_list = []
        for k in range(s_d4b.shape[0]):    
            s_corner_list.append((j_list[k] + 8*s_d4b[k,1], i_list[k] + 8*s_d4b[k,0]))

        s_corners = np.array(s_corner_list)
        new_corner = np.average(s_corners, 0)
        new_old_diff = new_corner - corners_orig[cidx]
        new_old_diff_norm = np.linalg.norm(new_old_diff)
        if (new_old_diff_norm > 1.5):
            print("too large diff norm: %f" % new_old_diff_norm)
        if (new_old_diff_norm > 3.0):
            # assert(false)
            print("Warning!! too large diff norm: %f" % new_old_diff_norm)
            #assert (False)
        else:
            new_corner_list.append(new_corner)
    return np.array(new_corner_list)

def refine_corners2(sess_cornerdet, corners_orig, img, crop_size = 20, margin_size = 6):
    """ corners_orig: num_corners x 2 ... original corner coordinates
        img: 2160 x 4000 x 3 ... greyscale image
        returns: num_corners x 2 ... refined
    """
    # print("Refine Corner")
    rcorns = np.round(corners_orig).astype(np.int32)
    shifts_arr = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]])
    numShifts = shifts_arr.shape[0]

    new_corner_list = []
    imgsList = []
    i_list_all = []
    j_list_all = []

    cids = []

    for cidx in range(rcorns.shape[0]):
        simg_list = []
        i_list = []
        j_list = []

        closetToBoundary = False
        for k in range(shifts_arr.shape[0]):
            im = rcorns[cidx,1] - int(crop_size/2) + shifts_arr[k, 1]
            jm = rcorns[cidx,0] - int(crop_size/2) + shifts_arr[k, 0]

            if im<0 or im+crop_size >= img.shape[0] or jm<0 or jm>=img.shape[1]:
                closetToBoundary = True
                break
            simg = img[im : im + crop_size, jm : jm + crop_size, 0]
            simg_list.append(simg)
            i_list.append(im + margin_size)
            j_list.append(jm + margin_size)


        if not closetToBoundary:
            imgsList.extend(simg_list)
            i_list_all.append(i_list)
            j_list_all.append(j_list)
            cids.append(cidx)

    assert not int(len(imgsList ) % numShifts)

    simg_arr = np.array(imgsList).reshape(-1, 20, 20, 1)
    (s_d4a_all, s_d4b_all) = cornerdet_inference(sess_cornerdet, simg_arr)

    for i in range(int(len(imgsList) / numShifts)):
        cidx = cids[i]
        j_list = j_list_all[i]
        i_list = i_list_all[i]
        s_corner_list = []

        s_d4b = s_d4b_all[i * numShifts:(i+1)*numShifts, :]

        for k in range(s_d4b.shape[0]):
            s_corner_list.append((j_list[k] + 8*s_d4b[k,1], i_list[k] + 8*s_d4b[k,0]))

        s_corners = np.array(s_corner_list)
        new_corner = np.average(s_corners, 0)
        new_old_diff = new_corner - corners_orig[cidx]
        new_old_diff_norm = np.linalg.norm(new_old_diff)
        if (new_old_diff_norm > 1.5):
            print("too large diff norm: %f" % new_old_diff_norm)
        if (new_old_diff_norm > 3.0):
            # assert(false)
            print("Warning!! too large diff norm: %f" % new_old_diff_norm)
            #assert (False)
        else:
            new_corner_list.append(new_corner)
    return np.array(new_corner_list)

def extract_corners_confids(d4a, d4b, i_list, j_list, confidence_threshold = -2):
    corner_list = []
    confid_list = []
    for k in range(d4a.shape[0]):
        if d4a[k] >= confidence_threshold:
            corner_list.append((j_list[k] + 8*d4b[k,1], i_list[k] + 8*d4b[k,0]))
            confid_list.append(d4a[k,0])
    corners = np.array(corner_list)
    confids = np.array(confid_list)
    return (corners, confids)

def run_rejector(sess_rejector, qp_imgs, batch_size = 6000):
    rejector_logits = np.zeros((qp_imgs.shape[0], 1))

    for b in range(0, qp_imgs.shape[0], batch_size):
        b_end = min(b + batch_size, qp_imgs.shape[0])
        rejector_dict = {"rejector/rejector_imgs_ph:0":qp_imgs[b:b_end,:,:,:], "rejector/rejector_pkeep_ph:0":1.0}
        rejector_logits[b:b_end] = sess_rejector.run("rejector/rejector_out:0", rejector_dict)

    return rejector_logits

def run_recognizer(sess_recognizer, imgs):
    recog_dict = {"recognizer/recognizer_imgs_ph:0":imgs, "recognizer/recognizer_pkeep_ph:0":1.0}

    [recog_dens4a, recog_dens4b] = sess_recognizer.run(["recognizer/recognizer_out_A:0", "recognizer/recognizer_out_B:0"], recog_dict)
    recog_predictions1 = np.argmax(recog_dens4a, 1)
    recog_predictions2 = np.argmax(recog_dens4b, 1)

    assert(recog_predictions1.shape[0] == recog_predictions2.shape[0])

    code_strs = []
    for i in range(recog_predictions1.shape[0]):
        c1 = inv_suit_dict[recog_predictions1[i]]
        c2 = inv_suit_dict[recog_predictions2[i]]
        code_strs.append(str(c1) + str(c2))
    return code_strs

def run_recognizer_with_prediction(sess_recognizer, imgs):
    recog_dict = {"recognizer/recognizer_imgs_ph:0":imgs, "recognizer/recognizer_pkeep_ph:0":1.0}

    [recog_dens4a, recog_dens4b] = sess_recognizer.run(["recognizer/recognizer_out_A:0", "recognizer/recognizer_out_B:0"], recog_dict)
    recog_predictions1 = np.argmax(recog_dens4a, 1)
    recog_predictions2 = np.argmax(recog_dens4b, 1)

    assert(recog_predictions1.shape[0] == recog_predictions2.shape[0])

    code_strs = []
    for i in range(recog_predictions1.shape[0]):
        c1 = inv_suit_dict[recog_predictions1[i]]
        c2 = inv_suit_dict[recog_predictions2[i]]
        code_strs.append(str(c1) + str(c2))
    return code_strs, recog_dens4a, recog_dens4b

def check_code_duplicates(code_strs, logf, err_set):
    codes_set = set()
    for item in iter(code_strs):
        if item in codes_set:
            print("Error: duplicate recognition of %s" % item, file=logf)
            err_set.add(item)
        codes_set.add(item)

def valid_codes_set(fname):
    """ returns set of tuples corresponding to valid codes in a CID file "fname"
        the tuples in the returned set are either 1-tuples (xxx,) or 2-tuples (xxx, xxx)
    """
    valid_codes = set()
    with open(fname, 'r') as f:
        for line in f:
            llist = line.split()
            if len(llist) == 2:
                valid_codes.add((llist[0],))
            elif len(llist) == 3:
                valid_codes.add((llist[0],))
                valid_codes.add((llist[1],))
                valid_codes.add((llist[0], llist[1]))
                valid_codes.add((llist[1], llist[0]))
            else:
                assert(False)        
    return valid_codes

def check_CIDs(code_strs, accept_qi, valid_codes, logf, err_set):
    corner_dict = {}
    for i, code in enumerate(code_strs):
        for j in range(4):
            corner_idx = accept_qi[i, j]
            if corner_idx not in corner_dict:
                corner_dict[corner_idx] = []
            corner_dict[corner_idx].append(code + str(j))     

    for corner_idx, code in corner_dict.items():
        if tuple(code) not in valid_codes:
            print("Error: wrong CID %s" % str(tuple(code)), file=logf)
            for it in iter(code):
                err_set.add(it[:2])