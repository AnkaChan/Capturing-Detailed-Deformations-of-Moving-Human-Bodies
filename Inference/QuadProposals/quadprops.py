import numpy as np
#import scipy.spatial
import math
import itertools
import cv2

from sklearn.neighbors import NearestNeighbors

def corners_inside(i_min, i_max, j_min, j_max, verts):
    """ returns a vector of indices of wsquares that are within the ij bounding box"""
    within_i = np.logical_and(verts[:,1] >= i_min, verts[:,1] <= i_max)
    within_j = np.logical_and(verts[:,0] >= j_min, verts[:,0] <= j_max)
    within_ij = np.logical_and(within_i, within_j)
    return np.flatnonzero(within_ij)

def get_min_pair(corners):
    """ corners ... Nx2 numpy array of N points
        task: finds closest two points
        returns (distance between the closest two points, index of the first, index of the second)
    """
    min_norm = np.inf
    min_pair = None
    if corners.shape[0] <= 1:
        return (min_norm, None, None)
    for i in range(corners.shape[0]):
        norms = np.linalg.norm(corners - corners[i, :], axis=1)
        norms[i] = np.inf
        mn = np.min(norms)
        if mn < min_norm:
            min_norm = mn
            min_pair = (i, np.argmin(norms))
    return (min_norm, min_pair[0], min_pair[1])

def cluster_points_with_confidences(corners, confids, close_distance_threshold = 3):
    """ corners ... Nx2 numpy array of N 2D points
        confids ... Nx1 array of confidences of the points
        task: if any two corners are too close (< close_distance_threshold), discard the one 
        with lower confidence value; repeat until all pairs >= close_distance_threshold
    """
    if corners.shape[0] <= 1:
        return (corners, confids)
    while True:
        (min_pair_dist, i0, i1) = get_min_pair(corners)
        #print(min_pair_dist, i0, i1, confids[i0], confids[i1])
        if min_pair_dist >= close_distance_threshold:
            break

        if confids[i0] < confids[i1]:
            del_idx = i0
        else:
            del_idx = i1

        corners = np.delete(corners, del_idx, axis=0)
        confids = np.delete(confids, del_idx, axis=0)
    return (corners, confids)

def cluster_points_with_confidences2(corners, confids, close_distance_threshold = 3):
    """ corners ... Nx2 numpy array of N 2D points
        confids ... Nx1 array of confidences of the points
        task: if any two corners are too close (< close_distance_threshold), discard the one
        with lower confidence value; repeat until all pairs >= close_distance_threshold
    """
    if corners.shape[0] <= 1:
        return (corners, confids)

    while True:

        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(corners)
        distances, indices = nbrs.kneighbors(corners)
        if np.min(distances[:,1]) >= close_distance_threshold:
            break

        del_ids = set()

        for i in np.where(distances[:,1] < close_distance_threshold)[0]:
            i0 = i
            i1 = indices[i, 1]
            if confids[i0] < confids[i1]:
                del_ids.add(i0)
            else:
                del_ids.add(i1)

        corners = np.delete(corners, list(del_ids), axis=0)
        confids = np.delete(confids, list(del_ids), axis=0)
    return (corners, confids)

def draw_quad(ax, points):
    arrow_params = {'linewidth':0.05, 'head_width':5, 'length_includes_head':True, 'overhang': 0.5}
    ax.arrow(points[0,0], points[0,1], points[1,0] - points[0,0], points[1,1] - points[0,1], color='red', **arrow_params) 
    ax.arrow(points[1,0], points[1,1], points[2,0] - points[1,0], points[2,1] - points[1,1], color='green', **arrow_params) 
    ax.arrow(points[2,0], points[2,1], points[3,0] - points[2,0], points[3,1] - points[2,1], color='blue', **arrow_params) 
    ax.arrow(points[3,0], points[3,1], points[0,0] - points[3,0], points[0,1] - points[3,1], color='yellow', **arrow_params) 

def angles_between_vecs_batch(v0, v1):
    """ input: v0 and v1 are Nx2 numpy arrays of N 2D vectors (batch)
        output: Nx1 array of angles (unsigned)
    """
    return np.rad2deg(np.arccos(np.clip((v0 * v1).sum(axis=1) / (np.linalg.norm(v0, axis=1) * np.linalg.norm(v1, axis=1)), -1, 1)))

def angles_quad_batch(points):
    """ input: 4 x N x 2 tensor of N 2D quads
        output: N x 4 matrix of internal quad angles (unsigned)
    """
    a0 = angles_between_vecs_batch(points[:,1,:] - points[:,0,:], points[:,3,:] - points[:,0,:])
    a1 = angles_between_vecs_batch(points[:,2,:] - points[:,1,:], points[:,0,:] - points[:,1,:])
    a2 = angles_between_vecs_batch(points[:,1,:] - points[:,2,:], points[:,3,:] - points[:,2,:])
    a3 = angles_between_vecs_batch(points[:,0,:] - points[:,3,:], points[:,2,:] - points[:,3,:])
    return np.array([a0, a1, a2, a3]).transpose()

def edgelens_quad_batch(points):
    """ input: 4 x N x 2 tensor of N 2D quads
        output: N x 4 matrix of quad edges
    """
    e1 = np.linalg.norm(points[:,1,:] - points[:,0,:], axis=1)
    e2 = np.linalg.norm(points[:,2,:] - points[:,1,:], axis=1)
    e3 = np.linalg.norm(points[:,3,:] - points[:,2,:], axis=1)
    e4 = np.linalg.norm(points[:,0,:] - points[:,3,:], axis=1)
    return np.array([e1, e2, e3, e4]).transpose()

def quad_proposals(quad_verts, max_width = 60, min_area = 450, min_edge_len = 15, max_edge_len = 55, min_angle = 45, max_angle = 135):
    """ quad_verts ... N x 2 numpy array of 2D coordinates of N points
        task: generate a set quads ("quad proposals") connecting vertices in quad_verts
        max_width ... [2*max_width, max_width] is the size of crop window to generate initial (unordered) four-sets of points (which will be later connected into quad proposals)
        min/max_edge_length, min/max_angle ... controls which quad proposals will be discarded
        returns: a list of four-tuples (ordered), each four-tuple contains four indices into quad_verts
    """
    # four-point set proposals (unordered, just sets):
    fourtuple_props = set()
    for i in range(quad_verts.shape[0]):
        v0 = quad_verts[i, :]
        cset = set(corners_inside(v0[1], v0[1] + max_width, v0[0] - max_width, v0[0] + max_width, quad_verts))
        cset.remove(i)
        for sub3 in itertools.combinations(cset, 3):
            sub3 += (i,)
            fourtuple_props.add(frozenset(sub3))

    # comput convex hulls of the four-point sets:
    hi_list = []
    rp_list = []
    for idx, item in enumerate(fourtuple_props):
        vert_indices = list(item)
        points = quad_verts[vert_indices, :]    
        hull = cv2.convexHull(points.astype(np.float32), returnPoints = False)
        
        if len(hull) <= 3:
            continue
        assert(len(hull) == 4)
        
        hull_indices = np.array(vert_indices)[hull[:,0]]    
            
        reorder_points = quad_verts[hull_indices, :]
        area = cv2.contourArea(reorder_points.astype(np.float32), True)
        assert(area >= 0) # otherwise wrong orientation
        if area >= min_area:
            rp_list.append(reorder_points)
            hi_list.append(hull_indices)  
        
    # batch filtering of quads whose angles or edges are out of the bounds:
    rp = np.array(rp_list)
    hi = np.array(hi_list)

    if rp.shape[0] == 0:
        return []

    elens = edgelens_quad_batch(rp)
    angles = angles_quad_batch(rp)

    valid_quads = np.all(np.logical_and(np.logical_and(elens >= min_edge_len, elens <= max_edge_len),
                np.logical_and(angles >= min_angle, angles <= max_angle)), axis=1)

    vq_idx = np.flatnonzero(valid_quads)
    hull_list = hi[vq_idx, :]

    return hull_list

def hull_list_to_qp(hull_list):
    """form final quad proposals (qp) by taking all four possible starting vertices of each hull    
    """
    qp = set()
    for _, item in enumerate(hull_list):
        qp.add((item[0], item[1], item[2], item[3]))
        qp.add((item[1], item[2], item[3], item[0]))
        qp.add((item[2], item[3], item[0], item[1]))
        qp.add((item[3], item[0], item[1], item[2]))

    return qp

def warped_subimage(img, qv_img):
    """ qv_img ... 4x2 pixel coordinates of four quad vertices (clockwise)
        img ... image where the quad lives (greyscale but with RGB)
        returns: 104x104x1 subimage warped to canonical position
    """
    qv_canonical = np.array([[20, 20], [84, 20], [84, 84], [20, 84]], dtype=np.float64)
    M = cv2.getPerspectiveTransform(qv_img.astype(np.float32), qv_canonical.astype(np.float32))    
    wimg = cv2.warpPerspective(img, M, (104, 104), flags=cv2.INTER_NEAREST)         
    wimg = wimg[:,:,0]
    return wimg.reshape(wimg.shape + (1,))

def gen_crops(img, crop_size = 20, margin_size = 6, stride_size = 8):
    """ img ... input image (greyscale)
    task: chops the input image into a bunch of crops
    output: crops ... num_crops x crop_size x crop_size x 1 (greyscale)
    i_list, j_list ... pixel coordinates of where each crop starts
    """
    subimg_list, i_list, j_list = [], [], []
    for i in range(margin_size, img.shape[0] - crop_size, stride_size):
        for j in range(margin_size, img.shape[1] - crop_size, stride_size):
            im = i - margin_size
            jm = j - margin_size
            subimg = img[im : im + crop_size, jm : jm + crop_size, 0]
            assert(subimg.shape[0] == crop_size and subimg.shape[1] == crop_size)
            subimg_list.append(subimg)
            i_list.append(i)
            j_list.append(j)

    crops = np.array(subimg_list)
    crops = crops.reshape(crops.shape + (1,))
    return (crops, i_list, j_list)