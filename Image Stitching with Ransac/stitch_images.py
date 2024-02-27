#!/usr/bin/env python
# coding: utf-8

import math
import cv2
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.ndimage as ndi
from matplotlib.patches import ConnectionPatch
from skimage.feature import SIFT, match_descriptors
from skimage.color import rgb2gray, rgba2rgb
from skimage.transform import resize, ProjectiveTransform, SimilarityTransform, warp
from skimage import measure, color
from skimage.measure import LineModelND
from scipy.spatial import distance
import random
import skimage.io as io



file1_path = 'campus_000.jpg' #'Rainier2.png'
file2_path = 'campus_001.jpg' #'Rainier1.png'




dst_img_rgb = np.asarray(Image.open(file1_path))
src_img_rgb = np.asarray(Image.open(file2_path))

if dst_img_rgb.shape[2] == 4:
    dst_img_rgb = rgba2rgb(dst_img_rgb)
if src_img_rgb.shape[2] == 4:
    src_img_rgb = rgba2rgb(src_img_rgb)

dst_img = rgb2gray(dst_img_rgb)
src_img = rgb2gray(src_img_rgb)

if dst_img_rgb.max()<=1:
    dst_img_rgb = (dst_img_rgb*255).astype('uint8')
if src_img_rgb.max()<=1:
    src_img_rgb = (src_img_rgb*255).astype('uint8')

W, H = src_img.shape

# detect keypoints and descriptors of source and dest images
detector1 = SIFT()
detector2 = SIFT()
detector1.detect_and_extract(dst_img)
detector2.detect_and_extract(src_img)
keypoints1 = detector1.keypoints
descriptors1 = detector1.descriptors
keypoints2 = detector2.keypoints
descriptors2 = detector2.descriptors


def match(descriptors1, descriptors2):
    
    src_index = []
    dist_index = []
    
    counter = 0
    for i in descriptors1:
        dist_vector = []
        for j in descriptors2:
            i = i.astype(np.float)
            j = j.astype(np.float)
            dist = distance.euclidean(i, j)
            dist_vector.append(dist)
     
        src_index.append(counter)
        dist_index.append(np.argmin(dist_vector))
        counter+=1
        
    matches = np.column_stack((src_index,dist_index))
    
    return matches


def plot_matches(keypoints1,keypoints2,matches,dst_img,src_img):
    dst = keypoints1[matches[:,0]]
    src = keypoints2[matches[:,1]]

    fig = plt.figure(figsize =(8,4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(dst_img)
    ax2.imshow(src_img)

    for i in range(src.shape[0]):
        coordB = [dst[i,1], dst[i,0]]
        coordA = [src[i,1], src[i,0]]
        con = ConnectionPatch(xyA=coordA, xyB=coordB, coordsA="data", coordsB="data", axesA=ax2, axesB=ax1, color="red")
        ax2.add_artist(con)
        ax1.plot(dst[i,1], dst[i,0], 'ro')
        ax2.plot(src[i,1], src[i,0], 'ro')


fig = plt.figure(figsize =(8,4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.imshow(dst_img_rgb)
ax2.imshow(src_img_rgb)

matches = match(descriptors1, descriptors2)

plot_matches(keypoints1,keypoints2,matches,dst_img_rgb,src_img_rgb)


def e2h(x):
    if len(x.shape) == 1:
        return np.hstack((x, [1]))
    return np.vstack((x, np.ones(x.shape[1])))
    
def h2e(tx):
    return tx[:-1]/tx[-1]


def compute_affine_transform(dst_p1, dst_p2, dst_p3, src_p1, src_p2, src_p3):
    
    M1 = np.vstack((np.hstack((e2h(dst_p1), np.zeros(3))), np.hstack((np.zeros(3), e2h(dst_p1)))))
    M2 = np.vstack((np.hstack((e2h(dst_p2), np.zeros(3))), np.hstack((np.zeros(3), e2h(dst_p2)))))
    M3 = np.vstack((np.hstack((e2h(dst_p3), np.zeros(3))), np.hstack((np.zeros(3), e2h(dst_p3)))))
    M  = np.vstack((M1, M2, M3))
    q = np.hstack((src_p1, src_p2, src_p3))
    
    return np.hstack(((np.linalg.inv(M) @ q), np.array([0, 0, 1]))).reshape(3, 3)

def compute_projective_transform(dst_p1, dst_p2, dst_p3, dst_p4,  src_p1, src_p2, src_p3, src_p4):
   
    M1 = np.vstack((np.hstack((e2h(dst_p1), np.zeros(3), np.array([-src_p1[0]*dst_p1[0], -src_p1[0]*dst_p1[1], -src_p1[0]]))), 
                    np.hstack((np.zeros(3), e2h(dst_p1), np.array([-src_p1[1]*dst_p1[0], -src_p1[1]*dst_p1[1], -src_p1[1]])))))
    
    M2 = np.vstack((np.hstack((e2h(dst_p2), np.zeros(3), np.array([-src_p2[0]*dst_p2[0], -src_p2[0]*dst_p2[1], -src_p2[0]]))), 
                    np.hstack((np.zeros(3), e2h(dst_p2), np.array([-src_p2[1]*dst_p2[0], -src_p2[1]*dst_p2[1], -src_p2[1]])))))
    
    M3 = np.vstack((np.hstack((e2h(dst_p3), np.zeros(3), np.array([-src_p3[0]*dst_p3[0], -src_p3[0]*dst_p3[1], -src_p3[0]]))), 
                    np.hstack((np.zeros(3), e2h(dst_p3), np.array([-src_p3[1]*dst_p3[0], -src_p3[1]*dst_p3[1], -src_p3[1]])))))
    
    M4 = np.vstack((np.hstack((e2h(dst_p4), np.zeros(3), np.array([-src_p4[0]*dst_p4[0], -src_p4[0]*dst_p4[1], -src_p4[0]]))), 
                    np.hstack((np.zeros(3), e2h(dst_p4), np.array([-src_p4[1]*dst_p4[0], -src_p4[1]*dst_p4[1], -src_p4[1]])))))

    M  = np.vstack((M1, M2, M3, M4))
    U, s, VT = np.linalg.svd(M, full_matrices=True)
    res = VT.T[:, -1]
    return res.reshape(3, 3)



def fit_model(data):
    M = np.empty((0,9))
    for (x, y), (xa, ya) in data:
        M = np.append(M, 
                      np.array([[x, y, 1, 0, 0, 0, -xa*x, -xa*y, -xa],
                                [0, 0, 0, x, y, 1, -ya*x, -ya*y, -ya]]),
                      axis=0)
    _, _, VT = np.linalg.svd(M)
    p = VT[-1]
    P = p.reshape((3,3))
    return P


def test_model(P, data, max_error=2):
    inliers = []
    for idx, dt in enumerate(data):
        yhat = h2e(P @ e2h(np.asarray(dt[0])))
        y    = dt[1]
        if max_error**2 >= (yhat[0]-y[0])**2 + (yhat[1]-y[1])**2:
            inliers.append(idx)
    return np.array(inliers)



def ransac(data, fit_model, test_model, test_model_pars, 
           n_model_data, n_iter):
    inliers = []
    for iter in range(n_iter):
        fit_data = random.sample(data, n_model_data)
        yfit_lsq = fit_model(fit_data)
        idx_inliers = test_model(yfit_lsq, data, max_error = test_model_pars['max_error'])
        if len(idx_inliers)>len(inliers):
            inliers = idx_inliers
    
    fit_inliers = [data[i] for i in inliers]
    model = fit_model(fit_inliers)
    
    return (model, inliers)


#ransac
data = [(tuple(keypoints1[m[0]][::-1]), tuple(keypoints2[m[1]][::-1])) for m in matches]
P, inliers = ransac(data, fit_model, test_model, {'max_error':2}, 4, 1000) 
matches = matches[inliers]
plot_matches(keypoints1,keypoints2,matches,dst_img_rgb,src_img_rgb)


#code for affine transform
idx = np.random.randint(low=0, high=len(matches), size=3)

dst_p1 = keypoints1[matches[idx[0], 0]][::-1]
src_p1 = keypoints2[matches[idx[0], 1]][::-1]

dst_p2 = keypoints1[matches[idx[1], 0]][::-1]
src_p2 = keypoints2[matches[idx[1], 1]][::-1]

dst_p3 = keypoints1[matches[idx[2], 0]][::-1]
src_p3 = keypoints2[matches[idx[2], 1]][::-1]

M, N = src_img_rgb.shape[:2]
A = compute_affine_transform(dst_p1, dst_p2, dst_p3, src_p1, src_p2, src_p3) 
res_r = (warp(dst_img_rgb[:, :, 0], np.linalg.inv(A), output_shape=(W, 2*H))*255).astype('uint8')
res_r[:M, :N] = src_img_rgb[:, :, 0]

res_g = (warp(dst_img_rgb[:, :, 1], np.linalg.inv(A), output_shape=(W, 2*H))*255).astype('uint8')
res_g[:M, :N] = src_img_rgb[:, :, 1]

res_b = (warp(dst_img_rgb[:, :, 2], np.linalg.inv(A), output_shape=(W, 2*H))*255).astype('uint8')
res_b[:M, :N] = src_img_rgb[:, :, 2]

res = cv2.merge([res_r,res_g,res_b])
idx = np.argwhere(np.all(res[..., :] == 0, axis=0))
res = np.delete(res, idx, axis=1)


#code for projective transform
idx = np.random.randint(low=0, high=len(matches), size=4)

dst_p1 = keypoints1[matches[idx[0], 0]][::-1]
src_p1 = keypoints2[matches[idx[0], 1]][::-1]

dst_p2 = keypoints1[matches[idx[1], 0]][::-1]
src_p2 = keypoints2[matches[idx[1], 1]][::-1]

dst_p3 = keypoints1[matches[idx[2], 0]][::-1]
src_p3 = keypoints2[matches[idx[2], 1]][::-1]

dst_p4 = keypoints1[matches[idx[3], 0]][::-1]
src_p4 = keypoints2[matches[idx[3], 1]][::-1]

M, N = src_img_rgb.shape[:2]
A = compute_projective_transform(dst_p1, dst_p2, dst_p3, dst_p4, src_p1, src_p2, src_p3, src_p4) 
res_r = (warp(dst_img_rgb[:, :, 0], np.linalg.inv(A), output_shape=(W, 2*H))*255).astype('uint8')
res_r[:M, :N] = src_img_rgb[:, :, 0]

res_g = (warp(dst_img_rgb[:, :, 1], np.linalg.inv(A), output_shape=(W, 2*H))*255).astype('uint8')
res_g[:M, :N] = src_img_rgb[:, :, 1]

res_b = (warp(dst_img_rgb[:, :, 2], np.linalg.inv(A), output_shape=(W, 2*H))*255).astype('uint8')
res_b[:M, :N] = src_img_rgb[:, :, 2]

res = cv2.merge([res_r,res_g,res_b])
idx = np.argwhere(np.all(res[..., :] == 0, axis=0))
res = np.delete(res, idx, axis=1)



res_r = (warp(dst_img_rgb[:, :, 0], np.linalg.inv(P), output_shape=(W, 2*H))*255).astype('uint8')
res_r[:M, :N] = src_img_rgb[:, :, 0].copy()

res_g = (warp(dst_img_rgb[:, :, 1], np.linalg.inv(P), output_shape=(W, 2*H))*255).astype('uint8')
res_g[:M, :N] = src_img_rgb[:, :, 1]

res_b = (warp(dst_img_rgb[:, :, 2], np.linalg.inv(P), output_shape=(W, 2*H))*255).astype('uint8')
res_b[:M, :N] = src_img_rgb[:, :, 2]

res = cv2.merge([res_r,res_g,res_b])
idx = np.argwhere(np.all(res[..., :] == 0, axis=0))
res = np.delete(res, idx, axis=1)


fig = plt.figure(figsize = (15, 8))
gs = GridSpec(2, 2, figure=fig)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])

ax1.imshow(dst_img_rgb)
ax1.set_title(file1_path.split('/')[-1])

ax2.imshow(src_img_rgb)
ax2.set_title(file2_path.split('/')[-1])


ax3.imshow(res)
ax3.set_title(' Transform Ransac')
io.imsave("stitched_img.png",res)
plt.show()




