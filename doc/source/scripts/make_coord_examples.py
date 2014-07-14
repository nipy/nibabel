#!/usr/bin/env python
""" Make graphics and example image for coordinate tutorial

Expects MNI nonlinear template t1 and t2 images in directory of script.

Requires nipy and matplotlib

Executing this script generates the following files in the current directory:

* localizer.png (pretend localizer sagittal image)
* someones_epi.nii.gz (pretend single EPI volume)
* someones_anatomy.nii.gz (pretend single subject structural)
"""
from __future__ import division, print_function

import math

import numpy as np
import numpy.linalg as npl

import nibabel.eulerangles as euler

import nipy
import nipy.core.api as nca
import nipy.algorithms.resample as rsm
import matplotlib.pyplot as plt

T1_IMG = 'mni_icbm152_t1_tal_nlin_asym_09a.nii'
T2_IMG = 'mni_icbm152_t2_tal_nlin_asym_09a.nii'

imgs = []
for img_fname in (T1_IMG, T2_IMG):
    img = nipy.load_image(img_fname)
    # Set affine as for FOV, not AC
    RZS = img.affine[:3, :3]
    vox_fov_center = -(np.array(img.shape) - 1) / 2.
    T = RZS.dot(vox_fov_center)
    img.affine[:3, 3] = T
    # Take stuff off the top of the full image, to emphasize FOV
    img_z_shave = 10
    # Take stuff off left and right to save disk space
    img_x_shave = 20
    img = img[img_x_shave:-img_x_shave, :, :-img_z_shave]
    imgs.append(img)

t1_img, t2_img = imgs

# Make fake localizer
data = t1_img.get_data()
n_x, n_y, n_z = img.shape
mid_x = round(n_x / 2)

sagittal = data[mid_x, :, :].T

# EPI bounding box
# 3 points on a not-completely-rectangular box. The box is to give a by-eye
# estimate, then we work out the box side lengths and make a rectangular box
# from those, using the origin point
bl = np.array((20, 15)) * 2
br = np.array((92, 70)) * 2
tl = np.array((7, 63)) * 2
# Find lengths of sides
epi_y_len = np.sqrt((np.subtract(bl, tl)**2).sum())
epi_x_len = np.sqrt((np.subtract(bl, br)**2).sum())
x, y = 0, 1
# Make a rectangular box with these sides
orth_epi_box = np.array((bl,
                         [bl[x] + epi_x_len, bl[y]],
                         [bl[x], bl[y] + epi_y_len],
                         [bl[x] + epi_x_len, bl[y] + epi_y_len]))

# Structural bounding box
bl = (25, 3)
anat_x_len = 185
anat_y_len = 155
anat_box = np.array((bl,
                         [bl[x] + anat_x_len, bl[y]],
                         [bl[x], bl[y] + anat_y_len],
                         [bl[x] + anat_x_len, bl[y] + anat_y_len]))


def plot_line(pt1, pt2, fmt='r-', label=None):
    plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], fmt,
             label=label)


def plot_box(box_def, fmt='r-', label=None):
    bl, br, tl, tr = box_def
    plot_line(bl, br, fmt, label=label)
    plot_line(bl, tl, fmt)
    plot_line(br, tr, fmt)
    plot_line(tl, tr, fmt)


def rotate_box(box_def, angle, origin):
    origin = np.atleast_2d(origin)
    box_def_zeroed = box_def - origin
    cost = math.cos(angle)
    sint = math.sin(angle)
    rot_array = np.array([[cost, -sint],
                          [sint, cost]])
    box_def_zeroed = np.dot(rot_array, box_def_zeroed.T).T
    return box_def_zeroed + origin


def labeled_point(pt, marker, text, markersize=10, color='k'):
    plt.plot(pt[0], pt[1], marker, markersize=markersize)
    plt.text(pt[0] + markersize / 2,
             pt[1] - markersize / 2,
             text,
             color=color)


def save_plot():
    # Plot using global variables
    plt.imshow(sagittal, cmap="gray", origin='lower')
    plot_box(rot_box, label='EPI bounding box')
    plot_box(anat_box, 'b-', label='Structural bounding box')
    labeled_point(epi_center, 'ro', 'EPI FOV center')
    labeled_point(anat_center, 'bo', 'Structural FOV center')
    labeled_point(iso_center, 'g^', 'Magnet isocenter')
    plt.axis('tight')
    plt.legend(loc='lower right')
    plt.savefig('localizer.png')


angle = 0.3
rot_box = rotate_box(orth_epi_box, angle, orth_epi_box[0])
epi_center = np.mean(rot_box, axis=0)
anat_center = np.mean(anat_box, axis=0)
iso_center = (np.array(sagittal.shape[::-1]) - 1) / 2

# Back to image coordinates
br_img = np.array([0, rot_box[0, 0], rot_box[0, 1]])
epi_trans = np.eye(4)
epi_trans[:3, 3] = -br_img
rot = np.eye(4)
rot[:3, :3] = euler.euler2mat(0, 0, -angle)
# downsample to make smaller output image
downsamp = 1/3
epi_scale = np.diag([downsamp, downsamp, downsamp, 1])
# template voxels to epi box image voxels
vox2epi_vox = epi_scale.dot(rot.dot(epi_trans))
# epi image voxels to mm
epi_vox2mm = t2_img.affine.dot(npl.inv(vox2epi_vox))
# downsampled image shape
epi_vox_shape = np.array([data.shape[0], epi_x_len, epi_y_len]) * downsamp
# Make sure dimensions are odd by rounding up or down
# This makes the voxel center an integer index, which is convenient
epi_vox_shape = [np.floor(d) if np.floor(d) % 2 else np.ceil(d)
                 for d in epi_vox_shape]
# resample, preserving affine
epi_cmap = nca.vox2mni(epi_vox2mm)
epi = rsm.resample(t2_img, epi_cmap, np.eye(4), epi_vox_shape)
epi_data = epi.get_data()
# Do the same kind of thing for the anatomical scan
anat_vox_sizes = [2.75, 2.75, 2.75]
anat_scale = npl.inv(np.diag(anat_vox_sizes + [1]))
anat_trans = np.eye(4)
anat_trans[:3, 3] = -np.array([0, anat_box[0, 0], anat_box[0, 1]])
vox2anat_vox = anat_scale.dot(anat_trans)
anat_vox2mm = t1_img.affine.dot(npl.inv(vox2anat_vox))
anat_vox_shape = np.round(np.divide(
        [data.shape[0], anat_x_len, anat_y_len], anat_vox_sizes))
anat_cmap = nca.vox2mni(anat_vox2mm)
anat = rsm.resample(t1_img, anat_cmap, np.eye(4), anat_vox_shape)
anat_data = anat.get_data()

save_plot()
nipy.save_image(epi, 'someones_epi.nii.gz', dtype_from='uint8')
nipy.save_image(anat, 'someones_anatomy.nii.gz', dtype_from='uint8')
