import cv2, numpy as np, open3d as o3d, torch, ray, time, trimesh, pyflex, os, subprocess, imageio
from os import devnull
from copy import deepcopy
from sklearn.cluster import DBSCAN
from Imath import PixelType
from scipy.ndimage import distance_transform_edt
from clothmate.utils.utils import compute_intrinsics, translate2d, scale2d, rot2d, rigid_transform_3D, superimpose, transform
from clothmate.utils.utils import get_transform_matrix, pixel_to_3d
from torchvision import transforms
from pathlib import Path
from itertools import product
import matplotlib.pyplot as plt

def shift_tensor(tensor, offset):
    new_tensor = torch.zeros_like(tensor).bool()
    #shifted up
    if offset > 0:
        new_tensor[:, :-offset, :] = tensor[:, offset:, :]
    #shifted down
    elif offset < 0:
        offset *= -1
        new_tensor[:, offset:, :] = tensor[:, :-offset, :]
    return new_tensor

def prepare_image(img, transformations, dim: int,
                  parallelize=False, log=False, orientation_net=None, nocs_mode=None, constant_positional_enc = False, inter_dim=256):

    assert nocs_mode == "collapsed" or nocs_mode == "distribution"

    if orientation_net is not None:

        mask = torch.sum(img[:3,], axis=0) > 0
        mask = torch.unsqueeze(mask, 0)

        #resize to network input shape
        input_img = transforms.functional.resize(img, (128, 128))


        with torch.no_grad():
            prepped_img = torch.unsqueeze(input_img[:3, :, :], 0).cpu()
            #print the type of prepped img
            out = ray.get(orientation_net.forward.remote(prepped_img))[0]
            # out = orientation_net.forward(torch.unsqueeze(input_img[:3, :, :], 0))[0]

        nocs_x_bins = out[:, 0, :, :]
        nocs_y_bins = out[:, 1, :, :]
        n_bins = out.shape[0]

        #out shape: 32, 2, 128, 128
        if nocs_mode == "collapsed":
            # mask = torch.cat(2*[torch.unsqueeze(mask, 0)], dim=0)
            #32 bins
            nocs_x = torch.unsqueeze(torch.argmax(nocs_x_bins, dim=0).type(torch.float32)/(n_bins-1), 0)
            nocs_y = torch.unsqueeze(torch.argmax(nocs_y_bins, dim=0).type(torch.float32)/(n_bins-1), 0)
            #mask out bg
            nocs = torch.cat([nocs_x, nocs_y], dim=0)

        elif nocs_mode == "distribution":
            # mask = torch.cat((n_bins * 2)*[torch.unsqueeze(mask, 0)], dim=0)

            nocs_x = torch.nn.functional.softmax(nocs_x_bins, dim=0)
            nocs_y = torch.nn.functional.softmax(nocs_y_bins, dim=0)

            nocs = torch.cat([nocs_x, nocs_y], dim=0)

            nocs = nocs[::2] + nocs[1::2]
        else:
            raise NotImplementedError 

        #to make things more computationally tractable
        # print("NOCS shape", nocs.shape)
        nocs = transforms.functional.resize(nocs, (img.shape[-1], img.shape[-2])).to(img.device)
        nocs = nocs * mask.int() + (1 - mask.int()) * 0.0


        img = torch.cat([img, nocs], dim=0)

    log = False
    if log:
        start = time()

    img = img.cpu()
    img = transforms.functional.resize(img, (inter_dim, inter_dim))
    imgs = torch.stack([transform(img, *t, dim=dim, constant_positional_encoding=constant_positional_enc) for t in transformations])

    if log:
        print(f'\r prepare_image took {float(time()-start):.02f}s with parallelization {parallelize}')

    return imgs.float()

def generate_workspace_mask(left_mask, right_mask, action_primitives, pix_place_dist, pix_grasp_dist):
                                
    workspace_masks = {}
    for primitive in action_primitives:
        if primitive == 'place':

            lowered_left_primitive_mask = shift_tensor(left_mask, -pix_place_dist)
            lowered_right_primitive_mask = shift_tensor(right_mask, -pix_place_dist)
            #WORKSPACE CONSTRAINTS (ensures that both the pickpoint and the place points are located within the workspace)
            left_primitive_mask = torch.logical_and(left_mask, lowered_left_primitive_mask)
            right_primitive_mask = torch.logical_and(right_mask, lowered_right_primitive_mask)
            primitive_workspace_mask = torch.logical_or(left_primitive_mask, right_primitive_mask)

        elif primitive == 'fling' or primitive == 'drag' or primitive == 'stretchdrag':

            raised_left_primitive_mask = shift_tensor(left_mask, pix_grasp_dist)
            lowered_left_primitive_mask = shift_tensor(left_mask, -pix_grasp_dist)
            raised_right_primitive_mask = shift_tensor(right_mask, pix_grasp_dist)
            lowered_right_primitive_mask = shift_tensor(right_mask, -pix_grasp_dist)
            #WORKSPACE CONSTRAINTS
            aligned_workspace_mask = torch.logical_and(raised_left_primitive_mask, lowered_right_primitive_mask)
            opposite_workspace_mask = torch.logical_and(raised_right_primitive_mask, lowered_left_primitive_mask)
            primitive_workspace_mask = torch.logical_or(aligned_workspace_mask, opposite_workspace_mask)
        
        workspace_masks[primitive] = primitive_workspace_mask

    return workspace_masks 


def generate_primitive_cloth_mask(cloth_mask, action_primitives, pix_place_dist, pix_grasp_dist):
    cloth_masks = {}
    for primitive in action_primitives:
        if primitive == 'place':
            primitive_cloth_mask = cloth_mask
        elif primitive == 'fling' or primitive == 'drag' or primitive == 'stretchdrag':
            #CLOTH MASK (both pickers grasp the cloth)
            raised_primitive_cloth_mask = shift_tensor(cloth_mask, pix_grasp_dist)
            lowered_primitive_cloth_mask = shift_tensor(cloth_mask, -pix_grasp_dist)
            primitive_cloth_mask = torch.logical_and(raised_primitive_cloth_mask, lowered_primitive_cloth_mask)
        else:
            raise NotImplementedError
        cloth_masks[primitive] = primitive_cloth_mask
    return cloth_masks