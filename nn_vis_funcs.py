from manim import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from skimage.measure import find_contours
from scipy.interpolate import RegularGridInterpolator

import torch
from torch import nn


# grid interpolation optimization
def build_prediction_grid(model, u_range, v_range, resolution, layer=None):
    u_vals = np.linspace(u_range[0], u_range[1], resolution[0])
    v_vals = np.linspace(v_range[0], v_range[1], resolution[1])
    layer_size = 1 if layer is None else model[layer].out_features

    # build every (u, v) combination as a batch
    uu, vv = np.meshgrid(u_vals, v_vals, indexing="ij")
    points = np.stack([uu.ravel(), vv.ravel()], axis=1)  # shape (N, 2)

    with torch.no_grad():
        if layer is None:
            preds = model(torch.from_numpy(points).float())
        else:
            preds = model[:layer+1](torch.from_numpy(points).float())

    z_grid = preds.numpy().reshape(resolution[0], resolution[1], layer_size)

    return u_vals, v_vals, z_grid



def curry_compute_location_fast(model, u_range, v_range, grid_resolution=(50, 50), normalize=True):
    u_vals, v_vals, z_grid = build_prediction_grid(model, u_range, v_range, grid_resolution)
    z_grid = z_grid.squeeze()
    if normalize:
        z_min = z_grid.min()
        print("z_min: ", z_grid.min())
        z_max = z_grid.max()
        print("z_max: ", z_grid.max())
        normalized_z = (z_grid - z_min) / (z_max - z_min)
        print(normalized_z.shape)
        z_grid = normalized_z
        print(normalized_z)
        z_grid = z_grid * 3
    interpolator = RegularGridInterpolator((u_vals, v_vals), z_grid)

    def f(u, v):
        z = interpolator([[u, v]])[0]
        return np.array([u, v, z])

    return f



def find_model_contours(model, u_range, v_range, grid_resolution=(50,50), layer=None):
    u_vals, v_vals, z_grid = build_prediction_grid(model, u_range, v_range, grid_resolution, layer=layer)
    out_features = 1 if layer is None else model[layer].out_features
    level = torch.tensor(z_grid).median() if layer is None else 0

    contours = []
    for node_num in range(out_features):
        contours += find_contours(z_grid[...,node_num], level=level)
    
    
    for i in range(len(contours)):
        if len(contours[i]) == 0:
            continue

        contours[i] = torch.tensor(contours[i]).squeeze()
        contours[i] /= torch.tensor([grid_resolution[0], grid_resolution[1]])
        contours[i] *= torch.tensor([u_range[1]-u_range[0], v_range[1]-v_range[0]])
        contours[i] += torch.tensor([u_range[0], v_range[0]])
        

    return contours


# returns a list of the layer sizes in a torch model
def get_layer_sizes(model):
    ls = []
    first_flag = True
    for layer in model:
        if (type(layer) == nn.Linear):
            if first_flag:
                ls.append(layer.in_features)
                first_flag = False
            ls.append(layer.out_features)
    return ls
