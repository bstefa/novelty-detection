#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 15:53:37 2021

Script for loading an image, segmenting it in to superpixels,
and forming an undirected graph of the result.

@author: brahste
"""
import yaml
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from utils import tools
from datasets import lunar_analogue
from skimage import segmentation, feature

# Read configuration file from command line args
config_file = tools.handle_command_line_arguments()

with open(config_file) as f:
    config = yaml.full_load(f)

# Import data and create generator object
data_obj = lunar_analogue.LunarAnalogueDataGenerator(config)
gen = data_obj.create_generator('train')

##! Temporary single image manipulation to get the jist of
##  creating a graph from superpixel segmentation
batch = next(gen)
im = batch[0]

#! Obtain superpixel segmentation; other algorithms will be investigated 
#  here later
seg_mask = segmentation.felzenszwalb(im, scale=1000, sigma=2)
seg_ints, seg_counts = np.unique(seg_mask, return_counts=True)

fig = plt.figure(figsize=(10,10))
plt.imshow(segmentation.mark_boundaries(im, seg_mask)); plt.show()

# Compute the Grey Level Coocurrance Matrix
glcm = feature.greycomatrix(
    seg_mask,
    distances=[1], 
    angles=[0, np.pi/2], 
    levels=len(seg_ints),
    symmetric=True,
    normed=True
)

# Here, sum the east-west and north-south cooccurances,
# divide by 4 because symmetric-ness uses transpose operation
# which double counts implicitly, and we're summing two symmetric
# matrices
connectivity_mat = (glcm[..., 0, 0] + glcm[..., 0, 1]) / 4  
print('Connectivity:\n----\n', connectivity_mat)

#%%
net = nx.Graph()
net.add_nodes_from(seg_ints)

for i, j in np.ndindex(connectivity_mat.shape):
    if connectivity_mat[i, j] > 0:
        net.add_edge(i, j, weight=connectivity_mat[i, j])

options = {
    "font_size": 6,
    "node_size": 30,
    "node_color": "white",
}

nx.draw_networkx(net, pos=nx.layout.spring_layout(net), **options)
# print(net.number_of_nodes())
# print(net.nodes)
# print(net.number_of_edges())
print(net.edges(2, 'weight'))
#%%
# Do some thresholding as a vertex feature
seg_mask_71 = seg_mask == 68
im[seg_mask_71] = 1.0
plt.imshow(im)
plt.show()


#%%




#%%
NUM_SUPERPIXELS = 10

G = nx.Graph()
G.add_nodes_from(range(NUM_SUPERPIXELS))
G.add_edges_from([(1,2), (2,3), (6,8), (2,1)])

#%%
print(G.nodes())
print(G.edges())
nx.draw(G)



