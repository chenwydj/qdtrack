import os
from pickle import TRUE
import time
import torch
import numpy as np
import json
import pdb
import cv2
import copy
import random
import torch
from glob import escape, glob
from PIL import Image
import torchvision.transforms as T

from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

# image_idx = "4"
# coord_path = "visualize_features/visualization_data/top_bottom_left_right_{}.npy".format(image_idx)
# raw_mask_path = "visualize_features/visualization_data/image_{}.npy".format(image_idx)
# layer1_path = "visualize_features/visualization_data/image_{}_conv1.npy".format(image_idx)
# layer2_path = "visualize_features/visualization_data/image_{}_layer2.npy".format(image_idx)
# layer3_path = "visualize_features/visualization_data/image_{}_layer3.npy".format(image_idx)
# layer4_path = "visualize_features/visualization_data/image_{}_layer4.npy".format(image_idx)
layer4_path = "work_dirs/feature.npy"


# coord = np.load(coord_path, allow_pickle=True)
# raw_mask = np.load(raw_mask_path, allow_pickle=True)
# cv2.imwrite("visualize_features/{}_masked.png".format(image_idx), raw_mask)
# feature_layer1 = np.load(layer1_path, allow_pickle=True)
# feature_layer1_mean = np.mean(feature_layer1, 0)
# feature_layer1_mean = ((feature_layer1_mean / np.max(feature_layer1_mean))*255).astype(np.uint8)
# cv2.imwrite("visualize_features/{}_feat_lay1.png".format(image_idx), feature_layer1_mean)

# feature_layer2 = np.load(layer2_path, allow_pickle=True)
# feature_layer2_mean = np.mean(feature_layer2, 0)
# feature_layer2_mean = ((feature_layer2_mean / np.max(feature_layer2_mean))*255).astype(np.uint8)
# cv2.imwrite("visualize_features/{}_feat_lay2.png".format(image_idx), feature_layer2_mean)

# feature_layer3 = np.load(layer3_path, allow_pickle=True)
# feature_layer3_mean = np.mean(feature_layer3, 0)
# feature_layer3_mean = ((feature_layer3_mean / np.max(feature_layer3_mean))*255).astype(np.uint8)
# cv2.imwrite("visualize_features/{}_feat_lay3.png".format(image_idx), feature_layer3_mean)

feature_layer4 = np.load(layer4_path, allow_pickle=True)
feature_layer4_mean = np.mean(feature_layer4, 0)
feature_layer4_mean = ((feature_layer4_mean / np.max(feature_layer4_mean))*255).astype(np.uint8)
cv2.imwrite("work_dirs/feat_lay4.png", feature_layer4_mean)