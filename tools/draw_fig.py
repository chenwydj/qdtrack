import numpy as np
from tqdm.notebook import tqdm
# config InlineBackend.figure_format = 'svg'
import matplotlib
from matplotlib import pyplot as plt

matplotlib.rc('text', usetex = True)
matplotlib.rc('font', **{'family' : "sans-serif"})
params= {'text.latex.preamble' : [r'\usepackage{amsmath}', r'\usepackage{amsfonts}']}
plt.rcParams.update(params)


#############
# IDF1 = [0.628, 0.522, 0.527, 0.536] #random zero
# IDF1 = [0.712, 0.708, 0.701, 0.687] #complexity zero
# ratios = [0.1, 0.2, 0.3, 0.4]
# fig, ax = plt.subplots(figsize=(8, 7))
# plt.scatter(ratios, IDF1)
# plt.title(r"Mask 60$\times$ 60", fontsize=20)

# plt.grid(True)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)

# plt.xlabel(r"Ratio: $\frac{\text{area of mask}}{\text{image size}}$", fontsize=18)
# plt.ylabel(r"IDF1 ($\uparrow$)", fontsize=18)

# plt.savefig("1.png")


#############
# IDF1_BS = [0.714, 0.712, 0.708, 0.701, 0.687] #complexity zero
# IDF1_DS2 = [0.58, 0.577, 0.57, 0.557, 0.535]
# IDF1_DS4 = [0.367, 0.35, 0.326, 0.294, 0.259]
# ratios = [0.0, 0.1, 0.2, 0.3, 0.4]
# fig, ax = plt.subplots(figsize=(9, 7))
# plt.scatter(ratios, IDF1_BS)
# plt.scatter(ratios, IDF1_DS2)
# plt.scatter(ratios, IDF1_DS4)

# plt.grid(True)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)

# plt.xlabel(r"Ratio", fontsize=18)
# plt.ylabel(r"IDF1 ($\uparrow$)", fontsize=18)
# plt.savefig("1.png")

#############
IDF1_BS = [0.714, 0.712, 0.708, 0.701, 0.687] #complexity zero
IDF1_DS2 = [0.58, 0.577, 0.57, 0.557, 0.535]
IDF1_DS4 = [0.367, 0.35, 0.326, 0.294, 0.259]
ratios = [0.0, 0.1, 0.2, 0.3, 0.4]
fig, ax = plt.subplots(figsize=(9, 7))
plt.scatter(ratios, IDF1_BS, label="baseline")
plt.scatter(ratios, IDF1_DS2, label="H/2, W/2")
plt.scatter(ratios, IDF1_DS4, label="H/4, W/4")

plt.legend(loc=3)
plt.grid(True)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.xlabel(r"Ratio", fontsize=18)
plt.ylabel(r"IDF1 ($\uparrow$)", fontsize=18)
plt.savefig("1.png")