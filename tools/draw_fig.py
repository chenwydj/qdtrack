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
# IDF1 = [0.493, 0.519, 0.526, 0.534] #random zero
IDF1 = [0.703, 0.707, 0.708, 0.709, 0.709, 0.708, 0.707, 0.706, 0.705, 0.703] #complexity zero
patch = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
fig, ax = plt.subplots(figsize=(9, 7))
plt.scatter(patch, IDF1)
plt.title(r"Ratio 0.2", fontsize=20)

plt.grid(True)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.xlabel(r"Mask Patch Size", fontsize=18)
plt.ylabel(r"IDF1 ($\uparrow$)", fontsize=18)

plt.savefig("1.png")