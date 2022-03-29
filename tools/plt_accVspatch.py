import numpy as np
from tqdm.notebook import tqdm
# %config InlineBackend.figure_format = 'svg'
import matplotlib
from matplotlib import pyplot as plt

matplotlib.rc('text', usetex = True)
matplotlib.rc('font', **{'family' : "sans-serif"})
params= {'text.latex.preamble' : [r'\usepackage{amsmath}', r'\usepackage{amsfonts}']}
plt.rcParams.update(params)
IDF1_random = [0.351, 0.47, 0.49, 0.519, 0.55, 0.618, 0.617]
IDF1_sobel = [0.695, 0.702, 0.699, 0.704, 0.685, 0.635, 0.656]

ratios = [40, 60, 80, 100, 180, 250, 300]
fig, ax = plt.subplots(figsize=(9, 8))

# plt.scatter(ratios, IDF1, label="Complexity", c='red')
# plt.scatter(ratios, IDF1_neighbor, label="Complexity + 3x3 Neighbor", c='blue')
# plt.scatter(ratios, IDF1_framet_1, label="by Frame t-1", c='cyan')

plt.plot(ratios, IDF1_random, label="Random", c='grey', marker='o')
plt.plot(ratios, IDF1_sobel, label=r"Saliency score", c='magenta', marker='^')


# plt.title(r"Mask 60$\times$ 60", fontsize=20)

plt.grid(True)
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
forward = lambda a: 1/(0.8 - a)
inverse = lambda b: 1/(0.8 - b)

# plt.xlim([-0.05, 0.7])
plt.ylim([0.1, 0.716])
# plt.yscale('function', functions=(forward, inverse))


plt.legend(fontsize=32, loc="lower left")

plt.xlabel(r"Patch Size", fontsize=32)
plt.ylabel(r"IDF1 ($\uparrow$)", fontsize=32)
plt.savefig("work_dirs/IDF1.png")

plt.close()


matplotlib.rc('text', usetex = True)
matplotlib.rc('font', **{'family' : "sans-serif"})
params= {'text.latex.preamble' : [r'\usepackage{amsmath}', r'\usepackage{amsfonts}']}
plt.rcParams.update(params)
IDF1_random = [0.079, 0.272, 0.3, 0.363, 0.413, 0.503, 0.493]
IDF1_sobel = [0.613, 0.615, 0.608, 0.622, 0.62, 0.517, 0.544]

ratios = [40, 60, 80, 100, 180, 250, 300]
fig, ax = plt.subplots(figsize=(10, 8))

# plt.scatter(ratios, IDF1, label="Complexity", c='red')
# plt.scatter(ratios, IDF1_neighbor, label="Complexity + 3x3 Neighbor", c='blue')
# plt.scatter(ratios, IDF1_framet_1, label="by Frame t-1", c='cyan')

plt.plot(ratios, IDF1_random, label="Random", c='grey', marker='o')
plt.plot(ratios, IDF1_sobel, label=r"Saliency score", c='magenta', marker='^')


# plt.title(r"Mask 60$\times$ 60", fontsize=20)

plt.grid(True)
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
forward = lambda a: 1/(0.8 - a)
inverse = lambda b: 1/(0.8 - b)

# plt.xlim([-0.05, 0.7])
plt.ylim([0.1, 0.716])
# plt.yscale('function', functions=(forward, inverse))


plt.legend(fontsize=32, loc="lower left")

plt.xlabel(r"Patch Size", fontsize=18)
plt.ylabel(r"MOTA ($\uparrow$)", fontsize=18)
plt.savefig("work_dirs/MOTA.png")

