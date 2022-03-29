from matplotlib import pyplot as plt
import numpy as np
import matplotlib
from scipy.interpolate import pchip

x = [3, 10, 15, 20]
print("x:", x)
random_res =[67.6, 68.1, 68.2, 67.9]
plt.figure(figsize=(9, 6))
x_smooth = range(3,21) # smooth with step_size=1

interpolation_rand = pchip(x, random_res)
rand_res_interpo = interpolation_rand(x_smooth)

plt.plot(x_smooth, rand_res_interpo, label='Random Drop',  c='blue', linewidth=2)
plt.scatter(x, random_res, c="navy", s=99)

plt.grid(True)
plt.xticks(fontsize=21)
plt.yticks(fontsize=21)
# plt.legend(fontsize=13, loc="lower left")
plt.ylim([66, 68.5])
plt.xlim([2.7, 21])
plt.xlabel(r"Neighbours", fontsize=21)
plt.ylabel(r"Acc % ($\uparrow$)", fontsize=21)


plt.savefig("work_dirs/IDF1.png")

plt.close()

