from matplotlib import pyplot as plt
import numpy as np
import matplotlib
from scipy.interpolate import pchip

x = [40, 60, 100, 180, 250]
print("x:", x)
random_res =[0.351, 0.47, 0.519, 0.55, 0.618]
sobel_res = [0.695, 0.702, 0.699, 0.704, 0.635]

plt.figure(figsize=(12.5, 10))
interpolation_sobel = pchip(x, sobel_res)
x_smooth = range(40,250) # smooth with step_size=1
sobel_res_interpo = interpolation_sobel(x_smooth)

interpolation_rand = pchip(x, random_res)
rand_res_interpo = interpolation_rand(x_smooth)

plt.plot(x_smooth, rand_res_interpo, label='Random Drop',  c='grey', linewidth=4)
plt.scatter(x, random_res, c="grey", s=199)
plt.plot(x_smooth, sobel_res_interpo, label='Saliency-score', c='red', linewidth=4)
plt.scatter(x, sobel_res, c="red", s=199)

plt.grid(True)
plt.xticks(fontsize=38)
plt.yticks(fontsize=38)
plt.legend(fontsize=38, loc="lower left")
plt.ylim([0.1, 0.726])
plt.xlabel(r"Patch Size", fontsize=38)
plt.ylabel(r"IDF1 ($\uparrow$)", fontsize=38)


plt.savefig("work_dirs/IDF1.png")

plt.close()


x = [40, 60, 100, 180, 250]
print("x:", x)
random_res = [0.079, 0.272, 0.363, 0.413, 0.503]
sobel_res = [0.613, 0.615, 0.622, 0.62, 0.517]

plt.figure(figsize=(12.5, 10))

interpolation_sobel = pchip(x, sobel_res)
x_smooth = range(40,250) # smooth with step_size=1
sobel_res_interpo = interpolation_sobel(x_smooth)

interpolation_rand = pchip(x, random_res)
rand_res_interpo = interpolation_rand(x_smooth)

plt.plot(x_smooth, rand_res_interpo, label='Random Drop',  c='grey', linewidth=4)
plt.scatter(x, random_res, c="grey", s=199)
plt.plot(x_smooth, sobel_res_interpo, label='Saliency-score', c='red', linewidth=4)
plt.scatter(x, sobel_res, c="red", s=199)

plt.grid(True)
plt.xticks(fontsize=38)
plt.yticks(fontsize=38)
plt.legend(fontsize=38, loc="lower right")
plt.ylim([0., 0.716])
plt.xlabel(r"Patch Size", fontsize=38)
plt.ylabel(r"MOTA ($\uparrow$)", fontsize=38)


plt.savefig("work_dirs/MOTA.png")

plt.close()
