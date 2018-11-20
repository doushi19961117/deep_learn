import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import product
from PIL import Image


# Fixing random state for reproducibility
np.random.seed(19680801)


N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = (30 * np.random.rand(N))**2  # 0 to 15 point radii

plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.show()

#
# img = Image.open('C:\\Users\\Administrator\\Desktop\\guangdong_round1_train1_20180903\\擦花20180901104310对照样本.jpg')
#
# fig2 = plt.figure()
# spec2 = gridspec.GridSpec(ncols=5, nrows=5,hspace=0.,wspace=0.)
#
# for i in range(5):
#     for j in range(5):
#         f2_ax1 =fig2.add_subplot(spec2[i, j])
#         f2_ax1.imshow(img)
#         f2_ax1.axis('off')
#
# plt.show()
