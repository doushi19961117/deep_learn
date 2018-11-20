# -*- coding:utf-8 -*

'''
   学习目标
   1.熟悉plot的用法，会用其来绘制函数
   2.绘制子图
reference
  https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot

matplotlib API
  https://matplotlib.org/api/_as_gen/matplotlib.pyplot.html
'''

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter  # useful for `logit` scale

import numpy as np

# -----------------
# 绘制多条曲线(折线图 plot())
# -----------------

x1 = np.arange(0, 1.0, 0.001)  # 产生等差数列
x2 = np.arange(0, 1.0, 0.1)
x3 = np.arange(0, 1.0, 0.2)

y1 = x1
y2 = x2 ** 2
y3 = x3 ** 3

# 样式中b--,b表示颜色,--代表样式，plot连接的方式是直线
plt.plot(x1, y1, 'b--', label='y=x')
plt.plot(x2, y2, 'ro', label='y=x^2')
plt.plot(x3, y3, 'g:', label='y=x^3')

plt.ylabel('y')  # 在y轴上显示的文字
plt.xlabel('x')

plt.title('y=f(x)')

# loc的取值 'best'  'upper right' 'lower left'  'center left'
leg = plt.legend(loc='lower right', ncol=1, shadow=False, fancybox=True)  # 增加对曲线的描述，显示label内容

plt.show()

# ---------------------------
#  plot绘制多个子图
# ---------------------------

x1 = np.arange(-10, 10, 0.01)
x2 = np.arange(-10, 10, 0.01)

y1 = np.sin(x1)
y2 = np.cos(x2)

plt.figure(1)  # ?

# y=sin(x)  211表示2行1列中的第1个
plt.subplot(211)
plt.xlabel('x')
plt.ylabel(('y'))
plt.title('y=sin(x)')
plt.plot(x1, y1)
plt.grid(True)  # 可以使背景加上格子

# y=cos(x)
plt.subplot(212)
plt.xlabel('x')
plt.ylabel(('y'))
plt.title('y=cos(x)')
plt.plot(x2, y2)
plt.grid(True)

# Format the minor tick labels of the y-axis into empty strings with
# `NullFormatter`, to avoid cumbering the axis with too many labels.
# plt.gca().yaxis.set_minor_formatter(NullFormatter()) ？

# Adjust the subplot layout, because the logit one may take more space
# than usual, due to y-tick labels like "1 - 10^{-3}"
# 用来调整布局，避免重叠（需要深入了解）
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)

plt.show()
