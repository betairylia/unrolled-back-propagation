import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from matplotlib import animation

from matplotlib.widgets import Slider, Button, RadioButtons

def network(x, w1, b1, w2, b2):
    return np.maximum(0, w1 * x + b1) + np.maximum(0, w2 * x + b2) - 1

def partNetwork(x, w, b):
    return np.maximum(0, w * x + b) - 1

def target(mean, x):
    o = np.maximum(0, -x + (mean - 4)) + np.maximum(0, x - (mean + 4)) - 1
    return o

# TODO: hinge loss & flat target classification function
def loss(mean, x, y):
    o = target(mean, x)
    return np.sum(0.5 * (y - o) ** 2, axis = 0)
    # return np.log(np.sum(0.5 * (y - o) ** 2, axis = 0))

xmin = -1
xmax =  1
ymin = -20
ymax =  20

xstep = 0.1
ystep = 1.0

inputmin = -20
inputmax = 20
inputstep = 1

w1, b1 = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
# print(w1)
x = np.arange(inputmin, inputmax, inputstep)
x_a2 = np.reshape(x, [x.shape[0], 1, 1])

net_output = network(x_a2, w1, b1, 1, -19)
z = loss(10, x_a2, net_output)

fig = plt.figure(figsize=(12, 15))
ax = plt.axes([0.15, 0.15, 0.7, 0.583], projection='3d', elev=50, azim=-50)
ax_2d = plt.axes([0.1, 0.77, 0.8, 0.2])

p = ax.plot_surface(w1, b1, z, norm=LogNorm(), rstride=1, cstride=1, 
                edgecolor='none', alpha=.8, cmap=plt.cm.jet)
# ax.plot(*minima_, f(*minima_), 'r*', markersize=10)

ax.set_xlabel('$x (w1)$')
ax.set_ylabel('$y (b1)$')
ax.set_zlabel('$z$')

ax.set_xlim((xmin, xmax))
ax.set_ylim((ymin, ymax))

axMean = plt.axes([0.2, 0.1, 0.6, 0.02], facecolor = 'red')
axw2 = plt.axes([0.2, 0.06, 0.6, 0.02], facecolor = 'red')
axb2 = plt.axes([0.2, 0.02, 0.6, 0.02], facecolor = 'red')

sMean = Slider(axMean, 'data mean', -10, 10, valinit = 10)
sw2 = Slider(axw2, 'w2', -1, 1, valinit = 1)
sb2 = Slider(axb2, 'b2', -20, 20, valinit = -14)

def update(val):

    w2 = sw2.val
    b2 = sb2.val
    mean = sMean.val

    z = loss(mean, x_a2, network(x_a2, w1, b1, w2, b2))

    # p.set_3d_properties(loss(mean, x_a2, network(x_a2, w1, b1, w2, b2)))
    ax.clear()
    p = ax.plot_surface(w1, b1, z, norm=LogNorm(), rstride=1, cstride=1, 
                edgecolor='none', alpha=.8, cmap=plt.cm.jet)
        
    ax_2d.clear()
    ax_2d.plot(x, target(mean, x))
    ax_2d.plot(x, partNetwork(x, w2, b2), 'g')
    ax_2d.set_ylim((-5, 30))

    # minimum
    minValue = np.min(z)
    minPos = np.argmin(z)
    minPosX = (minPos % w1.shape[1]) * xstep + xmin
    minPosY = (minPos // w1.shape[1]) * ystep + ymin
    minima = np.array([minPosX, minPosY]).reshape(1, -1)

    ax.plot([minPosX], [minPosY], [minValue], 'r*', markersize=10, zorder = 10)

    loss_overlap = loss(mean, x, network(x, w2, b2, w2, b2))
    ax.plot([w2], [b2], [loss_overlap], 'go', markersize=10, zorder = 10)

    ax.set_xlabel('$x (w1)$')
    ax.set_ylabel('$y (b1)$')
    ax.set_zlabel('$z$')

    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))
    # ax.set_zlim((0, 50000))

    fig.canvas.draw_idle()

sMean.on_changed(update)
sw2.on_changed(update)
sb2.on_changed(update)
update(0)

plt.show()
