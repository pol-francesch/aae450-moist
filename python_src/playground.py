import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# I fuck around in here
# This way we don't clog up our main files

# This is getting the random data into the DF
xc1, xc2, yc1, yc2 = 113.49805889531724, 115.5030664238035, 37.39995194888143, 38.789235929357105       

N = 1000
GSIZE = 20
x, y = np.random.multivariate_normal([(xc1 + xc2)*0.5, (yc1 + yc2)*0.5], [[0.1, 0.02], [0.02, 0.1]], size=N).T
value = np.ones(N)

df_points = pd.DataFrame({"x":x, "y":y, "v":value})

# This is using weird grids on it
X, Y = np.mgrid[x.min():x.max():GSIZE*1j, y.min():y.max():GSIZE*1j]
print(X.shape)
grid = np.c_[X.ravel(), Y.ravel()]
print(grid.shape)
points = np.c_[df_points.x, df_points.y]
print(points.shape)
tree = KDTree(grid)
dist, indices = tree.query(points)
lat_est = grid[indices, 0]
print(indices.shape)
print(lat_est.shape)

# WHat is happening here
print(points[0,0])
print(lat_est[0])
print(grid)