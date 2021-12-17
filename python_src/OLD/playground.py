import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fqs.fqs import quartic_roots
from sklearn.neighbors import BallTree

EARTH_RADIUS = 6371.009

# I fuck around in here
# This way we don't clog up our main files

rng = np.random.RandomState(0)

X = rng.random_sample((10,2))

print(X)
print(X[:1])

print(X.shape)
print(X[:1].shape)

tree = BallTree(X, leaf_size=2, metric='haversine')

# dist, ind = tree.query(X[:1], k=3)
# print(ind)
# print(dist)

print(tree.query_radius(X[:1], r=0.3, count_only=True))

ind = tree.query_radius(X[:1], r=0.3)  
print(ind)