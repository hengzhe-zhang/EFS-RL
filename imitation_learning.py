import pickle
import random

import numpy as np
from sklearn.metrics import r2_score

from efs.evolutionary_feature_synthesis import EFSRegressor, REGRESSION

random.seed(0)
np.random.seed(0)

with open("data/cartpole.pickle", "rb") as file:
    X, y = pickle.load(file)
    X, y = np.array(X), np.array(y)

for id, xx, yy in zip(range(len(X)), X, y):
    efs = EFSRegressor(
        max_gens=50,
        method=REGRESSION,
        num_additions=10,
        feature_size_limit=8,
        verbose=False,
    )
    efs.fit(xx, yy)

    print("R2", r2_score(yy, efs.predict(xx)))
    print("Model", id + 1, str(efs))
