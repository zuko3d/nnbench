from __future__ import print_function
import numpy as np
import time

print("Generating dataset...")
X = np.random.uniform(size = (100000, 100)).astype(np.float32)
y = X

# =====================================================

print("Preparing...")
ts = time.clock()



print("Time elapsed: ", time.clock() - ts)

# ===================================================

print("Calculating...")
ts = time.clock()



print("Time elapsed: ", time.clock() - ts)