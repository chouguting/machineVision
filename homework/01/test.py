import numpy as np
arr = np.ones((3, 3 ,3))
#print(arr)

arr[:, :, 1] = arr[:, :, 1] * 5
print(arr)