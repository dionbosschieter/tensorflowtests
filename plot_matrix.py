import matplotlib.pyplot as plt
import numpy as np

np.random.seed(101)
matrix = np.floor(np.random.random((100, 100)) + .5)

plt.subplot(211)
plt.imshow(matrix)
plt.show()
