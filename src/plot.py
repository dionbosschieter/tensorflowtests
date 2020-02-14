import matplotlib
import matplotlib.pyplot as plt
import numpy as np

y = np.arange(start=0.0, stop=2.0, step=0.01)
s = 1 + np.sin(2 * np.pi * y)

figure, ax = plt.subplots()
ax.plot(y, s)
ax.set(xlabel='time (s)', ylabel='voltage (mV)', title='Simple plot')
ax.grid()

# figure.savefig("test_plot.png")
plt.show()

if __name__ == '__main__':
    print(123)