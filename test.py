import numpy as np
from matplotlib import pyplot as plt

x = np.linspace(-10, 10, 1000)
y = np.sin(2*np.pi*x)

fw = np.fft.fft(y)
w = np.fft.fftfreq(len(x), x[1]-x[0])

# y2 = np.fft.ifft(fw)
# plt.plot(x,y2)
# plt.show()
plt.plot(w,fw)
plt.show()

plt.plot(x,y)
plt.show()