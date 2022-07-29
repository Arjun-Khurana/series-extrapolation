import numpy as np
from matplotlib import pyplot as plt


def p(x, y):
    # print(len(x))
    if len(x) == 0:
        return 0
    elif len(x) == 1:
        return y[0]
    if len(x) == 2:
        return (x[0] - x[1])/(y[0] - y[1])
    else:
        return (x[0] - x[-1])/(p(x[:-1], y[:-1]) - p(x[1:], y[1:])) + p(x[1:-1], y[1:-1])


def thiele(x, y, xp):
    print(len(x))
    if len(x) == 1:
        return 0
    else:
        return y[0] + ((xp - x[0]) / (p(x,y) + thiele(x[1:], y[1:], xp)))

def thieleInterpolator(x, y):
    p = [[yi]*(len(y)-i) for i, yi in enumerate(y)]
    for i in range(len(p)-1):
        p[i][1] = (x[i] - x[i+1]) / (p[i][0] - p[i+1][0])
    for i in range(2, len(p)):
        for j in range(len(p)-i):
            p[j][i] = (x[j]-x[j+i]) / (p[j][i-1]-p[j+1][i-1]) + p[j+1][i-2]
    p0 = p[0]
    def t(xin):
        a = 0
        for i in range(len(p0)-1, 1, -1):
            a = (xin - x[i-1]) / (p0[i] - p0[i-2] + a)
        return y[0] + (xin-x[0]) / (p0[1]+a)
    return t

def main():
    x = np.linspace(0, 2*np.pi, 100)
    y = np.sin(x)
    tsin = thieleInterpolator(x, y)
    xlong = np.linspace(0, 10*np.pi, 500)
    ylong = np.sin(xlong)
    plt.plot(xlong,ylong)
    plt.plot(xlong, tsin(xlong), marker='o')
    plt.ylim(-10, 10)
    # x = np.linspace(0,2*np.pi, 20)
    # y = np.sin(x)
    # xp = 10
    # x1 = x[:xp]
    # y1 = y[:xp]
    # yp = thiele(x1, y1, x[xp])
    # print(yp)
    # plt.plot(x1,y1)
    # plt.plot(x[xp], yp, marker='o')
    plt.show()


if __name__ == '__main__':
    main()