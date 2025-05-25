##
1.11
##
midpoint aproximation 
<pre>def midpoint(f,a,b,n):
    h=(b-a)/n
    sum=0
    for i in range(n):
        x=a+(i+0.5)*h
        sum+=f(x)
    print('integral est is:',(sum*h))
def f(x):
    return x**2
midpoint(f,0,2,100)</pre>
  ##
  2.2
  ##
  convalution with one var
  <pre>import numpy as np
from matplotlib import pyplot as plt
def onevar():
    def g(t, x):
        e = 2.71828182846
        return (1 / np.sqrt(4 * np.pi * t)) * e ** ((-x ** 2) / (4 * t))

    def f(x):
        if x > 0:
            return 1
        else:
            return 0

    def midpoint(t, x, a, b, n):
        h = (b - a) / n
        sum = 0
        for i in range(n):
            tau = a + (i + 0.5) * h
            sum += g(t, x - tau) * f(tau)
        return (sum * h)

    t_value = 1
    data = np.zeros((2, 11), dtype='float64')
    for i in range(-5, 6):
        data[0, i + 5] = i
        value = midpoint(t_value, i, -100, 10000, 10000)
        data[1, i + 5] = value
    print(data[0])
    print(data[1])
    plt.plot(data[0], data[1])
    plt.title('convolution f*g with t= 1')
    plt.xlabel('s')
    plt.ylabel('convolution')
    plt.xlim(-7, 7)
    plt.ylim(-0.5, 1.5)
    plt.show()
onevar()</pre>
  ##
  2.3
  ##
  convalution with two var
  <pre>import numpy as np
def midpoint(t, x, y, n,f,g):
    xmn = -10
    xmx = 10
    ymn = -10
    ymx = 10
    xh = (xmx - xmn) / n
    yh = (ymx - ymn) / n
    sum = 0
    for i in range(n):
        u = xmn + (i + 0.5) * xh
        for j in range(n):
            v = ymn + (j + 0.5) * yh
            sum += g(t, x - u, y - v) * f(u, v)
    print(sum * (xh * yh))

def f(x, y):
    if -1 <= x <= 1 and -1 <= y <= 1:
        return 1
    else:
        return 0
def g(t, x,y):
    e = 2.71828182846
    return (1 /(4 * np.pi * t)) * e ** ((-(y**2+x ** 2)) / (4 * t))
midpoint(1,1,1,1000,f,g)import numpy as np
def midpoint(t, x, y, n,f,g):
    xmn = -10
    xmx = 10
    ymn = -10
    ymx = 10
    xh = (xmx - xmn) / n
    yh = (ymx - ymn) / n
    sum = 0
    for i in range(n):
        u = xmn + (i + 0.5) * xh
        for j in range(n):
            v = ymn + (j + 0.5) * yh
            sum += g(t, x - u, y - v) * f(u, v)
    print(sum * (xh * yh))

def f(x, y):
    if -1 <= x <= 1 and -1 <= y <= 1:
        return 1
    else:
        return 0
def g(t, x,y):
    e = 2.71828182846
    return (1 /(4 * np.pi * t)) * e ** ((-(y**2+x ** 2)) / (4 * t))
midpoint(1,1,1,1000,f,g)</pre>
