##
heat equation graph
```python
import numpy as np
import matplotlib.pyplot as plt
def heat(x_step,t_step,tmx,a,b,lic,ric,bic):
    #set up
    l=(1.5**2)*(t_step/x_step**2)
    x_v=np.arange(a, b + x_step, x_step)
    t_v=np.arange(0, tmx, t_step)
    grid = np.empty((len(t_v), len(x_v)), dtype=object)
    for i, t in enumerate(t_v):
        flipped_i = len(t_v) - 1 - i
        for j, x in enumerate(x_v):
            grid[flipped_i, j] = (float(x), float(t))
    final=np.zeros((grid.shape))
    for i in range(len(t_v)):
        x,t=grid[i,0]
        final[i,0]=lic(t)
        final[i,-1]=ric(t)
    for i in range(len(x_v)):
        x,t=grid[0,i]
        final[-1,i]=bic(x)
    x_val = np.arange(a + x_step, b, x_step)
    mrx_one = np.zeros(((len(x_val)), (len(x_val))))
    mrx_two = np.zeros(((len(x_val)), 1))
    # get both matrix
    for i, t in enumerate(t_v[1:]):
        for n, x in enumerate(x_val):
            if n == 0:
                mrx_one[n, n] = (1 + 2 * l)
                mrx_one[n, n + 1] = -l
                mrx_two[n, 0] = final[len(t_v) - 1 - i, n + 1] + l * lic(t)
            elif n == len(x_val) - 1:
                mrx_one[n, n-1] = -l
                mrx_one[n, n] = (1 + 2 * l)
                mrx_two[n, 0] = final[len(t_v) - 1 - i, n + 1] + l * ric(t)
            else:
                mrx_one[n,n-1]=-l
                mrx_one[n,n]=(1+2*l)
                mrx_one[n,n+1]=-l
                mrx_two[n,0]=final[len(t_v) - 1 - i, n + 1]
        anw=np.linalg.solve(mrx_one,mrx_two)
        for o, a in enumerate(anw):
             final[- 2 - i, o + 1] = a.item()
    plt.imshow(final, aspect='auto', cmap='plasma',
               extent=[x_v[0], x_v[-1], t_v[0], t_v[-1]])
    plt.colorbar(label='Temperature')
    plt.xlabel('Position (x)')
    plt.ylabel('Time (t)')
    plt.title('Heat Map of Temperature Over Time and Space')
    plt.show()
def lic(t):
    return 0
def ric(t):
    return 0
def bic(x):
    return 2-abs(x)
heat(0.01,0.01,0.5,-1,1,lic,ric,bic)
```
