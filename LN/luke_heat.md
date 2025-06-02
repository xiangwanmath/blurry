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
## 
Nuemann BC heat equation 
``` python
import numpy as np
import matplotlib.pyplot as plt
def heat(x_step,t_step,tmx,a,b,lic,ric,bic):
    #set up
    l=(0.5**2)*(t_step/x_step**2)
    x_v=np.arange(a, b + x_step, x_step)
    t_v=np.arange(0, tmx, t_step)
    grid = np.empty((len(t_v), len(x_v)), dtype=object)
    for i, t in enumerate(t_v):
        flipped_i = len(t_v) - 1 - i
        for j, x in enumerate(x_v):
            grid[flipped_i, j] = (float(x), float(t))
    final=np.zeros((grid.shape))
    for i in range(len(x_v)):
        x,t=grid[0,i]
        final[-1,i]=bic(x)
    print(final)
    x_val = np.arange(a, b + x_step, x_step)
    mrx_one = np.zeros(((len(x_val)), (len(x_val))))
    mrx_two = np.zeros(((len(x_val)), 1))
    # get both matrix
    for i, t in enumerate(t_v[1:]):
        for n, x in enumerate(x_val):
            if n == 0:
                mrx_one[n, n] = -1
                mrx_one[n, n + 1] = 1
                mrx_two[n, 0] = lic(t) * x_step
            elif n == len(x_val)-1:
                mrx_one[n, n-1] = -1
                mrx_one[n, n] = 1
                mrx_two[n, 0] = ric(t) * x_step
            else:
                mrx_one[n,n-1]=-l
                mrx_one[n,n]=(1+2*l)
                mrx_one[n,n+1]=-l
                mrx_two[n,0]=final[len(t_v) - 1 - i, n]
        print(mrx_one)
        print(mrx_two)
        anw=np.linalg.solve(mrx_one,mrx_two)
        for o, a in enumerate(anw):
             final[- 2 - i, o] = a.item()
    print(final)
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
    return x**2
heat(0.01,0.01,1,-1,1,lic,ric,bic)
```
## 2d heat equation neumann
``` python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
def heat_neu_three(a,b,n,t_step,tmx,bic):
    xy_step=(b-a)/n
    c = (1.5 ** 2) * (t_step / xy_step ** 2)
    xy_v = np.arange(a, b+xy_step, xy_step)
    t_v = np.arange(0, tmx, t_step)
    grid = np.empty((len(xy_v), len(xy_v)), dtype=object)
    for i, y in enumerate(xy_v):
        for j, x in enumerate(xy_v):
            grid[i, j] = (x, y)
    final=np.zeros((len(t_v),len(xy_v),len(xy_v)))
    for i in range(len(xy_v)):
        for j in range(len(xy_v)):
            x,y=grid[i,j]
            final[0,i,j]=bic(x,y)
    num=len(xy_v)
    sc=0
    #solve
    for t in range(1, len(t_v)):
        mrx_one = np.zeros(((num ** 2), (num ** 2)))
        mrx_two = np.zeros(((num ** 2), 1))
        for m in range(num ** 2):
            i = m // num
            j = m % num
            index = i * num + j
            if 0 < i < num - 1 and 0 < j < num - 1:
                mrx_one[m, index] = 1 + 4 * c #self
                mrx_one[m, index - num] = -c  # top neighbor
                mrx_one[m, index + num] = -c  # bottom neighbor
                mrx_one[m, index - 1] = -c  # left neighbor
                mrx_one[m, index + 1] = -c #right neighbor
                mrx_two[m, 0] = final[t - 1, i, j]
            else:
                if i == 0:
                    mrx_one[m,index+num] = 1
                    mrx_one[m,index]=-1
                    mrx_two[m,0]=sc*xy_step
                elif i == num-1:
                    mrx_one[m,index-num] = -1
                    mrx_one[m,index]=1
                    mrx_two[m,0]=sc*xy_step
                elif j == 0:
                    mrx_one[m,index+1] = 1
                    mrx_one[m,index]=-1
                    mrx_two[m,0]=sc*xy_step
                elif j == num-1:
                    mrx_one[m,index-1] = -1
                    mrx_one[m,index]=1
                    mrx_two[m,0]=sc*xy_step
        anw = np.linalg.solve(mrx_one, mrx_two)
        for i in range(num):
            for j in range(num):
                ind=i * num + j
                final[t, i, j] = anw[ind].item()

    #  Animation
    fig, ax = plt.subplots()
    im = ax.imshow(final[0], aspect='auto', cmap='plasma', origin='lower',
                   extent=[xy_v[0], xy_v[-1], xy_v[0], xy_v[-1]], vmin=0, vmax=10)
    plt.colorbar(im, ax=ax, label='Temperature')
    ax.set_xlabel('Position (x)')
    ax.set_ylabel('Position (y)')
    ax.set_title('Heat Map of Temperature (Time Evolving)')
    def update(frame):
        im.set_array(final[frame])
        ax.set_title(f"Time: {frame * t_step:.2f} s")
        return [im]
    ani = animation.FuncAnimation(fig, update, frames=len(t_v), interval=1000, blit=True)
    ani.save("heat_animation.gif", writer="pillow", fps=1)
    plt.show()
def bic(x,y):
    if x>-0.75 and x<-0.25 and y<0.5 and y>0:
        return 10
    elif x<0.75 and x>0.25 and y<0.5 and y>0:
        return 10
    elif x>-0.75 and x<0.75 and y<-0.5 and y>-0.75:
        return 10
    else:
        return 0
heat_neu_three(-1,1,20,0.01,0.2,bic)
```
##
black and white explicit heat blur 
```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image

def new_kern(t_step, tmx, filename):
    img = Image.open(filename).convert('L')
    img_matrix = np.array(img) / 255.0
    h, w = img_matrix.shape
    crop_size = min(h, w)
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    image = img_matrix[start_h:start_h + crop_size, start_w:start_w + crop_size]
    xy_step = 1
    if t_step / xy_step > 0.5:
        raise ValueError("condition not met: xy_step / t_step must be ≤ 0.5")
    xy_v = image[0, :]
    num = len(xy_v)
    c = (1.5 ** 2) * (t_step / xy_step)
    t_v = np.arange(0, tmx, t_step)
    final = np.zeros((len(t_v), num, num))
    final[0, :, :] = image[:num, :num]

    def func(x):
        k = 1
        return 1 / (1 + (x / k) ** 2)

    # solve
    for t in range(len(t_v) - 1):
        for i in range(num):
            for j in range(num):
                if i == 0 or j == 0 or i == num - 1 or j == num - 1:
                    final[t + 1, i, j] = 0
                else:
                    n = final[t, i - 1, j] - final[t, i, j]
                    e = final[t, i, j + 1] - final[t, i, j]
                    s = final[t, i + 1, j] - final[t, i, j]
                    w = final[t, i, j - 1] - final[t, i, j]
                    final[t + 1, i, j] = c * (func(n) * n + func(e) * e + func(s) * s + func(w) * w) + final[t, i, j]

    # Animation
    fig, ax = plt.subplots()
    im = ax.imshow(final[0], aspect='auto', cmap='gray',
                   extent=[0, crop_size, 0, crop_size], vmin=0, vmax=1)

    def update(frame):
        im.set_array(final[frame])
        ax.set_title(f"Time: {frame * t_step:.2f} s")
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(t_v), interval=1000, blit=True)
    ani.save("new_kern.gif", writer="pillow", fps=1)
    plt.show()

new_kern(0.1, 1, 'abstrc.jpg')
```
