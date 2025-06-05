##
white noise creator 
``python 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image

def denoise(filename, t_step, tmx):
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
    t_v = np.arange(0, tmx, t_step)
    final = np.zeros((len(t_v), num, num))
    final[0, :, :] = image[:num, :num]
    for t in range(len(t_v) - 1):
        for j in range(num):
            for i in range(num):
                lam =1
                if i == 0 or j == 0 or i == num - 1 or j == num - 1:
                    final[t + 1, i, j] = 0
                else:
                    xfor = final[t, i, j + 1] - final[t, i, j]
                    xback = final[t, i, j] - final[t, i, j - 1]
                    yfor = final[t, i, i + 1] - final[t, i, j]
                    yback = final[t, i, j] - final[t, i - 1, j]
                    denom=np.sqrt(xfor**2+yfor**2+0.001)
                    dt=xback*(xfor/denom)+yback*(yfor/denom)-lam*(final[t,i,j]-final[0,i,j])
                    final[t+1,i,j]=final[t,i,j]+dt
    fig, ax = plt.subplots()
    im = ax.imshow(final[0], aspect='auto', cmap='gray',
                   extent=[0, crop_size, 0, crop_size], vmin=0, vmax=1)

    def update(frame):
        im.set_array(final[frame])
        ax.set_title(f"Time: {frame * t_step:.2f} s")
        return [im]
    ani = animation.FuncAnimation(fig, update, frames=len(t_v), interval=1000, blit=True)
    ani.save("denoise.gif", writer="pillow", fps=1)
    plt.show()
denoise('Lena.png',0.5,7)
```
