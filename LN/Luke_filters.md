##
white noise creator 
```python 
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
##
de-noise
```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def denoise(filename, t_step,sigma,stop):
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
    orig_image = image.copy()
    ux0 = np.zeros_like(image)
    uy0 = np.zeros_like(image)
    ux0[:, 1:-1] = (orig_image[:, 2:] - orig_image[:, :-2]) / 2
    uy0[1:-1, :] = (orig_image[2:, :] - orig_image[:-2, :]) / 2
    energy_history = [0]
    n=0
    q=0
    while q==0 and n<5000:
        n+=1
        ux = np.zeros((num, num))
        uy = np.zeros((num, num))
        ux[:, 1:-1] = (image[:, 2:] - image[:, :-2]) / 2
        uy[1:-1, :] = (image[2:, :] - image[:-2, :]) / 2
        # boundary
        ux[:, 0] = ux[:, 1]
        ux[:, -1] = ux[:, -2]
        uy[0, :] = uy[1, :]
        uy[-1, :] = uy[-2, :]

        eps = 1e-8
        denom = np.sqrt(ux ** 2 + uy ** 2 + eps)
        px=ux/denom
        py=uy/denom
        div = np.zeros_like(image)
        div[1:-1,1:-1] = (px[1:-1,2:] - px[1:-1,:-2])/2 + (py[2:,1:-1] - py[:-2,1:-1])/2

        #lamda
        integral = np.abs(denom - ((ux0 * ux + uy0 * uy) / denom))
        lam = 10*np.mean(integral) / (2 * sigma ** 2 + 1e-6)
        image -= t_step * (-div + lam * (image - orig_image))
        #e
        e = np.sum(np.sqrt(ux ** 2 + uy ** 2)) + (lam / 2) * np.sum((image - orig_image) ** 2)
        energy_history.append(e)
        if abs(energy_history[n] - energy_history[n-1]) < stop:
            q+=1
        if n % 10 == 0:
            print(n)
            print('lam=',lam)
            print('e=',e)
            print()
    plt.figure()
    plt.imshow(orig_image, cmap='gray', vmin=0, vmax=1)
    plt.title("Original Image")
    plt.axis('off')
    plt.show()

    plt.figure()
    plt.imshow(image, cmap='gray', vmin=0, vmax=1)
    plt.title("Denoised Image")
    plt.axis('off')
    plt.show()

    plt.figure()
    plt.plot(energy_history[1:], label='Energy')
    plt.xlabel('Iteration')
    plt.ylabel('Energy')
    plt.title('Energy Decay Over Time')
    plt.grid(True)
    plt.legend()
    plt.show()
denoise('Lena.jpg',0.0001, 0.4,0.01)
```
