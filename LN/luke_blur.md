##
2.4
##
black and white blur

<pre>def blur(bluramount,kernsize,picturesize):
    size = picturesize
    #makes the origional
    img = Image.open('ironman.jpg').convert('L')
    img_matrix = np.array(img) / 255.0
    h, w = img_matrix.shape
    crop_size = min(h, w)
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    square_matrix = img_matrix[start_h:start_h + crop_size, start_w:start_w + crop_size]
    #makes the kernel
    t_value = bluramount
    def g(t, x, y):
        return (1 / (4 * math.pi * t)) * math.exp(-(x**2 + y**2) / (4 * t))

    halfkern = kernsize // 2
    datagrid = np.zeros((kernsize,kernsize),dtype='object')
    dataval = np.zeros((kernsize,kernsize),dtype='float')
    for ix, i in enumerate(range(-halfkern+1, halfkern)):
        for iy, j in enumerate(reversed(range(-halfkern+1, halfkern))):
            datagrid[iy,ix] = i,j
            dataval[iy,ix]= round(g(t_value, i, j),20)
    dataval /= dataval.sum()

    new=np.zeros((size,size))
    padded = np.pad(square_matrix, kernsize//2, mode='edge')
    for i in range(size):
        for j in range(size):
            region = padded[i:i + kernsize, j:j + kernsize]
            if j>size//2:
                new[i, j] = square_matrix[i,j]
            else:
                new[i, j] = np.sum(region * dataval)
    plt.imshow(new, cmap='gray')
    plt.axis('off')
    plt.show()</pre>
##
2.5
##
RGB blur only part of image 
```python
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
def blurgb_part(bluramount,kernsize,picture_name,rmn,rmx,cmn,cmx):
    #makes the origional
    img = Image.open(picture_name).convert('RGB')
    img_matrix = np.array(img) / 255.0
    h, w,c = img_matrix.shape
    crop_size = min(h, w)
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    col = img_matrix[start_h:start_h + crop_size, start_w:start_w + crop_size, :]

    #makes the kernel
    t_value = bluramount
    def g(t, x, y):
        return (1 / (4 * np.pi * t)) * np.exp(-(x**2 + y**2) / (4 * t))
    halfkern = kernsize // 2
    datagrid = np.zeros((kernsize,kernsize),dtype='object')
    dataval = np.zeros((kernsize,kernsize),dtype='float')
    for ix, i in enumerate(range(-halfkern, halfkern+1)):
        for iy, j in enumerate(reversed(range(-halfkern, halfkern+1))):
            datagrid[iy,ix] = i,j
            dataval[iy,ix]= round(g(t_value, j, i),20)
    dataval /= dataval.sum()

    #implementation
    rmn,rmx= rmn/100,rmx/100
    cmn,cmx= cmn/100,cmx/100
    rst=round(int(rmn*crop_size),1)
    rnd=round(int(rmx*crop_size),1)
    cst=round(int(cmn*crop_size),1)
    cnd=round(int(cmx*crop_size),1)
    new = np.copy(col)
    for t in range(c):
        padded = np.pad(col[:,:,t], kernsize // 2, mode='edge')
        for i in range(rst,rnd):
            for j in range(cst,cnd):
                region = padded[i:i + kernsize, j:j + kernsize]
                new[i, j, t] = np.sum(region * dataval)
    #readjust and print
    plt.imshow(new)
    plt.title(f"Blur Amount: {bluramount}, Kernel: {kernsize}x{kernsize}")
    plt.axis('off')
    plt.show()
blurgb_part(10,23, 'ironman.jpg',30,70,30,70)
```

##
2.6
##
RGB blur effect
<pre>import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
def blurgb(bluramount,kernsize,picture_name):
    #makes the origional
    img = Image.open(picture_name).convert('RGB')
    img_matrix = np.array(img) / 255.0
    h, w,c = img_matrix.shape
    crop_size = min(h, w)
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    col = img_matrix[start_h:start_h + crop_size, start_w:start_w + crop_size, :]

    #makes the kernel
    t_value = bluramount
    def g(t, x, y):
        return (1 / (4 * np.pi * t)) * np.exp(-(x**2 + y**2) / (4 * t))
    halfkern = kernsize // 2
    datagrid = np.zeros((kernsize,kernsize),dtype='object')
    dataval = np.zeros((kernsize,kernsize),dtype='float')
    for ix, i in enumerate(range(-halfkern+1, halfkern)):
        for iy, j in enumerate(reversed(range(-halfkern+1, halfkern))):
            datagrid[iy,ix] = i,j
            dataval[iy,ix]= round(g(t_value, i, j),20)
    dataval /= dataval.sum()

    #implementation
    new = np.zeros((crop_size, crop_size,3))
    for t in range(c):
        padded = np.pad(col[:,:,t], kernsize // 2, mode='edge')
        for i in range(crop_size):
            for j in range(crop_size):
                region = padded[i:i + kernsize, j:j + kernsize]
                new[i, j,t] = np.sum(region * dataval)
    #readjust and print
    plt.imshow(new)
    plt.title(bluramount)
    plt.axis('off')
    plt.show()
blurgb(100,23, 'ironman.jpg')</pre>
##
circular blur effect
<pre>import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
def blurgb_circle(bluramount,kernsize,picture_name,size):
    #makes the origional
    img = Image.open(picture_name).convert('RGB')
    img_matrix = np.array(img) / 255.0
    h, w,c = img_matrix.shape
    crop_size = min(h, w)
    grid = np.zeros((crop_size, crop_size), dtype=object)
    center = crop_size // 2
    for i in range(crop_size):
        for j in range(crop_size):
            grid[i, j] = (i - center, center-j)
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    col = img_matrix[start_h:start_h + crop_size, start_w:start_w + crop_size, :]

    #makes the kernel
    t_value = bluramount
    def g(t, x, y):
        return (1 / (4 * np.pi * t)) * np.exp(-(x**2 + y**2) / (4 * t))
    halfkern = kernsize // 2
    datagrid = np.zeros((kernsize,kernsize),dtype='object')
    dataval = np.zeros((kernsize,kernsize),dtype='float')
    for ix, i in enumerate(range(-halfkern+1, halfkern)):
        for iy, j in enumerate(reversed(range(-halfkern+1, halfkern))):
            datagrid[iy,ix] = i,j
            dataval[iy,ix]= round(g(t_value, i, j),20)
    dataval /= dataval.sum()

    #implementation
    new = np.zeros((crop_size, crop_size,3))
    for t in range(c):
        padded = np.pad(col[:,:,t], kernsize // 2, mode='edge')
        for i in range(crop_size):
            for j in range(crop_size):
                a,b=grid[i,j]
                if np.linalg.norm((a,b), ord=4) <= size:
                    region = padded[i:i + kernsize, j:j + kernsize]
                    new[i, j, t] = np.sum(region * dataval)
                else:
                    new[i, j, t] = col[i, j, t]
    #readjust and print
    plt.imshow(new)
    plt.title(bluramount)
    plt.axis('off')
    plt.show()
blurgb_circle(100,23, 'ironman.jpg',150)</pre>
