##
1.6
##
newtons method 
```python
def newton(fx,dydx):
    sol=[]
    for i in range(-10,10,2):
        run = 0
        ame = 0.00000001
        estimate=i
        if dydx(estimate)==0:
            print("try a new estimate")
            run=1
        x=estimate
        while run==0:
            yel=0
            while yel<20 and run==0:
                y=x-(fx(x)/dydx(x))
                error=abs(fx(x)/dydx(x))
                x=y
                yel+=1
                if error<ame:
                    x=round(x,4)
                    if x in sol:
                        run=1
                    else:
                        sol.append(x)
                        run=1
            run=1
    print("solution is",sol)

def maxmin(sol, dydx):
    for i in range(len(sol)):
        rep=dydx(sol[i])
        if rep>0:
            print(sol[i],'is a minimum')
        if rep<0:
            print(sol[i],'is a maximum')

def fx(x):
    return 4*x**4-6*x**2+3*x-(11/4)
def dydx(x):
    return 16*x**3-12*x+3

newton(fx,dydx)
```

##
1.8
##
one variable gradient decent 
```python
import math
def grade_with_x(fx, dydx, step, min, max, goal=0.000001):
    sol=[]
    for i in range(min,max+1,1):
        count = 0
        done = 0
        x = i
        while count<1000 and done==0:
            y=x-step*dydx(x)
            diff= abs(y-x)
            if diff < goal or y<=min or y>=max:
                if y>1000000 or y<-1000000:
                    y='inf'
                if y<=min:
                    y=min
                if y>=max:
                    y=max
                else:
                    y=round(y,4)
                if y in sol:
                    done=1
                else:
                    sol.append(y)
                    done=1
            else:
                x=y
                count+=1
    y_value=[]
    b=0
    for a in sol:
        if a=='inf':
            print('golbal min at negativite infinity')
            b=1
        else:
            print('min at', a)
            y_value.append(fx(a))

    if b==0:
        bigmin=1000000
        for b in y_value:
            if b<bigmin:
                bigmin=b
        print('global min of', bigmin)
def f(x):
    return math.sin(x)-0.1*x**2
def dydx(x):
    return math.cos(x)-0.2*x
grade_with_x(f,dydx,0.1,-1,1)
```

###
1.9
##
two variable gradient decent with cicular norm 

```python
import math
def gradecon(z,dzdx,dzdy,step, goal=0.00001):
    def in_bounds(x,y):
        return x**2+y**2<=16
    def project(x, y):
        dist = math.hypot(x, y)
        scale = 4.0 / dist
        return x * scale, y * scale
    sol = []
    for xstart in range(-40,40,5):
        for ystart in range(-40,40,5):
            x = xstart/10
            y = ystart/10
            if not in_bounds(x,y):
                continue
            count = 0
            done=0
            while count < 100 and done == 0:
                newx=x-step*dzdx(x,y)
                newy=y-step*dzdy(x,y)
                if not in_bounds(newx,newy):
                    newx, newy= project(newx,newy)
                diff = abs(y-newy)+abs(x-newx)
                if diff < goal:
                    x=round(newx,4)
                    y=round(newy,4)
                    if(x,y) in sol:
                        done=1
                    else:
                        sol.append((x,y))
                        done=1
                else:
                    count+=1
                    x=newx
                    y=newy
    print(sol)
    bigmin = 1000000
    for a in sol:
        x, y = a
        if z(x, y) < bigmin:
            bigmin = z(x, y)
            bif = (x, y)
    print('global min at', bif, 'of', bigmin)
```
  ###
  1.10
  ##
  four variabe gradient decent 
  ```python
def gradefourvar(f,dfdx,dfdy,dfdz,dfda,step, goal=1e-6):
    xmn = -1
    xmx = 1
    ymn = -1
    ymx = 1
    zmn = -1
    zmx = 1
    amn = -1
    amx = 1
    sol=[]
    for xstart in range(xmn,xmx+1,1):
        for ystart in range(ymn,ymx+1,1):
            for zstart in range(zmn,zmx+1,1):
                for astart in range(amn,amx+1,1):
                    count = 0
                    done = 0
                    x = xstart
                    y=ystart
                    z=zstart
                    a=astart
                    while count < 10000 and done == 0:
                        newx=x-step*dfdx(x,y,z,a)
                        newy=y-step*dfdy(x,y,z,a)
                        newz=z-step*dfdz(x,y,z,a)
                        newa=a-step*dfda(x,y,z,a)
                        diff = abs(y-newy)+abs(x-newx)+abs(z-newz)+abs(a-newa)
                        if abs(f(newx,newy,newz,newa))>10000:
                            break
                        if diff < goal:
                            x=round(newx,3)
                            y=round(newy,3)
                            z=round(newz,3)
                            a=round(newa,3)
                            if(x,y,z,a) in sol:
                                done=1
                            else:
                                sol.append((x,y,z,a))
                        else:
                            count+=1
                            x=newx
                            y=newy
                            z=newz
                            a=newa
    print(sol)
    bigmin=1000000

    for num in sol:
        x, y, z, a = num
        if f(x,y,z,a)<bigmin:
            bigmin=f(x,y,z,a)
            bif=(x,y,z,a)
    print('global min at', bif, 'of', bigmin)
def f(x,y,z,a):
    return x**2+3*y**2+2*z**2+a**2+x*y-2*y*z+z*a-4*x+y+2*z-a+10
def dfdx(x,y,z,a):
    return 2*x+y-4
def dfdy(x,y,z,a):
    return 6*y+x-2*z+1
def dfdz(x,y,z,a):
    return 4*z-2*y+a+2
def dfda(x,y,z,a):
    return 2*a+z-1
gradefourvar(f, dfdx,dfdy,dfdz,dfda,0.2)
```
