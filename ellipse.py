import matplotlib.pyplot as plt
from numpy import *
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.image import AxesImage
from glob import glob
from astropy.convolution import convolve, Gaussian2DKernel, Tophat2DKernel

from astropy.io import fits
from random import shuffle

from glob import glob
import sys

files = []

if len(sys.argv)>1:
    files = glob(sys.argv[1])



def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

x_plot=[]
y_plot=[]
data=array([[],[]])
pars = ()
select_dat = []
global_vmin = 0
global_vmax = 0

shift_pressed = False
i_pressed = False

def on_key(event):
    global shift_pressed
    global i_pressed
    if event.key=="shift":
        shift_pressed = True
    if event.key=="i":
        i_pressed = True
        print("i")

def off_key(event):
    global shift_pressed
    global i_pressed
    if event.key=="shift":
        shift_pressed = False
    if event.key=="i":
        i_pressed = False
        print("I")


def change_lim(event):
    global global_vmin
    global global_vmax
    global data
    #print global_vmin
    if shift_pressed:
        global_vmax += event.step
        if global_vmax < global_vmin:
            global_vmax = global_vmin+0.1
    else:
        global_vmin += event.step/10.
        if global_vmin > global_vmax:
            global_vmin = global_vmax-0.1
    elements = plt.gca().get_children()
    ind_ima = where(array([isinstance(i,AxesImage) for i in elements]))[0]
    elements[ind_ima[-1]].remove()
    plt.imshow(log10(data+0.0001), origin='lower', cmap='gray',vmin=global_vmin, vmax=global_vmax)
    plt.draw()

def isophotal(level):
    #data_re=rebin(data[0].data,(100,100),)
    global data
    tophat_kernel = Tophat2DKernel(5)
    data_re = convolve(data, tophat_kernel)
    #data_re = data
    #print(level)
    pix_ind=where(floor(log10(data_re)*50)==floor(log10(level)*50))
    #print(pix_ind)
    X = array(pix_ind[1])
    Y = array(pix_ind[0])
    #plt.plot(X,Y,'.')
    ipars = fit_ellipse(X,Y)
    ell = plot_ellipse(plt.gca(),ipars)
    plt.draw()

def onclick(event):
  global i_pressed
  global data
  if i_pressed:
    xpos = int(event.xdata)
    ypos = int(event.ydata)

    level = median(data[ypos-2:ypos+3,xpos-2:xpos+2])

    isophotal(level)

  else:
    global x_plot
    global y_plot
    global pars
    #global data
    global select_dat

    elements = plt.gca().get_children()

    if event.dblclick:
        ind_ima = where(array([isinstance(i,AxesImage) for i in elements]))[0]
        elements[ind_ima[-1]].remove()
        global_vmin=0
        global_vmax=log10(data.max())
        plt.imshow(log10(data+0.0001), origin='lower', cmap='gray',vmin=global_vmin, vmax=global_vmax)

    else:


        ind_points = where(array([isinstance(i,Line2D) for i in elements]))[0]
        ind_ellipse = where(array([isinstance(i,Ellipse) for i in elements]))[0]
        ind_contours = where(array([isinstance(i,LineCollection) for i in elements]))[0]

        #print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #      ('double' if event.dblclick else 'single', event.button,
        #       event.x, event.y, event.xdata, event.ydata))

        if event.button == 3 and len(x_plot)>0:
            x_plot=x_plot[:-1]
            y_plot=y_plot[:-1]
            select_dat=select_dat[:-1]

            if(len(x_plot)<5) and ind_ellipse.shape[0] == 1:
                elements[ind_ellipse[0]].remove()
                if ind_points.shape[0] > 1:
                    elements[ind_points[1]].remove()
        else:
            x_plot.append(event.xdata)
            y_plot.append(event.ydata)
            select_dat.append(median(data[int(event.ydata)-2:int(event.ydata)+3,int(event.xdata)-2:int(event.xdata)+3]))

            if ind_contours.shape[0]>0:
               # print ind_contours
                cont=[]
                for e in ind_contours:
                    cont.append(elements[e])
                for co in cont:
                    co.remove()

            tophat_kernel = Tophat2DKernel(5)
            data_re = convolve(data, tophat_kernel)
            plt.contour(data_re,[median(select_dat)*0.5,median(select_dat),median(select_dat)*2],alpha=0.5)

#    elements = plt.gca().get_children()
#    ind_points = where(array([isinstance(i,Line2D) for i in elements]))[0]
#    ind_ellipse = where(array([isinstance(i,Ellipse) for i in elements]))[0]

        if ind_points.shape[0] == 0:
            plt.plot(array(x_plot),array(y_plot),'y.')
        else:
            #print ind_points
            #print elements[ind_points[0]]
            elements[ind_points[0]].set_data(array(x_plot),array(y_plot))

        if(len(x_plot)>4):
            pars = fit_ellipse(array(x_plot),array(y_plot))
            if ind_ellipse.shape[0] == 1:
                elements[ind_ellipse[0]].remove()
            ell = plot_ellipse(plt.gca(),pars)

            if ind_points.shape[0] == 1:
                plt.plot(pars[0],pars[1],'rx')
            else:
                elements[ind_points[1]].set_data(pars[0],pars[1])

    plt.draw()



#def fit_ansac_ellipse(inX,inY,fixed_center=True):

#    if fixed_center:
#        min = 3
#    else:
#        min = 5
#
#    x_cen = data.shape[1]/2
#    y_cen = data.shape[0]/2
#
#    x_sub = list(itertools.combinations(inX, min))
#    y_sub = list(itertools.combinations(inY, min))
#
#    for xsu,ysu in zip(x_sub,ysub):
#        pell = fit_ellipse(x_sub,y_sub,center_fixed)
#
#        theta=arange(0,2*pi,0.02)
#        allx = pell[2] * pell[4] * cos(theta-radians(pell[3])) + pell[0]
#        ally = pell[2] * sin(theta-radians(pell[3])) + pell[1]




def fit_ellipse(inX,inY,center_fixed=True):
    x_cen = data.shape[1]/2
    y_cen = data.shape[0]/2

    X = array([(inX)]).T-x_cen
    Y = array([(inY)]).T-y_cen

    #print X.shape

    if center_fixed:
        A = hstack([X**2, X * Y, Y**2])
    else:
        A = hstack([X**2, X * Y, Y**2, X, Y])

    b = ones_like(X)

    out = linalg.lstsq(A, b,rcond=None)
    #print out[1]
    x = out[0].squeeze()
    #print('The ellipse is given by {0:.3}x^2 + {1:.3}xy+{2:.3}y^2+{3:.3}x+{4:.3}y = 1'.format(x[0], x[1],x[2],x[3],x[4]))


    A=x[0]
    B=x[1]
    C=x[2]

    if center_fixed:
        D=0
        E=0
    else:
        D=x[3]
        E=x[4]
    F=-1

    a = (-sqrt(2*(A*E**2+C*D**2-B*D*E+(B**2-4*A*C)*F)*(A+C+sqrt((A-C)**2+B**2))))/(B**2-4*A*C)
    b = (-sqrt(2*(A*E**2+C*D**2-B*D*E+(B**2-4*A*C)*F)*(A+C-sqrt((A-C)**2+B**2))))/(B**2-4*A*C)

    X_c = (2*C*D-B*E)/(B**2-4*A*C)+x_cen
    Y_c = (2*A*E-B*D)/(B**2-4*A*C)+y_cen



    theta = (0 if A<C else pi) if B==0 else arctan((C-A-sqrt((A-C)**2+B**2))/B)

    theta = degrees(theta)

    # OBS changed to the version installed on the macbook! - even though I dont see why...
    ba = b/a

    return X_c,Y_c,a,b, theta, ba


def plot_ellipse(ax,pars):
    X_c,Y_c,a,b,theta,ba = pars
    # OBS changed according to the updated definition of ba
    ell = Ellipse(xy=(X_c,Y_c),width=2*a,height=2*a*ba,angle=(theta),fill=False,lw=1,zorder=100)
    ax.add_artist(ell)
    ell.set_alpha(1.0)
    ell.set_facecolor((1,0.2,0.2))
    ell.set_edgecolor((1,0,0))
    return ell

if files == []:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.update()
    files = filedialog.askopenfilenames()
    root.update()

shuffle(list(files))


print("# %20s  %9s %9s %8s %8s %9s %3s" % ("filename","x_cen","y_cen","a","b","theta(CCW_from_+x)","b/a"))


for f in files:
    #global x_plot
    #global y_plot
    #global pars
    #global data
    #global global_vmin
    #global global_vmax
    pars = ()

    x_plot=[]
    y_plot=[]

    fig, ax = plt.subplots(figsize=(6, 6))
    data = fits.open(f)
    data = data[0].data
    global_vmin = 0
    global_vmax = log10(data.max())
    plt.title(f)
    plt.imshow(log10(data+0.0001), origin='lower', cmap='gray',vmin=global_vmin)
        #plt.contour(data,[0.3,0.8,1,10],alpha=0.5)


        #ax.plot(np.random.rand(10))
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('scroll_event',change_lim)
    fig.canvas.mpl_connect('key_press_event',on_key)
    fig.canvas.mpl_connect('key_release_event',off_key)

    plt.show()

    print("%23s " % f, " ".join("%9.3f" % p for p in list(pars)))
