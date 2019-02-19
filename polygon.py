from numpy import *
from astropy.io import fits
import pyds9
from scipy.ndimage.measurements import label 
from skimage.measure import approximate_polygon,  find_contours
from matplotlib.path import Path
import sys

tolerance = 1.5

files = []

fil =""
#f len(sys.argv)!=3:
#   print "HELP"
if len(sys.argv)==3:
    fil = sys.argv[1]
    maskfil = sys.argv[2]


x_plot=[]
y_plot=[]
polygons=[]
data=array([[],[]])
allmask=array([[],[]])






if fil == "":
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.update()
    fil = filedialog.askopenfilename()
    root.update()
    maskfil = filedialog.askopenfilename()
    root.update()


d = pyds9.DS9('fool')


# TBD extensions?

print fil, maskfil
data = fits.open(fil)[0].data



# work with masks...
mask =fits.open(maskfil)[0].data

mask_bin = (mask>0)
print mask_bin.shape
# CHECK UNIQUE VALUES?
# the value of the mask should possibly remain so that the label is only used for the indexing / selection...
d.set_np2arr(data)

d.set('scale log')
d.set('scale limits -10 10000')

d.set('zoom to fit')


structure = ones((3, 3), dtype=int) # this is for 8 connected... could also be 4 connected?
structure = array([[0,1,0],[1,1,1],[0,1,0]]) # this is for 4 connected

labeled, ncomponents = label(mask_bin.astype(int), structure)

# TAKE INTO ACCOUNT DISTINCT MASK VALUES...
# CAN CONNECTED COMPONENTS BE FOUND AND THEN THE INDEX VALUES MADE UNIQUE?

if ncomponents>1:
  for i in range(ncomponents)[1:]:
    
    test = (labeled==i)*1.0             # this picks one connected component
    cont = find_contours(test,0.4)      

    simplier_cont = approximate_polygon(cont[0],tolerance)

    polygons.append([(y,x) for x,y in simplier_cont])
# possibly only use the polygon approximation in ds9  

# potentially create circle/ellipse/rectangle by middle point / radius or middle points +2 or three corners 

# label -> dilate / open / close / erode...


inkey=""
while(inkey!="q"):
       for polygon,i in zip(polygons,range(len(polygons))):
            if(len(polygon)>2):
                print " ".join("%6.2f %6.2f" % (p[0],p[1]) for p in polygon)
                d.set("region command {polygon %s # color=blue fill=0}" % " ".join("%6.2f %6.2f" % (p[0],p[1]) for p in polygon))
                print array(polygon)[:,0]
                d.set('region command {text %6.2f %6.2f # text="%d" color=red}' % (mean(array(polygon)[:,0]),mean(array(polygon)[:,1]),i+1) )# index -> i+1

       for x,y in zip(x_plot,y_plot):
                    d.set("region command {point %6.2f %6.2f # point=x}" % (x,y))

       try:
           userinput = d.get('iexam key coordinate image').split()
       except:
           break
       inkey = userinput[0].lower()
       if(inkey=="c"):
           x_plot.append(float(userinput[1]))
           y_plot.append(float(userinput[2]))
       if(inkey=="p"):
           x_plot.append(float(userinput[1]))
           y_plot.append(float(userinput[2]))
           if(len(x_plot)>2):
               polygons.append([(x,y) for x,y in zip(x_plot,y_plot)])
               x_plot=[]
               y_plot=[]
       if(inkey=="b"):
           if(len(x_plot)>0):
              x_plot=x_plot[:-1]
              y_plot=y_plot[:-1]
           else:
              polygons=polygons[:-1]
       if(inkey=="x" and len(polygons)>0):  # MAYBE EVEN BETTER: DETERMINE WHICH ONE TO DELETE BY THE MOUSE POSITION!! | THEN IN A SIMILAR ERODE|DILATE|OPEN|CLOSE ETC COULD WORK
            if(len(polygons)==1):
                polygons=[]
            else:
                index=input("Index of polygon to delete:")
                del polygons[index-1]
        
       d.set('regions deleteall')


    

width=data.shape[0]
height=data.shape[1]

allmask = zeros(data.shape) # or input mask... 
for polygon,i in zip(polygons,range(len(polygons))):
        poly_path=Path(polygon)

        x, y = mgrid[:height, :width]
        coors=hstack((y.reshape(-1, 1), x.reshape(-1,1))) 

        mask = poly_path.contains_points(coors)
        allmask = allmask+mask.reshape(height, width)*(i+1)

hdu = fits.PrimaryHDU(allmask)   ### ADD TO PREVIOUS MASK!
hdu.writeto("mask1.fits",overwrite=True)



