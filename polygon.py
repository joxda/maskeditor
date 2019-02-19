from numpy import *
from astropy.io import fits
import pyds9
from scipy.ndimage.measurements import label 
from scipy.ndimage import binary_dilation , binary_erosion, generate_binary_structure
from skimage.measure import approximate_polygon,  find_contours
from matplotlib.path import Path
import sys

tolerance = 0.5

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

def poly_display():
    d.set('regions deleteall')
    for polygon,i in zip(polygons,range(len(polygons))):
            if(len(polygon)>2):
                #print " ".join("%6.2f %6.2f" % (p[0],p[1]) for p in polygon)
                d.set("region command {polygon %s # color=blue fill=0}" % " ".join("%6.2f %6.2f" % (p[0]+1,p[1]+1) for p in polygon))
                #print array(polygon)[:,0]
                d.set('region command {text %6.2f %6.2f # text="%d" color=red}' % (mean(array(polygon)[:,0]),mean(array(polygon)[:,1]),i+1) )# index -> i+1

    for x,y in zip(x_plot,y_plot):
                    d.set("region command {point %6.2f %6.2f # point=x}" % (x+1,y+1))

def get_polygons(point):
    ret_poly=[]
    for polygon in polygons:
        poly_path=Path(polygon)
        test = poly_path.contains_points([point])
        if test[0]:
            ret_poly.append(polygon)
    return ret_poly

def mask_to_polygon(mask_bin):

    structure = ones((3, 3), dtype=int) # this is for 8 connected... could also be 4 connected?
    structure = array([[0,1,0],[1,1,1],[0,1,0]]) # this is for 4 connected

    enlarge = zeros((mask_bin.shape[0]+10,(mask_bin.shape[1]+10)))
    enlarge[5:-5,5:-5] = mask_bin

    labeled, ncomponents = label(enlarge.astype(int), structure)
    offset=5
    print ncomponents
    ret_poly = []
# TAKE INTO ACCOUNT DISTINCT MASK VALUES...
# CAN CONNECTED COMPONENTS BE FOUND AND THEN THE INDEX VALUES MADE UNIQUE?
    print unique(labeled)
    if ncomponents>1:
        print range(ncomponents)
        for i in range(ncomponents):
            print "HERE", i+1
   
        #for i in range(ncomponents):
            test = (labeled==i+1)*1.0             # this picks one connected component
            cont = find_contours(test,0.4)      
            #print cont

            simplier_cont = approximate_polygon(cont[0],tolerance)
            ret_poly.append([(y-offset,x-offset) for x,y in simplier_cont])
            #ret_poly.append([(y,x) for x,y in cont[0]])

    elif ncomponents==1:
            test = (labeled==1.0)             # this picks one connected component
            cont = find_contours(test,0.4)      
            #print cont
            simplier_cont = approximate_polygon(cont[0],tolerance)
            ret_poly.append([(y-offset,x-offset) for x,y in simplier_cont])
       
    return ret_poly


def polypath_to_mask(poly_path):
    x, y = mgrid[:height, :width]
    coors=hstack((y.reshape(-1, 1), x.reshape(-1,1))) 

    m = poly_path.contains_points(coors).reshape(height, width)
    return m

def polygon_to_mask(polygon):
    poly_path=Path(polygon)
    return polypath_to_mask(poly_path)



for p in mask_to_polygon(mask_bin):
    polygons.append(p)
# possibly only use the polygon approximation in ds9  

# potentially create circle/ellipse/rectangle by middle point / radius or middle points +2 or three corners 

# label -> dilate / open / close / erode...


width=data.shape[0]
height=data.shape[1]

inkey=""
while(inkey!="q"):
       poly_display()

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
              polygons=polygons[:-1] # THIS MAY LEAD TO CONFUSION FOR NON USER DEFINED POLYGONS.... # PROBABLY BEST TO KEEP INPUT MASK AND POLYGONS SEPARATE?
              
       if(inkey=="x" and len(polygons)>0): 
           del_poly = get_polygons((float(userinput[1])-1,float(userinput[2])-1))
           for polygon in del_poly:
               polygons.remove(polygon)
               
       if(inkey=="d" and len(polygons)>0): 
           dil_poly = get_polygons((float(userinput[1])-1,float(userinput[2])-1))
           # INSTEAD GET THE MASK PART..?
           for polygon in dil_poly:
                   m = polygon_to_mask(polygon)
                   struct = generate_binary_structure(2, 2)
                   newmask = binary_dilation(m, structure=struct)
                   #print newmask.shape
                   #print newmask
                   newpoly = mask_to_polygon(newmask)

                   #print newpoly
                   polygons.remove(polygon)
                   for np in newpoly:
                       polygons.append(np)

       if(inkey=="e" and len(polygons)>0): 
           dil_poly = get_polygons((float(userinput[1])-1,float(userinput[2])-1))
           for polygon in dil_poly:
                   m = polygon_to_mask(polygon)
                   struct = generate_binary_structure(2, 2)
                   newmask = binary_erosion(m, structure=struct)
                   #print newmask.shape
                   #print newmask
                   newpoly = mask_to_polygon(newmask)

                   print len(newpoly)
                   polygons.remove(polygon)
                   for np in newpoly:
                       polygons.append(np)
                   #poly_display()
        



allmask = zeros(data.shape) # or input mask... 

for polygon,i in zip(polygons,range(len(polygons))):
        mask = polygon_to_mask(polygon)
        allmask = allmask+mask*(i+1)

hdu = fits.PrimaryHDU(allmask)   ### ADD TO PREVIOUS MASK!
hdu.writeto("mask1.fits",overwrite=True)



