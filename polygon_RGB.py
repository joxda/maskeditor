from numpy import *
from astropy.io import fits
import pyds9
from scipy.stats import mode
from scipy.ndimage.measurements import label 
from scipy.ndimage import binary_dilation , binary_erosion, generate_binary_structure
from skimage.measure import approximate_polygon,  find_contours
from matplotlib.path import Path
import sys

tolerance = 1.5

files = []

fil =""
if len(sys.argv)!=4:
   print "python polygon.py IMAGE MASK OUTPUT"
if len(sys.argv)==4:
    fil = sys.argv[1]
    maskfil = sys.argv[2]
    maskout = sys.argv[3]

global polygons
global others

x_plot=[]
y_plot=[]
polygons=[]
others=[]
in_polygons=[]
data=array([[],[]])
allmask=array([[],[]])

sel_poly=[]

#    (possibly find center etc pp.)

# TBD alternative display of the input mask
#     in another frame

# POSSIBLE PROBLEM: CONNECTED REGIONS WITH DIFFERENT MASK VALUES


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
structure = array([[0,1,0],[1,1,1],[0,1,0]]) # this is for 4 connected

# TBD extensions?

#print fil, maskfil
data = fits.open(fil)[0].data

def get_number(ar):
    counts = bincount(ar.flatten())
    return argmax(counts)

def alt_display():
    #d.set('frame 2')
    d.set('regions deleteall')
    allmask = mask # or input mask... 
    gg = mask.max() 

    for polygon,i in zip(polygons,range(len(polygons))):
        nmask = polygon_to_mask(polygon)
        #gg = mask.max() 
        allmask = allmask+nmask*(i+1+gg)
#    for j,other in enumerate(others):
#        n2mask = other_to_mask(other)
#        allmask = allmask+n2mask*(j+len(polygons)+1+gg)
   
   #d.set_np2arr(allmask)
    dataM = where(allmask>0,where(data-20>0,data/10.,100-data),data)
    d.set_np2arr(dataM)
    #d.set('scale linear')
    #d.set('scale limits 0 1')
   
    for x,y in zip(x_plot,y_plot):
                    d.set("region command {point %6.2f %6.2f # point=x}" % (x+1,y+1))

 #   d.set('zoom to fit')

def poly_display():
    alt_display()

def poly_display2():

    # TBD selected different color...
    d.set('regions deleteall')
    ij=0
    for polygon,i in zip(in_polygons,range(len(in_polygons))):
            if(len(polygon)>2):
                #print " ".join("%6.2f %6.2f" % (p[0],p[1]) for p in polygon)
                d.set("region command {polygon %s # color=cyan fill=0}" % " ".join("%6.2f %6.2f" % (p[0]+1,p[1]+1) for p in polygon))
                #print array(polygon)[:,0]
                d.set('region command {text %6.2f %6.2f # text="%d" color=red}' % (mean(array(polygon)[:,0]),mean(array(polygon)[:,1]),i+1) )# index -> i+1
                ij=i
    for polygon,j in zip(polygons,range(len(polygons))):
            if(len(polygon)>2):
                #print " ".join("%6.2f %6.2f" % (p[0],p[1]) for p in polygon)
                d.set("region command {polygon %s # color=blue fill=0}" % " ".join("%6.2f %6.2f" % (p[0]+1,p[1]+1) for p in polygon))
                #print array(polygon)[:,0]
                d.set('region command {text %6.2f %6.2f # text="%d" color=red}' % (mean(array(polygon)[:,0]),mean(array(polygon)[:,1]),j+ij+2) )# index -> i+1
    if sel_poly!=[]:
        sel_polies, sel_in = get_selection()
        for polygon in sel_polies:
            if(len(polygon)>2):
                d.set("region command {polygon %s # color=blue width=5 fill=0}" % " ".join("%6.2f %6.2f" % (p[0]+1,p[1]+1) for p in polygon))
        sel_mask_bin = (isin(mask_label,sel_in))
        sel_in_pol, ml, ns = mask_to_polygon(sel_mask_bin)
        for polygon in sel_in_pol:
            if(len(polygon)>2):
                d.set("region command {polygon %s # color=cyan width=5 fill=0}" % " ".join("%6.2f %6.2f" % (p[0]+1,p[1]+1) for p in polygon))
        
        if(len(sel_poly)>2):
                d.set("region command {polygon %s # color=green width=1 fill=0}" % " ".join("%6.2f %6.2f" % (p[0]+1,p[1]+1) for p in sel_poly))
        
    for x,y in zip(x_plot,y_plot):
                    d.set("region command {point %6.2f %6.2f # point=x}" % (x+1,y+1))

def get_polygons(point):
    global polygons
    ret_poly=[]
    ret_in_poly=[]
    for polygon in polygons:
        poly_path=Path(polygon)
        test = poly_path.contains_points([point])
        if test[0]:
            ret_poly.append(polygon)
#    for polygon in in_polygons:
#        poly_path=Path(polygon)
#        test = poly_path.contains_points([point])
#        if test[0]:
#            ret_in_poly.append(polygon)
#    return ret_poly, ret_in_poly
    value = mask_label[int(point[1])-1,int(point[0])-1]
    inret = []
    #print value
    if value!=0:
        inret=[value]
    return ret_poly, inret

def mask_to_polygon(mask_bini):

#    structure = ones((3, 3), dtype=int) # this is for 8 connected... could also be 4 connected?
###    structure = array([[0,1,0],[1,1,1],[0,1,0]]) # this is for 4 connected

    enlarge = zeros((mask_bini.shape[0]+10,(mask_bini.shape[1]+10)))
    enlarge[5:-5,5:-5] = mask_bini

    labeled, ncomponents = label(enlarge.astype(int), structure)
    offset=0
#    print ncomponents
    ret_poly = []
# TAKE INTO ACCOUNT DISTINCT MASK VALUES...
# CAN CONNECTED COMPONENTS BE FOUND AND THEN THE INDEX VALUES MADE UNIQUE?
#    print unique(labeled)
#    if ncomponents>1:
#        print range(ncomponents)
#        for i in range(ncomponents):
#            print "HERE", i+1
#   
#        #for i in range(ncomponents):
#            test = (labeled==i+1)*1.0             # this picks one connected component
#            cont = find_contours(test,0.4)      
#            #print cont
#
#            simplier_cont = approximate_polygon(cont[0],tolerance)
#            ret_poly.append([(y-offset,x-offset) for x,y in simplier_cont])
#            #ret_poly.append([(y,x) for x,y in cont[0]])
#
#    elif ncomponents==1:
#            test = (labeled==1.0)             # this picks one connected component
#            cont = find_contours(test,0.4)      
#            #print cont
#            simplier_cont = approximate_polygon(cont[0],tolerance)
#            ret_poly.append([(y-offset,x-offset) for x,y in simplier_cont])
#       

    cont = find_contours(mask_bini,0.4)
    for c in cont:
        simplier_cont = approximate_polygon(c,tolerance)
        ret_poly.append([(y-offset,x-offset) for x,y in simplier_cont])

    return ret_poly,labeled[5:-5,5:-5], ncomponents

def other_to_mask(other):
   global polygons
   y, x = mgrid[:height, :width]

   if other[0]=="box":
      x1 = other[1][0]
      x2 = other[2][0]
      y1 = other[1][1]
      y2 = other[2][1]
      if x1 > x2:
         tmp = x1
         x1 = x2
         x2 = tmp
      if y1 > y2:
         tmp = y1
         y1 = y2
         y2 = tmp
      m = where( (x>=x1) & (x<=x2) & (y>=y1) & (y<=y2), 1, 0 )

   if other[0]=="circle" or other[0]=="ellipse":
      x_cen = other[1][0]
      y_cen = other[1][1]

      smi = sqrt( (x_cen-other[2][0])**2 + (y_cen-other[2][1])**2 )
      sma = smi
      angle = 0

      if other[0]=="ellipse":
         angle = arctan2(y_cen-other[2][1],x_cen-other[2][0])-pi/2.
         angle2 = arctan2(y_cen-other[3][1],x_cen-other[3][0])
         phi = angle2-angle

         y_a = - (other[3][0]-x_cen)*sin(angle) + (other[3][1]-y_cen)* cos(angle)
         smi = y_a / sin(phi) 
         
      ba = smi/sma

      x=x-x_cen
      y=y-y_cen

      xcosang = x*cos(angle)
      xsinang = x*sin(angle)
      xtemp =  xcosang + y*sin(angle)
      ytemp = -xsinang + y*cos(angle)
      dist = sqrt( (xtemp/ba)**2 + (ytemp)**2 )
      m = where(dist <= sma, 1, 0)

   poly,lab, n = mask_to_polygon(m)
   polygons.append(poly[0])
   others=[]

def polypath_to_mask(poly_path):
    x, y = mgrid[:height, :width]
    coors=hstack((y.reshape(-1, 1), x.reshape(-1,1))) 

    m = poly_path.contains_points(coors).reshape(height, width)
    return m

def polygon_to_mask(polygon):
    poly_path=Path(polygon)
    return polypath_to_mask(poly_path)



def update_mask():
    global mask, mask_bin,mask_label,in_polygons,ncomponents
    mask_bin = (mask>0)
    in_polygons, mask_label, ncomponents = mask_to_polygon(mask_bin)
    
def reset_mask():
    global mask,mask_orig,polygons
    mask=mask_orig
    update_mask()
    polygons=[]

def get_selection():
    sel_mask = polygon_to_mask(sel_poly)
    polies=[]
    for polygon in polygons:
        test_mask = polygon_to_mask(polygon)
        if any(logical_and(sel_mask,test_mask)):
            polies.append(polygon)
    i=unique(mask_label[where(sel_mask)])
    return polies, delete(i,where(i==0))


def stats(data,mask):
    print "MEAN:   ", mean(data), mean(data[mask==0])
    print "STDEV:  ", std(data), std(data[mask==0])
    print "MEDIAN: ", median(data), median(data[mask==0])
#    print "MODE:   ", mode(data,axis=None)
    print "MIN:    ", data.min(),data[mask==0].min()
    print "MAX:    ", data.max(), data[mask==0].max()
    


# work with masks...
mask =fits.open(maskfil)[0].data
mask_orig = mask.copy()

update_mask()
#mask_bin = (mask>0)
#structure = array([[0,1,0],[1,1,1],[0,1,0]]) # this is for 4 connected
#enlarge = zeros((mask_bin.shape[0]+10,(mask_bin.shape[1]+10)))
#enlarge[5:-5,5:-5] = mask_bin
#in_polygons, mask_label, ncomponents = mask_to_polygon(mask_bin)
#

#print mask_bin.shape
# CHECK UNIQUE VALUES?
# the value of the mask should possibly remain so that the label is only used for the indexing / selection...

d.set('rgb')
d.set('rgb red')
d.set_np2arr(data)
d.set('rgb green')
d.set_np2arr(data)
d.set('rgb blue')
d.set_np2arr(data)

d.set('rgb lock scale')
d.set('rgb lock scalelimits')
d.set('rgb lock colorbar')
d.set('scale log')
d.set('scale limits -1 5000')

d.set('zoom to fit')
d.set('rgb red')
d.set('mode','region')
#for p in mask_to_polygon(mask_bin):
#    polygons.append(p)
# possibly only use the polygon approximation in ds9  

# potentially create circle/ellipse/rectangle by middle point / radius or middle points +2 or three corners 

# label -> dilate / open / close / erode...


width=data.shape[1]
height=data.shape[0]

inkey=""

while(inkey!="q"):
       poly_display()

       try:
           userinput = d.get('iexam key coordinate image').split()
       except:
           break
       inkey = userinput[0].lower()


       if(inkey=="h"):
          print("|--------------------------------------|")
          print("| q - quit editing and write mask file |")
          print("| h - print this help                  |")
          print("| r - reset mask                       |")
          print("| ------------------------------------ |")
          print("| c - mark point                       |")
          print("| p - new polygon                      |")
          print("| 2 - new circle                       |")
          print("| 3 - new ellipse                      |")
          print("| 4 - new box                          |")
          print("| ------------------------------------ |")
          print("| x - delete masked area               |")
          print("| d - dilate masked area               |")
          print("| e - erode masked area                |")
          print("| b - undo last                        |")
          print("| ----------- do not use ------------- |")
          print("| (m - print some statistics)          |")
          print("| (s - select m regions w triangle     |")
          print("| (a - select whole image)             |")
          print("|--------------------------------------|")
           
       if(inkey=="r"):
           reset_mask()


       if(inkey=="c"):
           x_plot.append(float(userinput[1]))
           y_plot.append(float(userinput[2]))

       if(inkey=="s"):
           if sel_poly==[]:
               x_plot.append(float(userinput[1]))
               y_plot.append(float(userinput[2]))
               if(len(x_plot)>2):
                   sel_poly = [(x,y) for x,y in zip(x_plot,y_plot)]
                   x_plot=[]
                   y_plot=[]
           else:
               sel_poly=[]

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
             # POSSIBLY JUST GET THE INDEX VALUE AT THE COORDINATES....?
            
       if(inkey=="2"):
           x_plot.append(float(userinput[1]))
           y_plot.append(float(userinput[2]))
           if(len(x_plot)>=2):
               others.append(["circle"]+[(x,y) for x,y in zip(x_plot[:2],y_plot[:2])])
               x_plot=[]
               y_plot=[]
               for other in others:
                  other_to_mask(other)
               others=[]
              
       if(inkey=="3"):
           x_plot.append(float(userinput[1]))
           y_plot.append(float(userinput[2]))
           if(len(x_plot)>2):
               others.append(["ellipse"]+[(x,y) for x,y in zip(x_plot[:3],y_plot[:3])])
               x_plot=[]
               y_plot=[]
               for other in others:
                  other_to_mask(other)
               others=[]

       if(inkey=="4"):
           x_plot.append(float(userinput[1]))
           y_plot.append(float(userinput[2]))
           if(len(x_plot)>=2):
               others.append(["box"]+[(x,y) for x,y in zip(x_plot[:2],y_plot[:2])])
               x_plot=[]
               y_plot=[]
               for other in others:
                  other_to_mask(other)
               others=[]

       if(inkey=="m"):
           # Stats...
           allmask = mask # or input mask... 

           gg = mask.max() 
           for polygon,i in zip(polygons,range(len(polygons))):
               nmask = polygon_to_mask(polygon)
               allmask = allmask+nmask*(i+1+gg)
           if sel_poly!=[]:
                indi = where(polygon_to_mask(sel_poly))
                    
                stats(data[indi],allmask[indi])
           else:
                stats(data,allmask)
            
       if(inkey=="a"):
           # Stats...
           sel_poly=[(0,0),(height,0),(height,width),(0,width)]

       if(inkey=="x"): 
           if sel_poly==[]:
               del_poly, del_in = get_polygons((float(userinput[1])-1,float(userinput[2])-1))
           else:
               del_poly, del_in = get_selection()
               print del_in

           for di in del_in:
                   #ind_del = where(mask_label==di)
                   ind_del = where(mask_label==di)
                   mask[ind_del]=0
                   update_mask()
                   
           for polygon in del_poly: 
                   polygons.remove(polygon)
           sel_poly=[] 
              
       if(inkey=="d"): 
           if sel_poly==[]:
               dil_poly, dil_in = get_polygons((float(userinput[1])-1,float(userinput[2])-1))
           else:
               dil_poly, dil_in = get_selection()
           # INSTEAD GET THE MASK PART..?
           struct = structure #generate_binary_structure(2, 2)
           for polygon in dil_poly:
                   m = polygon_to_mask(polygon)
                   newmask = binary_dilation(m, structure=struct)
                   #print newmask.shape
                   #print newmask
                   newpoly,tmp,tmp1 = mask_to_polygon(newmask)

                   #print newpoly
                   polygons.remove(polygon)
                   for np in newpoly:
                       polygons.append(np)
           #for di in dil_in:
                   #m = polygon_to_mask(polygon)
                   #ind=where(m)
                   #num=get_number(mask_label[ind])                
           m = isin(mask_label,dil_in)
           newmask = binary_dilation(m, structure=struct)
           ind_mod = where(newmask)
           mask[ind_mod]=1
           update_mask()

#       if(inkey=="9"):
#           d.set("regions system image")
#           a= d.get("regions","list")
#           print a


       if(inkey=="e"): 
           if sel_poly==[]:
               dil_poly, dil_in = get_polygons((float(userinput[1])-1,float(userinput[2])-1))
           else:
               dil_poly, dil_in = get_selection()
           struct = structure #generate_binary_structure(2, 2)
           for polygon in dil_poly:
                   m = polygon_to_mask(polygon)
                   newmask = binary_erosion(m, structure=struct)
                   #print newmask.shape
                   #print newmask
                   newpoly,tmp,tmp1 = mask_to_polygon(newmask)

                   #print len(newpoly)
                   polygons.remove(polygon)
                   for np in newpoly:
                       polygons.append(np)
                   #poly_display()
           #for di in dil_in:
                   #m = polygon_to_mask(polygon)
                   #ind=where(m)
                   #num=get_number(mask_label[ind])
           mi = isin(mask_label,dil_in)
           newmask = binary_erosion(mi, structure=struct)
           ind_mod = where(newmask!=mi)
           mask[ind_mod]=0
           update_mask()
         

allmask = zeros(data.shape) # or input mask... 
allmask = mask # or input mask... 

gg = mask.max() 

for polygon,i in zip(polygons,range(len(polygons))):
        nmask = polygon_to_mask(polygon)
        allmask = allmask+nmask*(i+1+gg)
#for j,other in enumerate(others):
#        n2mask = other_to_mask(other)
#        allmask = allmask+n2mask*(j+len(polygons)+1+gg)

hdu = fits.PrimaryHDU(allmask)   ### ADD TO PREVIOUS MASK!
hdu.writeto(maskout,overwrite=True)



