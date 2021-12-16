import numpy as np
import sys, os
import math
from netCDF4 import Dataset
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from skimage import measure
import argparse
import copy
import scipy.ndimage

parser = argparse.ArgumentParser()

# Required command line arguments
#parser.add_argument('diri', help='data input directory')
parser.add_argument('filename', help='data input filename')

# Optional command line arguments
parser.add_argument('--topofile', help='topography filename')
parser.add_argument('-p','--plotname', help='plot name and path')
parser.add_argument('-pd','--plotdetail', help='flag: plot each step', action="store_true")
parser.add_argument('-p_red','--pres_red', help='flag: pressure reduction to sea level', action="store_true")
parser.add_argument('-2D','--two_dim', help='flag: two dimensional cyclone mask', action="store_true")
parser.add_argument('-t0','--timestep', help='timestep', type=int, default=0)
parser.add_argument('-k0','--modelevel', help='model level', type=int)

parser.add_argument('-psn','--pname', help='pressure variable name')
parser.add_argument('-topon','--toponame', help='topography variable name')
parser.add_argument('-latn','--latname', help='latitude variable name')
parser.add_argument('-lonn','--lonname', help='longitude variable name')
parser.add_argument('-zn','--zname', help='height variable name')
parser.add_argument('-tn','--tname', help='time variable name')
parser.add_argument('-tempn','--tempname', help='temperature variable name')

parser.add_argument('-minlat','--minlat', help='1th lat for subdomain', type=float)
parser.add_argument('-maxlat','--maxlat', help='2th lat for subdomain', type=float)
parser.add_argument('-minlon','--minlon', help='1th lon for subdomain', type=float)
parser.add_argument('-maxlon','--maxlon', help='2th lon for subdomain', type=float)

parser.add_argument('-c','--cyclic', help='slp data cyclic in x-direction', action="store_true")
parser.add_argument('-o','--outfile', help='output file')

cl_args = parser.parse_args() # Command line argument parser

#diri=cl_args.diri             # Input directory
filename=cl_args.filename     # Input file name
print( 'Inputfile: ' + filename )

if cl_args.topofile: 
    topofilename = cl_args.topofile
    print( 'with topography file: ' + topofilename )
else:
    topofilename = filename

if cl_args.two_dim:
    two_dim = True
    print( "Two (spatial) dimension cyclone mask" )
else:
    two_dim = False
    print( "Three (spatial) dimension cyclone mask" )
    
if cl_args.pres_red: 
    p_redu = True
    print( "Pressure reduction true" )
else:
    p_redu = False
    print( "Pressure reduction false" )

# Name of surface pressure field in file
if cl_args.pname:
    pname = cl_args.pname
else:    
    pname = 'PS'
# Name of topography field in file
if cl_args.toponame:
    toponame = cl_args.toponame
else:    
    toponame = 'TOPOGRAPHY'
# Name of latitude dimension in file
if cl_args.latname:
    latname = cl_args.latname
else:    
    latname = 'lat'
# Name of longitude dimension in file
if cl_args.lonname:
    lonname = cl_args.lonname
else:    
    lonname = 'lon'
# Name of height dimension in file
if cl_args.zname:
    heightname = cl_args.zname
else:
    heightname = 'height'
# Name of time   dimension in file
if cl_args.tname:
    timename = cl_args.tname
else:
    timename = 'time'
# Name of tempeprature field in file
if cl_args.tempname:
    tname =  cl_args.tempname
else:
    tname = 'T'   
# output filename
if cl_args.outfile:
    outfilename = cl_args.outfile
else:
    outfilename = '~/test_cid.nc'        

if cl_args.cyclic:
    cyclic = True
    print( "SLP data cyclic in x-direction" )
else:
    cyclic = False
    
dp = 2.                             # Pressure interval in hPa
min_p = 900.                        # Minimum pressure contour in hPa
max_p = 1100.                       # Maximum pressure contour in hPa
min_clength = 200.                  # Minimal contour length in km
max_clength = 10000.                 # Maximal contour length in km
cluster_radius = 2000.              # Radius for clustering extrema in km
min_carea = 100. * 1000.            # Minimum contour area in qkm
min_p_filter = 900.                 # Min filter for extrema in hPa
max_p_filter = 1200.                # Max filter for extrema in hPa
topo_filter  = 1500.                # Topopgrahy filter in m


import time as cpu_time
start_time = cpu_time.time()


# Time level. Default is 0 
t0 = cl_args.timestep

# Subdomain
if cl_args.minlon:
    subdomain = True

    minlat = cl_args.minlat
    maxlat = cl_args.maxlat
    minlon = cl_args.minlon
    maxlon = cl_args.maxlon
else:
    subdomain = False


# Plot options
if cl_args.plotname:
    plot = True
    plot_name = cl_args.plotname
    if cl_args.plotdetail: 
        plot_all = True
        print( 'Detailed plot to ' + plot_name )
    else:
        plot_all = False
        print( 'Plot to ' + plot_name )
else:
    plot = False
    plot_all = False

#--------------------------------------------------------
#------------- Utility functions and objects ------------
#--------------------------------------------------------

# Define empty objects for clustering minima and associated contours
class minima(object):
    pass
    
class contour(object):
    pass

# Define maskable list object for sorting contour line arrays and points
from itertools import compress, combinations
class MaskableList(list):

    def __getitem__(self, index):

        try: return super(MaskableList, self).__getitem__(index)
        except TypeError: return MaskableList(compress(self, index))

# Find nearest value in array and get its index
def find_nearest_ind(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

# Copy all attributes associated with a numpy array to another array
class CopyArrayDict(np.ndarray):

    def __new__(cls, input_array, dicts):
     
        obj = np.asarray(input_array).view(cls)

        for key in dicts.keys():
            setattr(obj, key, dicts[key] ) 

        return obj

    def __array_finalize__(self, obj):
 
        if obj is None: return

# Print information on array data
def min_max_arr( arr):

    nan = np.isnan( arr ).any()
    inf =  np.isinf( arr )
    if hasattr( arr, 'name' ):
        name = arr.name
    else:
        name = ""

    if hasattr( arr, 'standard_name'):
        name = arr.standard_name
    else:
        name = ""

    print( name + " Max: " + str( np.amax(arr) )  + " Min: " + str( np.amin(arr) ) + " Mean: "+str( np.mean(arr) ) +" NaN? " + str(nan.any()) + " Inf? " + str(inf.any()) + " Dtype: " + str(arr.dtype) )


# Pressure reduction to sea-level using barometeric formula
g = 9.80665 # Gravitational acceleration in m s^-2
Ra = 287.04781 # Specific gas constant of dry air in J kg^-1 K^-1
def baro_pres( z, P, T, z0=0 ):
    # Variant with mean temperature
    # Tn = T + 0.00325 * z
    # Variant with constant temperature 
    # h  = z - z0
    # P0 = P * np.exp(  g * h / ( Ra * Tn ) )
    # Variant with linear temperature profile
    lapse = -0.0065
    exp = g / ( Ra * lapse )
    P0 = P * ( T / ( T + lapse * ( z0 - z ) ) )**exp

    return P0

# Plot routine for contours and extrema
def plot_contour( pp, x, y, z, title, lev=None, list_contours=[], points=None, \
                      fill=False, format='%d', contourtyp='cord'):
    def get_label(var):

        name = getattr(var, 'name', str(var))

        if hasattr(var,'units'):
            units = " [" + getattr(var,'units') +"]"
        else:
            units = ""
            
        if hasattr(var,'long_name'):
            long_name = getattr(var, 'long_name', str(var))
        else:
            long_name = name

        return name, long_name, units

    x_meta = get_label( x )
    y_meta = get_label( y )    
    z_meta = get_label( z )

    fig, ax = plt.subplots()
    if fill:
        cm = ax.contourf( y, x, z, levels=lev, )
    else:
        cm = ax.contour( y, x, z, colors='black', levels=lev, linewidths=.5 )
        ax.clabel(cm, fontsize=8, fmt=format)
    #q = ax.quiver(y, x, u, v, units='width')

    ax.set_xlabel( x_meta[1] + " " + x_meta[2] )
    ax.set_ylabel( y_meta[1] + " " + y_meta[2] )
    ax.set_title( title )
   


    for i in range(len(list_contours)):
        contour= list_contours[i]
        if contourtyp == 'latlon':
            xp = contour.latlon[:, 1]
            yp = contour.latlon[:, 0]
        elif contourtyp == 'xy_sinu':
            xp = contour.xy_sinu[:, 1]
            yp = contour.xy_sinu[:, 0]
        else:
            xp = contour.contour[:, 1]
            yp = contour.contour[:, 0]

        ax.plot(xp, yp, linewidth=.5, color='red')

    if points is not None:
        for i in range(len(points)):
            point = points[i]
            if contourtyp == 'latlon':
                xp = point.latlon[1]
                yp = point.latlon[0]
            elif contourtyp == 'xy_sinu':
                xp = point.xy_sinu[1]
                yp = point.xy_sinu[0]
            else:
                xp = point.cord[1]
                yp = point.cord[0]
            val = point.value

            if hasattr( point, 'color' ):
                color = getattr(point, 'color')
            else:
                color = 'cyan'

            ax.plot( xp, yp, 'o', color=color, markersize=3)
            txt = val
            ax.annotate(format%txt, (xp, yp),size=10, color=color)

    plt.tight_layout()
    pp.savefig()


#--------------------------------------------------------
#------------- Support functions ------------------------
#--------------------------------------------------------

from math import sin, cos, sqrt, atan2, radians, degrees
lon0 = 0.             # Central meridian
Re = 6371.0088        # Approximate radius of earth in km

# Ray tracing. Returns true if point is inside polygon
def ray_tracing_method( x, y, poly):

    n = len(poly)
    inside = False

    p1x, p1y = poly[0]
    for i in range(n+1):
        p2x, p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = ( y - p1y )*( p2x - p1x )/( p2y - p1y ) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


# Find collinear point pairs in contour and mark the nearest grid points
from collections import Counter
def collinear_pairs( contour, x, y  ):

    obj = contour
    points = obj.contour
    N = len( points )

    xlist = { }
    ylist = { }

    comb = combinations(points, 2)

    for p1, p2 in comb:

        if p1[1] == p2[1]:
            i1 = find_nearest_ind( x[:,0], p1[0] )
            i2 = find_nearest_ind( x[:,0], p2[0] )
            j  = find_nearest_ind( y[0,:], p1[1] )
            i = ( min(i1,i2),max(i1,i2) )
            if j not in xlist:
                xlist[j] = [ i ]
            else:
                xlist[j].append( i )

        if p1[0] == p2[0]:
            i = find_nearest_ind( x[:,0], p1[0] )
            j1  = find_nearest_ind( y[0,:], p1[1] )
            j2  = find_nearest_ind( y[0,:], p2[1] )
            j = ( min(j1,j2),max(j1,j2) )
            if i not in ylist:
                ylist[i] = [ j ]
            else:
                ylist[i].append( j ) 

    obj.collinear_xpairs = xlist
    obj.collinear_ypairs = ylist
   
# Sinusoidal equal-area map projection  
# Input of lat and lon in radians
def sinusodial_map( lat, lon ):

    shape = np.shape( lon )
    x = np.zeros( shape )
    y = np.zeros( shape )

    for i in np.ndindex(shape):
        
        y[i] = Re * ( lon[i] - lon0 ) * cos( lat[i] )
        x[i] = Re * lat[i]

    return x, y # in km 

# Interpolate latitude and longitude along the trajectory
def interpolate_coordinates( contour, x, y, x_int, y_int ):

    n = len(contour)
    xy = np.zeros( (n,2) )

    for i in range(0,n):
        xy[i][0] = np.interp( contour[i][0]  , x[:,0], x_int[:,0] )
        xy[i][1] = np.interp( contour[i][1]  , y[0,:], y_int[0,:] )

    return xy #x_vert, y_vert

# Get contour coordinates in sinusodial map projection 
def sinusodial_contours( contour_latlon ):
    
    n = len(contour_latlon)
    xy = np.zeros( (n,2) )

    for i in range(0,n):
        xy[i][0] = Re * radians(contour_latlon[i][0])
        xy[i][1] = Re * ( radians(contour_latlon[i][1]) - lon0 ) * cos( radians(contour_latlon[i][0]) )

    return xy

# Calculate distance between two points on a sphere using Haversine formula
def calculate_distance( lat1, lon1, lat2, lon2, units=''):
    
    # Convert to radians if necessary 
    if not 'radians' in units:
        lat1 = radians(lat1)
        lon1 = radians(lon1)
        lat2 = radians(lat2)
        lon2 = radians(lon2)
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Haversine formula
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = atan2(sqrt(a), sqrt(1 - a))

    distance = 2 * Re * c

    return distance # in km

# Calculate distance in a cartesian grid
def calculate_distance_cartesian( x1, y1, x2, y2 ):

    distance = np.sqrt( (x1 - x2)**2. + (y1 - y2)**2. )

    return distance # in km

# Calculate contour length in km using great-circle-distance
# Lat, lon are coordinate vectors in degree or radians
def contour_length( contour ):
    
    n = len(contour)
    dis = 0

    for i in range(1,n):
        lat1 = contour[i-1][0]
        lon1 = contour[i-1][1]
        lat2 = contour[i][0]
        lon2 = contour[i][1]

        dis = dis + calculate_distance( lat1, lon1, lat2, lon2) 
    
    return dis # in km

# Calculate area enclosed by contour using Green's theorem
# with coordinats in sinusodial map projection
def contour_area( contour ):

    x_vert, y_vert =  np.hsplit( contour, 2 )
    x_vert  = x_vert.flatten()
    y_vert  = y_vert.flatten()

    area = 0.5 * np.sum( y_vert[:-1] * np.diff(x_vert) - x_vert[:-1] * np.diff(y_vert) )
    area = np.abs( area )

    return area # in qkm

# Put a box around a contour and return true if point is inside
def contour_boxing( contour, xp, yp ):

    L = len(contour)
                
    x_max = contour[0][0]
    y_max = contour[0][1]
    x_min = contour[0][0]
    y_min = contour[0][1]
    
    for k in range(1,L):
        x_max = max(x_max, contour[k][0])
        x_min = min(x_min, contour[k][0])
        y_max = max(y_max, contour[k][1])
        y_min = min(y_min, contour[k][1])
            
    if xp >= x_min and xp <= x_max and yp >= y_min and yp <= y_max:
        return True
    else:
        return False
    

#--------------------------------------------------------
#------------- Find contours and extrema  ---------------
#--------------------------------------------------------


# Find local extrema (currently only minima) in array 
def find_extrema( arr, x, y, x_sinu, y_sinu, order = 1 ):
    
    # Only consider full blocks 
    # Ignore elements close to domain boundaries
    block_min_size = ( 2 * order + 1  )**2

    # List of extrema indicies
    extrema = MaskableList( [] )
   
    # Loop through every element of array
    shape = arr.shape 
    for index in np.ndindex(shape):

        i_s = index[0] - order
        i_e = index[0] + order + 1
        j_s = index[1] - order
        j_e = index[1] + order + 1

        val = arr[index]

        block = arr[ i_s:i_e, j_s:j_e  ].copy()

        if block.size >= block_min_size:
            # Adjust value of element so that it passes the check 
            # since we are only interested in its neighbours
            block[(order, order)] = block[(order, order)] + 0.1
            check = val <= block
            if check.all():
                ext = minima()
                ext.type = 'local minima'                     # Type of extrema
                ext.cord = index                              # Coordinates in gridpoints
                ext.latlon  = ( x[index], y[index] )          # Coordinates in latlon
                ext.xy_sinu = ( x_sinu[index], y_sinu[index] )# Coordinates in sinusodial proj 
                ext.value = arr[ index ]                      # Pressure value

                extrema.append( ext )
    
    return extrema # a list of extrema with coordinates and value as attributes

# Find contours in array for a list of value
# x, y are coordinate indicies; lat, lon corrosponding coordinate values
def find_contours( arr, level, x, y, lat, lon, x_sinu, y_sinu ):
    
    list_contours =  MaskableList( [] )
    for val in level:

        contours = measure.find_contours(arr, val)
        for i, ele in enumerate(contours):
            obj = contour()
            obj.contour = ele # Contour coordinate array
            obj.value   = val   # Contour value
            # Contour in latlon
            obj.latlon  = interpolate_coordinates( obj.contour, x, y, lat, lon )
            # Contour in sinusodial projection 
            obj.xy_sinu = sinusodial_contours( obj.latlon )
            obj.length  = contour_length( obj.latlon ) # Contour length in km
            obj.cyclone_extrema = []
            list_contours.append( obj )

    return list_contours

# Alternative implementation finding points inside contours based on scipy.ndimage
def find_all_points_inside_ndimage( list_contours, list_extrema, value_array, pvar):

    for i, extrema in enumerate( list_extrema ):
        test = copy.copy(pvar)
        test[test>extrema.largest_enclosing_contour_val] = 0
        label, num_obj = scipy.ndimage.label(test)
        obj_w_extremum = label[extrema.cord]
        value_array[label==obj_w_extremum] = value_array[label==obj_w_extremum]+1
    return value_array

# Check if extrema is inside closed contour
def find_points_inside_contour( list_contours, list_points ):

    N = len(list_contours)
    M = len(list_points)

    for i in range(N):

        obj = list_contours[i]
        contour = obj.contour
        value = obj.value
        
        obj.extrema = MaskableList( [] )
        
        # Check for each extrema in list_points and each contour
        for j in range(M):

            point = list_points[j]
            extrema = point.value
            cord = point.cord

            if not hasattr( point, 'enclosing_contours'):
                point.enclosing_contours =  MaskableList( [] )

            # Make sure contour value is larger than minima
            if value > extrema:

                # Currently disabled since both methods are of similiar performance
                # if contour_boxing( contour, cord[0], cord[1] ): 
                
                inside = ray_tracing_method( cord[0], cord[1], contour) 
                # If inside attribute extrema to contour and vice versa
                if inside:
                    obj.extrema.append( point )
                    point.enclosing_contours.append( obj )

# Find all points inside contour
def find_all_points_inside_contours( list_contours, x, y, value_array ):

    shape = np.shape( x )

    for i in np.ndindex(shape):
            
        for j, contour in enumerate( list_contours ):
            
            if value_array[i] == 0:
                
                if contour_boxing( contour.contour,  x[i], y[i] ): 
                
                    # Verify that point really is inside contour
                    inside = ray_tracing_method( x[i], y[i], contour.contour) 
                    if inside:
                        value_array[i] = 1
                       

    return value_array

# Find all collinear pairs
def find_all_collinear_pairs( list_contours, x, y ):

    N = len(list_contours)
    
    for i in range(N):

        obj = list_contours[i]
        collinear_pairs( obj, x, y )

# Find all points inside contour by slicing between collinear pairs
def find_all_points_inside_contours_slicing( list_contours, arr ):
    
    N = len(list_contours)

    arr_x = np.zeros( arr.shape )
    arr_y = np.zeros( arr.shape )
    
    for i in range(N):

         obj = list_contours[i]
   
         plist =  obj.collinear_xpairs
         for key, p in plist.items():
             for i in range(len(p)):
                 i1 = p[i][0]
                 i2 = p[i][1]
                 arr_x[ i1:i2, key ] =  1
         plist =  obj.collinear_ypairs
         for key, p in plist.items():
             for i in range(len(p)):
                 i1 = p[i][0]
                 i2 = p[i][1]
                 arr_y[ key, i1:i2 ] =  1
    arr = arr_x

    arr = np.where( arr_y == 1, arr, 0 ) 
    return arr

# Characterize minima and their neighbours
# Cluster and chraracterize minima depending on distance, depth 
# and if they share an enclosing contour
# Also find largest enclosing contour for each minima
def cluster_extrema( list_extrema, cluster_radius ):

    N = len( list_extrema ) 

    for i in range(N):

        extrema = list_extrema[i]
        pos = extrema.latlon
        value = extrema.value
        contours = extrema.enclosing_contours
        extrema.contour_flag = [True] * len( extrema.enclosing_contours )

        extrema.list_neighbours = []
        # Check if other extrema are in cluster radius
        for j in range(N):

            if i != j:
               
                another_extrema = list_extrema[j]
                pos2 = another_extrema.latlon
                distance = calculate_distance( pos[0], pos[1], pos2[0], pos2[1] ) 

                # Check if extrema share a contour
                contour_set1 = set( contours )
                contour_set2 = set( another_extrema.enclosing_contours )
                share = contour_set1.intersection( contour_set2 )

                # Neighbours in same cyclone 
                if len( share ) > 0 and distance <= cluster_radius:
                    extrema.list_neighbours.append( another_extrema )

                # Neighbours in different cyclones. Remove sharing contours
                if len( share ) > 0 and distance > cluster_radius:
                    contour_set1.difference_update( share )
                    contour_set2.difference_update( share )

                   
                    for i, contour in enumerate( extrema.enclosing_contours  ):
                        if contour in share:
                            extrema.contour_flag[i] = False
                        
                    # extrema.enclosing_contours =  MaskableList( [] )
                    # for i, contour in enumerate( contour_set1 ):
                    #     extrema.enclosing_contours.append( contour )

                    # another_extrema.enclosing_contours =  MaskableList( [] )
                    # for i, contour in enumerate( contour_set2 ):
                    #     another_extrema.enclosing_contours.append( contour )
                        

   
        # Characterize extrema in regard to its neighbours
        M = len(extrema.list_neighbours)
        if M > 0:
            neighbour_values = []
            for j in range(M):
                another_extrema = extrema.list_neighbours[j]
                neighbour_values.append( another_extrema.value )
        
            test = value < neighbour_values
            
            if test.all():
                extrema.type  = 'deepest minimum'
                extrema.color = 'red'
            else:
                extrema.type  = 'secondary minima'
                extrema.color = 'blue'
        else:
            extrema.type = 'free minima'
            extrema.color = 'green'

    # Remove shared contours of different cyclones and find largest contour
    for i in range(N):

        extrema  = list_extrema[i]
        #contours = extrema.
        flag     = extrema.contour_flag
        contours = extrema.enclosing_contours
        contours = contours[ flag ]
        if not isinstance(contours, list):   
            contours = [ contours ]
            flag = [ flag ]
        

        # Find largest enclosing contour
        if len(contours) > 0: # andextrema.type == 'deepest minimum' or extrema.type == 'free minima' \
            #print( extrema.type + "  " + str(extrema.value) + ' with ' + str(len(contours) ) + ' contours' \
             #          + ' out of prior ' + str(len(flag)) )
          
            values = []
            for i, contour in enumerate( contours ):
                values.append( contour.value )
           
            if len(values) > 1:
                max_value = max( values )
                max_index = [i for i, j in enumerate(values) if j == max_value]

                largest_contour = contours[max_index[0]]
                extrema.largest_enclosing_contour = largest_contour
                # additional property needed for cyclone point identification with scipy.ndimage
                extrema.largest_enclosing_contour_val = max_value
                largest_contour.cyclone_extrema.append( extrema )
 
#--------------------------------------------------------
#------------- Test and filter functions ----------------
#--------------------------------------------------------
   

# Test if contour is closed
def test_closed( list_contours ):
    
    N = len(list_contours)
    mask = [False] * N
    for i in range(N):

        obj = list_contours[i]
        start = obj.contour[0,:]
        end = obj.contour[-1,:]

        if start[0] == end[0] and start[1] == end[1]:
            mask[i] = True

    list_contours = list_contours[mask]

    return list_contours    

# Test if contour encloses an extrema
def test_point_inside( list_contours, list_points ):

    N = len(list_contours)
    M = len(list_points)
    mask = [False] * N
    for i in range(N):

        obj = list_contours[i]
        contour = obj.contour
        value = obj.value
        
        if hasattr( obj, 'extrema'):
            if len(obj.extrema) > 0:
                mask[i] = True
        else:       
            # Check for each extrema in list_points and each contour
            for j in range(M):
            
                point = list_points[j]
                extrema = point.value
                cord = point.cord

                # Make sure contour value is larger/smaller than extrema
                if value > extrema:
                    # If a contour encloses multiple extrema only count once 
                    if not mask[i]:
                        mask[i] = ray_tracing_method( cord[0], cord[1], contour) 
   
    list_contours = list_contours[mask]

    return list_contours

# Test if contour is to long or to short
def test_contour_length( list_contours, minlen, maxlen ):

    N = len(list_contours)
    mask = [True] * N
    for i in range(N):

        contour = list_contours[i]    
        length = contour.length 

        if length < minlen:
            mask[i] = False
        if length > maxlen:
            mask[i] = False

    list_contours = list_contours[mask]   

    return list_contours

# Test if contour area is too small
def test_contour_area( list_contours, x, y, x_sinu, y_sinu, minarea ):

    N = len(list_contours)
    mask = [True] * N
    for i in range(N):

        contour = list_contours[i]
        if hasattr(contour, 'area'):
            area = contour.area
        else:
            area = contour_area( contour.xy_sinu )

        if area < minarea:
            mask[i] = False

    list_contours = list_contours[mask] 
    
    return list_contours

# Test if extrema is locatated to high in topography 
def test_topography( list_extrema, topo, altitude_threshold = topo_filter ):

    N = len(list_extrema)
    mask = [True] * N
    for i in range(N):
        extrema = list_extrema[i]
        index = extrema.cord

        altitude = topo[index]
       
        if altitude >= altitude_threshold:
            mask[i] = False
 
    list_extrema = list_extrema[mask]

    return list_extrema

# Test if value of extrema is unreasonable low or high
def test_extrema_depth( list_extrema, min_threshold = min_p_filter, max_threshold = max_p_filter ):

    N = len(list_extrema)
    mask = [True] * N
    for i in range(N):
        extrema = list_extrema[i]
        value = extrema.value

        if value <= min_threshold or value >= max_threshold:
            mask[i] = False
   
    list_extrema = list_extrema[mask]

    return list_extrema       

# Get a list of largest enclosing contours
def filter_largest_contour( list_extrema ):

    #list_extrema_new  = MaskableList( [] )
    list_contours     = MaskableList( [] )
    for i, point in enumerate( list_extrema ):
        if hasattr(point, 'largest_enclosing_contour'):
            contour = getattr(point, 'largest_enclosing_contour')
            list_contours.append( contour )
            #list_extrema_new.append( point ) 

    return list_contours

# Get a list of extrema with enclosing contours
def filter_extrema_w_contour( list_contour ):

    new_list_extrema = MaskableList( [] )
    for i, contour in enumerate( list_contour ):
        if hasattr(contour, 'cyclone_extrema'):
            extrema = getattr(contour, 'cyclone_extrema')
            for j, point in enumerate( extrema ):
                new_list_extrema.append( point )
    # Remove duplicates
    new_list_extrema = list(  dict.fromkeys( new_list_extrema ) )
    return new_list_extrema

# Filter out all extrema without any enclosing contours
def filter_extrema_no_contour( list_extrema  ):
    
    N = len(list_extrema)
    mask = [True] * N
    for i in range(N):
        extrema = list_extrema[i]
        M = len( extrema.enclosing_contours)
        if M == 0:
            mask[i] = False
        
    list_extrema = list_extrema[mask]

    return list_extrema   

#--------------------------------------------------------------------     
        

# Input file
rootgrp = Dataset(filename, "r", format="NETCDF4")
topogrp = Dataset(topofilename, "r", format="NETCDF4")

lati = rootgrp.variables[latname]
loni = rootgrp.variables[lonname]
vari = rootgrp.variables[pname]
if toponame in topogrp.variables.keys():
    topi = topogrp.variables[toponame]
    topo_check = True
else:
    topo_check = False
    print('Topography check disabled')

time = rootgrp.variables[timename]
time = time[t0]

# Output file
outgrp = Dataset(outfilename, "w", format="NETCDF4")
outgrp.setncatts( rootgrp.__dict__ )
for name, dimension in rootgrp.dimensions.items():
    outgrp.createDimension( name, (len(dimension) if not dimension.isunlimited() else None ))

#outgrp = rootgrp

# Subdomain 
if subdomain:
    i0 = find_nearest_ind(lati,minlat)
    iend = find_nearest_ind(lati,maxlat)
    j0 = find_nearest_ind(loni,minlon)
    jend = find_nearest_ind(loni,maxlon)
else:
    i0 = 0
    iend = rootgrp.dimensions[latname].size #- 1
    j0 = 0
    jend = rootgrp.dimensions[lonname].size #- 1

if cyclic:
    jend = jend*2

# Model level
# Remember for ICON level 0 is TOA, last is surface
if not two_dim:
    if cl_args.modelevel: 
        k0 = cl_args.modelevel
    else:
        k0 = rootgrp.dimensions[heightname].size - 1
    nz   = rootgrp.dimensions[heightname].size
    
nlat = iend - i0
nlon = jend - j0


# Output variable
# Read cyclone index variable if in file, otherwise create it
if 'label' in  outgrp.variables:
    cyclone_index = outgrp['label']
else:
    if not two_dim:
        cyclone_index = outgrp.createVariable( 'label', float, (timename, heightname, latname, lonname,) )
    else:
        cyclone_index = outgrp.createVariable( 'label', float, (timename, latname, lonname,) )
cyclone_index.long_name = "cyclone index"
cyclone_index.standard_name = "label"
cyclone_index.units = ""

lon_out = outgrp.createVariable( 'lon', float, ("longitude",) )
lat_out = outgrp.createVariable( 'lat', float, ("latitude",) )

lon_out[:] = loni[:]
lat_out[:] = lati[:]

if not two_dim:
    for k in range(nz):
        cyclone_index[t0,k,:,:] = 0
else:
    cyclone_index[t0,:,:] = 0

# future time loop begins here
    
# Only surface pressure is a 3D field, every other is 4D [time,level,lat,lon]
if pname == 'P':
    var = CopyArrayDict( vari[t0,k0,i0:iend,j0:jend], vari.__dict__ )
else:
    var = CopyArrayDict( vari[t0,i0:iend,j0:jend], vari.__dict__ )

if var.units.lower() == 'hpa':
    pass
else:
    var = var / 100.
    var.units = 'hPa'

# does not yet work with topography filter!
if cyclic:
    loni_ext = np.zeros(2*loni.shape[0])
    vari_ext = np.zeros((vari.shape[1],vari.shape[2]*2))
    loni_ext[0:np.int(loni.shape[0]/2)] = loni[np.int(loni.shape[0]/2):]-360
    loni_ext[np.int(loni.shape[0]/2):np.int(3*loni.shape[0]/2)] = loni[:]
    loni_ext[np.int(3*loni.shape[0]/2):] = loni[:np.int(loni.shape[0]/2)]+360
    vari_ext[:,0:np.int(loni.shape[0]/2)] = np.squeeze(vari[:,:,np.int(loni.shape[0]/2):])
    vari_ext[:,np.int(loni.shape[0]/2):np.int(3*loni.shape[0]/2)] = np.squeeze(vari[:,:,:])
    vari_ext[:,np.int(3*loni.shape[0]/2):] = np.squeeze(vari[:,:,:np.int(loni.shape[0]/2)])
    var = CopyArrayDict(vari_ext/100., vari.__dict__)

if topo_check:
    top = CopyArrayDict( topi[i0:iend,j0:jend],   topi.__dict__ )

# Pressure reduction to sea-level
if p_redu:
    tempi = rootgrp.variables[tname]
    temp = tempi[t0,k0,i0:iend,j0:jend]
    var = baro_pres( top, var, temp )

if cyclic:
    lon = CopyArrayDict(loni_ext, loni.__dict__)
    lat, lon = np.meshgrid(  lati[i0:iend], loni[j0:jend],  indexing='ij' )
else:
    lat, lon = np.meshgrid(  lati[i0:iend], lon[j0:jend],  indexing='ij' )


lat = CopyArrayDict( lat                  ,   lati.__dict__ ) 
lon = CopyArrayDict( lon                  ,   loni.__dict__ ) 


# Convert degrees into radians
latr = np.zeros( np.shape(lat) )
latr = CopyArrayDict( latr                 ,   lati.__dict__ ) 
for i in np.ndindex(lat.shape):
    latr[i] = radians(lat[i])
lonr = np.zeros( np.shape(lon) )
lonr = CopyArrayDict( lonr                  ,   loni.__dict__ ) 
for i in np.ndindex(lon.shape):
    lonr[i] = radians(lon[i])

latr.units = 'radians'
lonr.units = 'radians'


# Order of looking for minima dependet on grid size. Search within 5°
dlat = abs(lati[0] - lati[1])
dlon = abs(loni[0] - loni[1])
dd = (dlat + dlon) / 2.
order_ext = int(5. / dd)


# Grid point arrays used for all contour analysis
x, y = np.meshgrid( np.arange( 0., nlat, 1. ), np.arange( 0., nlon, 1. ),  indexing='ij'   )
y = CopyArrayDict( y,           lon.__dict__ )
y.units = 'grid points'
x = CopyArrayDict( x,           lat.__dict__ )
x.units = 'grid points'

# Sinusodial map projection used for calculating contour area
x_sinu, y_sinu = sinusodial_map( latr, lonr )
x_sinu = CopyArrayDict( x_sinu,           lat.__dict__ )
x_sinu.units = 'km' 
y_sinu = CopyArrayDict( y_sinu,           lon.__dict__ )
y_sinu.units = 'km'



                       
# Pressure intervals
plevel = np.arange(min_p,max_p,dp) 

# Calculate contours
list_pcontours = find_contours( var, plevel, x, y, lat, lon, x_sinu, y_sinu)

# Find local minima 
list_extrema = find_extrema( var, lat, lon, x_sinu, y_sinu, order = order_ext )


if plot:
    pp = PdfPages( plot_name )

if plot_all:
    plot_contour(pp,x,y,var,"Surface pressure minima order "+str(order_ext),lev=plevel,points=list_extrema)

# Test if extrema are to high in topography
if topo_check:
    list_extrema = test_topography(list_extrema, top)

    if plot_all:
        plot_contour(pp,x,y,var,"Surface pressure minima after topography filter",lev=plevel,points=list_extrema)


# Test if extrema are to extreme
list_extrema = test_extrema_depth( list_extrema)

if plot_all:
    plot_contour(pp,x,y,var,"Surface pressure minima after threshold filter",lev=plevel,points=list_extrema)


# Test if contours are closed
list_pcontours = test_closed( list_pcontours )

if plot_all:
    plot_contour(pp,x,y,var,"Contour is closed?",lev=plevel,list_contours=list_pcontours,points=list_extrema)


# Test if contours are too long or to short
list_pcontours = test_contour_length( list_pcontours, min_clength, max_clength)

if plot_all:
    plot_contour(pp,x,y,var,"Contour too long or short?", lev=plevel, list_contours=list_pcontours,points=list_extrema)

# Test if area enclosed by area is too small
list_pcontours = test_contour_area( list_pcontours, x, y, x_sinu, y_sinu, min_carea )

if plot_all:
    plot_contour(pp,x,y,var,"Contour area too small?", lev=plevel,list_contours=list_pcontours,points=list_extrema)


# Associate minima with enclosing contours
find_points_inside_contour( list_pcontours, list_extrema )
list_extrema = filter_extrema_no_contour( list_extrema )

# Test if contour encloses at least one minima
list_pcontours = test_point_inside( list_pcontours, list_extrema )

if plot_all:
    plot_contour(pp,x,y,var,"Contour encloses minima?", lev=plevel,list_contours=list_pcontours,points=list_extrema)

# Characterize minima and associate them with largest enclosing contour
cluster_extrema( list_extrema, cluster_radius ) 

list_pcontours = filter_largest_contour( list_extrema )
list_extrema = filter_extrema_w_contour( list_pcontours  )

if plot:
    plot_contour(pp,x,y,var,"Cyclone clustering at " + str(time) , lev=plevel,list_contours=list_pcontours,points=list_extrema)


print( 'Number of cyclones found ' + str(len( list_pcontours )) + ' with ' + str(len( list_extrema )) + ' extremas' )


# Calculate cyclone index
shape = np.shape( cyclone_index )
if not two_dim:
    dim_cvar = (shape[2], shape[3], )
else:
    dim_cvar = (shape[1], shape[2], )
cvar = np.zeros(  dim_cvar, dtype=cyclone_index.dtype )

# Find collinear points in the contours
find_all_collinear_pairs( list_pcontours, x, y )

cvar = find_all_points_inside_contours_slicing( list_pcontours, cvar)

#find_all_points_inside_ndimage( list_pcontours, list_extrema, cvar, var )

print( "Index max: ", str(cvar.max()), " min: ", str(cvar.min()) )

if not two_dim:
    for k in range(nz):
        if cyclic:
            cyclone_index[t0,k,:,:] = cvar[:,np.int(loni.shape[0]/2):np.int(3*loni.shape[0]/2)]
        else:
            cyclone_index[t0,k,:,:] = cvar
else:
    if cyclic:
        cyclone_index[t0,k,:] = cvar[:,np.int(loni.shape[0]/2):np.int(3*loni.shape[0]/2)]
    else:
        cyclone_index[t0,:,:] = cvar

    
# Close writing 
outgrp.close()


if plot:
    plot_contour(pp,x,y,cvar,"Cyclone index", lev=[0.5,1.5], list_contours=list_pcontours,points=list_extrema, fill=True)


if plot:
    pp.close()

print("Finished after:")
print("--- %s seconds ---" % (cpu_time.time() - start_time))
