# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Helper functions for calculating 2D and 3D bounding box IoU.

Collected and written by Charles R. Qi
Last modified: Jul 2019
"""
from __future__ import print_function

import numpy as np
from scipy.spatial import ConvexHull

import copy
import open3d as o3d

def polygon_clip(subjectPolygon, clipPolygon):
   """ Clip a polygon with another polygon.

   Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

   Args:
     subjectPolygon: a list of (x,y) 2d points, any polygon.
     clipPolygon: a list of (x,y) 2d points, has to be *convex*
   Note:
     **points have to be counter-clockwise ordered**

   Return:
     a list of (x,y) vertex point for the intersection polygon.
   """
   def inside(p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
 
   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0] 
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
 
   outputList = subjectPolygon
   cp1 = clipPolygon[-1]
 
   for clipVertex in clipPolygon:
      cp2 = clipVertex
      inputList = outputList
      outputList = []
      s = inputList[-1]
 
      for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
            outputList.append(e)
         elif inside(s):
            outputList.append(computeIntersection())
         s = e
      cp1 = cp2
      if len(outputList) == 0:
          return None
   return(outputList)

def poly_area(x,y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0  

def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c

def is_clockwise(p):
    x = p[:,0]
    y = p[:,1]
    return np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)) > 0

def box3d_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

    todo (rqi): add more description on corner points' orders.
    '''
    try:

        # corner points are in counter clockwise order
        rect1 = [(corners1[i,0], corners1[i,2]) for i in range(3,-1,-1)]
        rect2 = [(corners2[i,0], corners2[i,2]) for i in range(3,-1,-1)] 
        area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
        area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])

        inter, inter_area = convex_hull_intersection(rect1, rect2)
        #print('convex_hull_intersection completed')

    except:

        inter, inter_area = 0,0

        print('convex_hull_intersection failed')
        # show boxes for debugging
        #verts1=np.asarray(corners1)
        #bbox1=o3d.geometry.OrientedBoundingBox().create_from_points(o3d.utility.Vector3dVector(verts1))
        #bbox1.color=[1,.1,.1]
        #verts2=np.asarray(corners2)
        #bbox2=o3d.geometry.OrientedBoundingBox().create_from_points(o3d.utility.Vector3dVector(verts2))
        #bbox2.color=[.1,1,.1]
        #o3d.visualization.draw_geometries([bbox1, bbox2])

    iou_2d = inter_area/(area1+area2-inter_area)
    ymax = min(corners1[0,1], corners2[0,1])
    ymin = max(corners1[4,1], corners2[4,1])
    inter_vol = inter_area * max(0.0, ymax-ymin)
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two 2D bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:

        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def box2d_iou(box1, box2):
    ''' Compute 2D bounding box IoU.

    Input:
        box1: tuple of (xmin,ymin,xmax,ymax)
        box2: tuple of (xmin,ymin,xmax,ymax)
    Output:
        iou: 2D IoU scalar
    '''
    return get_iou({'x1':box1[0], 'y1':box1[1], 'x2':box1[2], 'y2':box1[3]}, \
        {'x1':box2[0], 'y1':box2[1], 'x2':box2[2], 'y2':box2[3]})

# -----------------------------------------------------------
# Convert from box parameters to 
# -----------------------------------------------------------

def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                    [0,  c, -s],
                    [0,  s,  c]])

def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                    [0,  1,  0],
                    [-s, 0,  c]])

def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                    [s,  c,  0],
                    [0,  0,  1]])

def roty_batch(t):
    """Rotation about the y-axis.
    t: (x1,x2,...xn)
    return: (x1,x2,...,xn,3,3)
    """
    input_shape = t.shape
    output = np.zeros(tuple(list(input_shape)+[3,3]))
    c = np.cos(t)
    s = np.sin(t)
    output[...,0,0] = c
    output[...,0,2] = s
    output[...,1,1] = 1
    output[...,2,0] = -s
    output[...,2,2] = c
    return output


def get_3d_box(box_size, heading_angle, center):
    ''' box_size is array(l,w,h), heading_angle is radius clockwise from pos x axis, center is xyz of box center
        output (8,3) array for 3D box cornders
        Similar to utils/compute_orientation_3d
    '''
    #print('heading_angle:', heading_angle)

    #if len(heading_angle)==1:       # original method
    if True:                         # force original for debugging
        angle=heading_angle[2]
        #angle=heading_angle
        Rx = rotx(heading_angle[0])
        Ry = roty(heading_angle[2])  # previous method (switches z to y, then uses roty as z rotation)
        Rz = rotz(heading_angle[1])

        l,w,h = box_size
        x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
        y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2];
        z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
        corners_3d = np.vstack([x_corners,y_corners,z_corners])

        #bbox0=o3d.geometry.OrientedBoundingBox().create_from_points(o3d.utility.Vector3dVector(np.transpose(corners_3d)))
        #bbox0.color=[1,.6,.6]

        #corners_3d = np.dot(Rx, np.vstack(corners_3d))
        corners_3d = np.dot(Ry, np.vstack(corners_3d))
        #corners_3d = np.dot(Rz, np.vstack(corners_3d))

        #bbox1=o3d.geometry.OrientedBoundingBox().create_from_points(o3d.utility.Vector3dVector(np.transpose(corners_3d)))
        #bbox1.color=[1,.4,.4]

        corners_3d[0,:] = corners_3d[0,:] + center[0];
        corners_3d[1,:] = corners_3d[1,:] + center[1];
        corners_3d[2,:] = corners_3d[2,:] + center[2];

        #bbox2=o3d.geometry.OrientedBoundingBox().create_from_points(o3d.utility.Vector3dVector(np.transpose(corners_3d)))
        #bbox2.color=[1,.1,.1]

        corners_3d = np.transpose(corners_3d)
        
    # if len(heading_angle)==3: # three axis method added by th    

    #     # use standard Z up right hand rule frame
    #     Rx = rotx(heading_angle[0]) # x angle from x heading
    #     Ry = roty(heading_angle[1]) # y angle from y heading
    #     Rz = rotz(-heading_angle[2]) # z angle from -z heading

    #     l,w,h = box_size
    #     x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    #     y_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];  
    #     z_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2];
    #     corners_3d = np.vstack([x_corners,y_corners,z_corners])

    #     #bbox3=o3d.geometry.OrientedBoundingBox().create_from_points(o3d.utility.Vector3dVector(np.transpose(corners_3d)))
    #     #box3.color=[.8,1,.8]

    #     corners_3d = np.matmul(Rx, corners_3d) # apply three rotations seperately (for debugging)
    #     corners_3d = np.matmul(Ry, corners_3d)
    #     corners_3d = np.matmul(Rz, corners_3d)

    #     #bbox4=o3d.geometry.OrientedBoundingBox().create_from_points(o3d.utility.Vector3dVector(np.transpose(corners_3d)))
    #     #bbox4.color=[.6,1,.6]

    #     # convert to charles coords by rotating by 90 in the x ?
    #     corners_3d = np.matmul(rotx(np.pi/2), corners_3d)
   
    #     #bbox5=o3d.geometry.OrientedBoundingBox().create_from_points(o3d.utility.Vector3dVector(np.transpose(corners_3d)))
    #     #bbox5.color=[.4,1,.4]

    #     #corners_3d = np.transpose(corners_3d)
    #     corners_3d[0,:] = corners_3d[0,:] + center[0];
    #     corners_3d[1,:] = corners_3d[1,:] + center[1];
    #     corners_3d[2,:] = corners_3d[2,:] + center[2];

    #     #bbox6=o3d.geometry.OrientedBoundingBox().create_from_points(o3d.utility.Vector3dVector(np.transpose(corners_3d)))
    #     #bbox6.color=[.1,1,.1]

    #     corners_3d = np.transpose(corners_3d)

    # ##graphic for debugging rotation
    # #origin_base = o3d.geometry.TriangleMesh.create_coordinate_frame()
    # #origin=copy.deepcopy(origin_base).scale(0.25, center=(0,0,0))        
    # #o3d.visualization.draw_geometries([origin, bbox0, bbox1, bbox2, bbox3, bbox4, bbox5, bbox6])  

    return corners_3d

def get_3d_box_batch(box_size, heading_angle, center):
    ''' box_size: [x1,x2,...,xn,3]
        heading_angle: [x1,x2,...,xn]
        center: [x1,x2,...,xn,3]
    Return:
        [x1,x3,...,xn,8,3]
    '''
    input_shape = heading_angle.shape
    R = roty_batch(heading_angle)
    l = np.expand_dims(box_size[...,0], -1) # [x1,...,xn,1]
    w = np.expand_dims(box_size[...,1], -1)
    h = np.expand_dims(box_size[...,2], -1)
    corners_3d = np.zeros(tuple(list(input_shape)+[8,3]))
    corners_3d[...,:,0] = np.concatenate((l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2), -1)
    corners_3d[...,:,1] = np.concatenate((h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2), -1)
    corners_3d[...,:,2] = np.concatenate((w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2), -1)
    tlist = [i for i in range(len(input_shape))]
    tlist += [len(input_shape)+1, len(input_shape)]
    corners_3d = np.matmul(corners_3d, np.transpose(R, tuple(tlist)))
    corners_3d += np.expand_dims(center, -2)
    return corners_3d


if __name__=='__main__':

    # Function for polygon ploting
    import matplotlib
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    import matplotlib.pyplot as plt
    def plot_polys(plist,scale=500.0):
        fig, ax = plt.subplots()
        patches = []
        for p in plist:
            poly = Polygon(np.array(p)/scale, True)
            patches.append(poly)

    pc = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.5)
    colors = 100*np.random.rand(len(patches))
    pc.set_array(np.array(colors))
    ax.add_collection(pc)
    plt.show()
 
    # Demo on ConvexHull
    points = np.random.rand(30, 2)   # 30 random points in 2-D
    hull = ConvexHull(points)
    # **In 2D "volume" is is area, "area" is perimeter
    print(('Hull area: ', hull.volume))
    for simplex in hull.simplices:
        print(simplex)

    # Demo on convex hull overlaps
    sub_poly = [(0,0),(300,0),(300,300),(0,300)]
    clip_poly = [(150,150),(300,300),(150,450),(0,300)] 
    inter_poly = polygon_clip(sub_poly, clip_poly)
    print(poly_area(np.array(inter_poly)[:,0], np.array(inter_poly)[:,1]))
    
    # Test convex hull interaction function
    rect1 = [(50,0),(50,300),(300,300),(300,0)]
    rect2 = [(150,150),(300,300),(150,450),(0,300)] 
    plot_polys([rect1, rect2])
    inter, area = convex_hull_intersection(rect1, rect2)
    print((inter, area))
    if inter is not None:
        print(poly_area(np.array(inter)[:,0], np.array(inter)[:,1]))
    
    print('------------------')
    rect1 = [(0.30026005199835404, 8.9408694211408424), \
             (-1.1571105364358421, 9.4686676477075533), \
             (0.1777082043006144, 13.154404877812102), \
             (1.6350787927348105, 12.626606651245391)]
    rect1 = [rect1[0], rect1[3], rect1[2], rect1[1]]
    rect2 = [(0.23908745901608636, 8.8551095691132886), \
             (-1.2771419487733995, 9.4269062966181956), \
             (0.13138836963152717, 13.161896351296868), \
             (1.647617777421013, 12.590099623791961)]
    rect2 = [rect2[0], rect2[3], rect2[2], rect2[1]]
    plot_polys([rect1, rect2])
    inter, area = convex_hull_intersection(rect1, rect2)
    print((inter, area))


