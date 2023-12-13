import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from scipy import ndimage, misc
from PIL import Image


# Return the skew matrix formed from a 3x1 vector
def skew(vector):
    return np.array([[0, -vector[2], vector[1]],
                     [vector[2], 0, -vector[0]],
                     [-vector[1], vector[0], 0]])


def case_one():
    # Read images
    imgr = cv2.imread('direita.ppm')  # right image
    imgl = cv2.imread('esquerda.ppm') # left image

    # intrinsic parameter matrix

    fm = 403.657593 # Fical distantce in pixels
    cx = 161.644318 # Principal point - x-coordinate (pixels)
    cy = 124.202080 # Principal point - y-coordinate (pixels)
    bl = 119.929 # baseline (mm)
    # for the right camera
    Kr = np.array([[ fm, 0, cx],[0, fm, cy],[0, 0, 1.0000]])

    # for the left camera
    Kl = np.array([[fm, 0, cx],[0, fm, cy],[0, 0, 1.0000]])

    # Extrinsec parameters
    # Translation between cameras
    T1 = np.array([-bl, 0, 0])
    T = T1/np.linalg.norm(T1)
    T_hat = skew(T)
    # Rotation
    R = np.array([[ 1,0,0],[ 0,1,0],[0,0,1]])
    
    print('Intrinsic Paramenters')
    print('Left_K:\n', Kl)
    print('Right_K:\n', Kr)

    print('Extrinsic Paramenters')
    print('R:\n', R)
    print('T:\n', T)



    return imgl,imgr,Kl,Kr,bl

def plane_sweep_ncc(im_l,im_r,start,steps,wid):
    """ Find disparity image using normalized cross-correlation. """
    m,n = im_l.shape
    
    # arrays to hold the different sums
    mean_l = np.zeros((m,n))
    mean_r = np.zeros((m,n))
    s = np.zeros((m,n))
    s_l = np.zeros((m,n))
    s_r = np.zeros((m,n))
    # array to hold depth planes
    dmaps = np.zeros((m,n,steps))
    
    # compute mean of patch
    ndimage.uniform_filter(im_l,wid,mean_l)
    ndimage.uniform_filter(im_r,wid,mean_r)
    
    # normalized images
    norm_l = im_l - mean_l
    norm_r = im_r - mean_r
    
    # try different disparities
    for displ in range(steps):
      # move left image to the right, compute sums
      
      ndimage.uniform_filter(norm_l*np.roll(norm_r,displ+start),wid,s) # sum nominator
      ndimage.uniform_filter(norm_l*norm_l,wid,s_l)   
      ndimage.uniform_filter(np.roll(norm_r,displ+start)*np.roll(norm_r,displ+start),wid,s_r) # sum denominator
      # store ncc scores      
      dmaps[:,:,displ] = s/np.sqrt(np.absolute(s_l*s_r))
      
      
    # pick best depth for each pixel
    best_map = np.argmax(dmaps,axis=2) + start
    
    return best_map

  
def plane_sweep_gauss(im_l,im_r,start,steps,wid):
    """ Find disparity image using normalized cross-correlation
    with Gaussian weighted neigborhoods. """
    m,n = im_l.shape
    
    # arrays to hold the different sums
    mean_l = np.zeros((m,n))
    mean_r = np.zeros((m,n))
    s = np.zeros((m,n))
    s_l = np.zeros((m,n))
    s_r = np.zeros((m,n))
    
    # array to hold depth planes
    dmaps = np.zeros((m,n,steps))
    
    # compute mean
    ndimage.gaussian_filter(im_l,wid,0,mean_l)
    ndimage.gaussian_filter(im_r,wid,0,mean_r)
    
    # normalized images
    norm_l = im_l - mean_l
    norm_r = im_r - mean_r
    
    # try different disparities
    for displ in range(steps):
      # move left image to the right, compute sums
      ndimage.gaussian_filter(norm_l*np.roll(norm_r,displ+start),wid,0,s)  # sum nominator
      ndimage.gaussian_filter(norm_l*norm_l,wid,0,s_l)  
      ndimage.gaussian_filter(np.roll(norm_r,displ+start)*np.roll(norm_r,displ+start),wid,0,s_r) # sum denominator
      
      # store ncc scores
      dmaps[:,:,displ] = s/np.sqrt(s_l*s_r)

      # pick best depth for each pixel
      best_map = np.argmax(dmaps,axis=2)+ start
    
    
    return best_map




IL,IR,Kl,Kr,bl = case_one()
im_l = np.array(Image.open('esquerda.ppm').convert('L'),'f')
im_r = np.array(Image.open('direita.ppm').convert('L'),'f')
# starting displacement and steps
steps = 45
start = 10 


fm = Kl[0,0]
cx = Kl[0,2]
cy = Kl[1,2]



m,n = im_l.shape
wid1 = 9
wid2 = 3
res1 = plane_sweep_ncc(im_l,im_r,start,steps,wid1)
res2 = plane_sweep_gauss(im_l,im_r,start,steps,wid2)

Z = np.zeros((m,n))
for i in range(m):
  for j in range(n):
    if (res2[i,j]== 0):
      # Consider Z = inf for points that were not defined in the depthmap and are filled with zero
      Z[i,j] = np.inf 
    else: Z[i,j]=fm*bl/res2[i,j]

X,Y = np.meshgrid(np.arange(n),np.arange(m))
X = np.reshape(X, m*n)
Y = np.reshape(Y, m*n)
Z = np.reshape(Z, m*n)

x3d = np.multiply(((X - cx)/fm),Z)
y3d = np.multiply(((Y - cy)/fm),Z)

# Filter erroneous depths
good = np.where((Z>1000) & (Z<3000))

x3d = x3d[good]
y3d = y3d[good]
z3d = Z[good]

pixel_color=[]

for i in range(X[good].shape[0]):
    pixel_color.append (IL[int(Y[good][i]),int(X[good][i])])

pixel_color = np.asarray(pixel_color)

# Plot 3D points with their original colors from the image
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x3d[0:x3d.shape[0]:5], y3d[0:y3d.shape[0]:5], z3d[0:z3d.shape[0]:5], c=pixel_color[0:x3d.shape[0]:5]/255.0)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')


ax.view_init(elev=-23,azim=-91)


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x3d, y3d, z3d, c=pixel_color/255.0)
#ax.set_aspect('equal')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')




ax.view_init(elev=-57,azim=-91)
    
    
plt.show()