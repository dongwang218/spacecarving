import os
import cv2
import numpy as np
import scipy.io as sio
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--img_path', default='edmund/cropped')
ap.add_argument('--mask_path', default='edmund/cropped/masks')
args = ap.parse_args()

indexes = np.array([4, 54, 106, 161, 219, 282, 351, 427, 514, 615, 751])
diff = indexes[1:] - indexes[:-1]
diff[1:].astype(np.float) / diff[:-1]

# use 351 to 427
start = 351
end = 427

avg = 360.0 / (end - start)

start_speed = 360.0*2/(427-282)
end_speed = 2*avg - start_speed # 360.0*2/(514-351)
acceleration = (start_speed - end_speed) / (427-351)

# need to adjust the center as we cropped the image

width = 501
height = 450
start_y = 660-height+10
end_y = 660+10
start_x = 834-width//2
end_x = start_x+width

# seems having to the center is a not a good idea
# also the silhouette is not accurate, does not show shape of the side face.
K = np.array([[1.60876904e+03, 0.00000000e+00, (6.14503662e+02)/2],
              [0.00000000e+00, 1.59493511e+03, (3.65560350e+02) - start_y],
              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

# base on page 156, R and t are the camera center in world coordinate system
# so here pick world at 0,0, 0 and camera 0 at 0, 0, -radius, direction of z from camera to world
# The R t not relative to camera 0.

# 2 meters
radius = 2000
# should use camera 0 as world coordindate
frames = []
center = [0, 0, 0, 1]
head = [100, -100, 0, 1]

for i in range(end-start):
  theta = (start_speed + (start_speed - acceleration * i))/2*i
  theta_r = theta * np.pi / 180

  T = np.array([radius * np.sin(np.pi - theta_r), 0, radius * np.cos(np.pi - theta_r)]).reshape((-1, 1))

  R = np.array([[np.cos(theta_r), 0, np.sin(theta_r)],
                [0, 1, 0],
                [-np.sin(theta_r), 0, np.cos(theta_r)]])
  P1 = R.dot(np.hstack((np.eye(3), -T.reshape((-1, 1)) )))
  P = K.dot(P1)
  img = cv2.imread(os.path.join(args.img_path, 'f%04d.png' % (start+i)))
  silhouette = np.load(os.path.join(args.mask_path, 'f%04d.npy' % (start+i)))
  frames.append([img, P, K, R, T, 1, silhouette])

  #cv2.imshow('%s' % i, silhouette.astype(np.uint8)*255)
  #cv2.waitKey()

  x = P.dot(head)
  print(x/x[-1], theta, P1.dot(head)) #, R, T)

sio.savemat('carving/frames.mat', {'frames': [frames]})
