import os
import cv2
import numpy as np
import scipy.io as sio
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--img_path', default='dong')
ap.add_argument('--mask_path', default='dong/masks2')
args = ap.parse_args()

# first cycle is slower
#indexes = np.array([15, 88, 159, 233, 315, 407, 506, 619])
# indexes = np.array([54, 123, 195, 272, 360, 455, 560]), camera moved up
#
# select to have the nose be right most and left most
indexes = np.array([395, 515, 641, 777, 922, 1081])
#indexes = np.array([31, 98, 169, 243])

#was = np.array([4, 54, 106, 161, 219, 282, 351, 427, 514, 615, 751])
diff = indexes[1:] - indexes[:-1]
diff[1:].astype(np.float) / diff[:-1]

# use 351 to 427
start = 641
end = 777

avg = 360.0 / (end - start)

adjust = 45 # so that 717 get 180
start_speed = 360.0/(717-584 + adjust)
end_speed = 2*avg - start_speed  # 360.0*2/(514-351)
acceleration = (start_speed - end_speed) / (end-start)

# need to adjust the center as we cropped the image

# seems having to the center is a not a good idea
# also the silhouette is not accurate, does not show shape of the side face.
K = np.array([[3666.0, 0, 1280/2],
              [0, 3666, 720/2],
              [0, 0.00000000e+00, 1]])

radius = 2000
# should use camera 0 as world coordindate
frames = []
center = [0, 0, 0, 1]
nose = [285, 45, 0, 1] # adjust nose world to match its image 1164, 440

# 395: 1115, 415 looking at right
# 462: 154, 432 looking at left
# 515: 1144, 431
# 584: 138, 442
# 641: 1151, 433
# 717: 123, 452
# 777: 1164, 440
# 859: 113, 463
# 922: 1173, 445
# 1011: 100, 462
# 1081: 1182, 448
# 1180: 94, 463

# adjust so that nose matches the image coordindate in 0 and pi rotation, this conditioned on center matches and K is correct

for i in range(end-start):
  theta = -(start_speed + (start_speed - acceleration * i))/2*i
  theta_r = theta * np.pi / 180

  # camera center in world coordindate
  C = np.array([radius * np.cos(-np.pi/2 + theta_r), 0, radius * np.sin(-np.pi/2 + theta_r)]).reshape((-1, 1))

  # world axis in camera coordindates
  R = np.array([[np.cos(theta_r), 0, -np.sin(theta_r)],
                [0, 1, 0],
                [np.sin(theta_r), 0, np.cos(theta_r)]])
  P1 = R.T.dot(np.hstack((np.eye(3), -C.reshape((-1, 1)) )))
  P = K.dot(P1)

  x = P.dot(nose)
  print('id', start+i, x/x[-1], theta, P1.dot(nose))#, R, C)

  img = cv2.imread(os.path.join(args.img_path, 'f%04d.png' % (start+i)))
  mask_file = os.path.join(args.mask_path, 'f%04d.npy' % (start+i))
  if os.path.exists(mask_file):
    # we have manully removed some mask that looks bad
    silhouette = np.load(mask_file)
    frames.append([img, P, K, R, C, 1, silhouette])


sio.savemat('carving/frames.mat', {'frames': [frames]})
