import cv2
import os
import glob

pole = [834, 660]
frames = glob.glob('*.png')

width = 501
height = 450
start_y = 660-height+10
end_y = 660+10
start_x = 834-width//2
end_x = start_x+width

for f in frames:
  frame = cv2.imread(f)
  newf = frame[start_y:end_y, start_x:end_x]
  cv2.imwrite(os.path.join('cropped', f), newf)
