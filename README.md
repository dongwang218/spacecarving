# 3D head reconstruction using space carving

Experimental study of using space carving to reconstruct 3d head model. These are the steps involved:
* Setup a fix camera record a video where I sit on a spinner.
* Use estimated spin speed to construct a series of camera matrix.
* Use [Mask RCNN](https://github.com/matterport/Mask_RCNN) to segment rough silhouette, then get detailed mask via opencv grabcut.
* Use space carving to carve voxels outside of each silhouette.
* Visualize resulting voxel using [viewvox](http://www.patrickmin.com/viewvox).

The reconstruction lacks details though, possible improvements:
* More accurate silhouette.
* Use face keypoint matching to improve camera matrix.
* Diverse views, eg from top and bottom directions.
* Adding [photometric stereo constraints](http://grail.cs.washington.edu/projects/liangshu/0351.pdf).
* Using a [SMPL](http://smpl.is.tue.mpg.de/) human body model.

![alt tag](https://raw.githubusercontent.com/dongwang218/spacecarving/carve/dong_sideview.png)


A strong prior like a 3dmm can generates more detailed face reconstruction, eg using [eos](https://github.com/patrikhuber/eos/) on one frame of the captured video.
![alt tag](https://raw.githubusercontent.com/dongwang218/spacecarving/carve/dong_3dmm.png)
