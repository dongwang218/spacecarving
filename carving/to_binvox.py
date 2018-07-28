import binvox_rw
import numpy as np
voxels = np.load('../voxels.npy')
start = np.min(voxels, axis = 0)
end = np.max(voxels, axis = 0)
res = np.amax(voxels[1,:] - voxels[0,:])

dim = np.max(np.ceil((end-start) / res + 2).astype(np.int))
dims = np.array([dim, dim, dim])
data = np.zeros(dims, dtype = np.bool)

for voxel in voxels:
  index = np.floor((voxel - start) / res).astype(np.int)
  data[index[0], index[1], index[2]] = True

binvox = binvox_rw.Voxels(data, dims, np.zeros(3), scale=res, axis_order='xzy')
binvox_rw.write(binvox, open('dong.binvox', 'wb'))
