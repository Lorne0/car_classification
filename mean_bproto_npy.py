from caffe.proto import caffe_pb2
from caffe.io import blobproto_to_array
import numpy as np

mean_bin = 'imagenet_mean.binaryproto'

mean_blob = caffe_pb2.BlobProto()
mean_blob.ParseFromString(open(mean_bin,'rb').read())

mean_npy = blobproto_to_array(mean_blob)
mean_npy_shape = mean_npy.shape
mean_npy = mean_npy.reshape(mean_npy_shape[1],mean_npy_shape[2],mean_npy_shape[3])

np.save('imagenet_mean',mean_npy)
print 'Done.'

