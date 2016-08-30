import cPickle as pkl
import numpy as np
train = np.load('train_y_class.npy')
test = np.load('test_y_class.npy')

fpath = '/root/car/new_data/sv_data/image_256/'

with open("train.txt",'w+') as fp:
    for i in range(31148):
        fp.write(fpath+'train/'+str(i+1)+'.jpg '+str(train[i]-1)+'\n')

with open("test.txt",'w+') as fp:
    for i in range(13333):
        fp.write(fpath+'test/'+str(i+1)+'.jpg '+str(test[i]-1)+'\n')

