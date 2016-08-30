import cPickle as pkl
import numpy as np
train = np.load('../train_y_class.npy')
test = np.load('../test_y_class.npy')

with open("train.txt",'w+') as fp:
    for i in range(36456):
        fp.write('/root/car/new_data/data/image_256/train/'+str(i+1)+'.jpg '+str(train[i]-1)+'\n')

with open("test.txt",'w+') as fp:
    for i in range(15627):
        fp.write('/root/car/new_data/data/image_256/test/'+str(i+1)+'.jpg '+str(test[i]-1)+'\n')

