import cPickle as pkl
import numpy as np
import scipy.io as sio
web_train = np.load('../web_train_y_class.npy')
web_test = np.load('../web_test_y_class.npy')
sv_train = np.load('../sv_train_y_class.npy')
sv_test = np.load('../sv_test_y_class.npy')

sv = sio.loadmat('../../sv_data/sv_make_model_name.mat')
sv = sv['sv_make_model_name']
dic = {}
for i in range(281):
        dic[i+1] = int(sv[i,2][0][0])

with open('../../../model_class.pkl') as fp:
    mc = pkl.load(fp)

errorcnt = 0
with open("train.txt",'w+') as fp:
    for i in range(36456):
        fp.write(str(i+1)+'.jpg '+str(web_train[i]-1)+'\n')
    for i in range(36456,67604):
        if dic[sv_train[i-36456]] in mc:
            fp.write(str(i+1)+'.jpg '+str(mc[dic[sv_train[i-36456]]]-1)+'\n')
        else:
            errorcnt+=1
            #fp.write(str(i+1)+'.jpg '+str(10000)+'\n')
print errorcnt


errorcnt = 0
with open("test.txt",'w+') as fp:
    for i in range(15627):
        fp.write(str(i+1)+'.jpg '+str(web_test[i]-1)+'\n')
    for i in range(15627,28960):
        if dic[sv_test[i-15627]] in mc:
            fp.write(str(i+1)+'.jpg '+str(mc[dic[sv_test[i-15627]]]-1)+'\n')
        else:
            errorcnt+=1
            #fp.write(str(i+1)+'.jpg '+str(10000)+'\n')
print errorcnt

cnt=0
for i in range(281):
    if dic[i+1] not in mc:
        cnt+=1
print cnt
