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


idmap = {}
test_y_class = []
for i in range(15627):
    idmap[i+1] = i+1
    #fp.write(str(i+1)+'.jpg '+str(web_test[i]-1)+'\n')
    test_y_class.append(web_test[i])

cnt = 15628
for i in range(15627,28960):
    if dic[sv_test[i-15627]] in mc:
        idmap[cnt] = i+1
        cnt+=1
        #fp.write(str(i+1)+'.jpg '+str(mc[dic[sv_test[i-15627]]]-1)+'\n')
        test_y_class.append(mc[dic[sv_test[i-15627]]])

with open('idmap.pkl',"w+") as fp:
    pkl.dump(idmap,fp,pkl.HIGHEST_PROTOCOL)

np.save('mix_test_y_class',np.array(test_y_class))

print idmap[25257]
