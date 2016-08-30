import cPickle as pkl
import numpy as np
import scipy.io as sio

y = []
with open('test.txt') as fp:
    for f in fp:
        f = f.strip('/')
        a = int(f.split(' ')[1])+1
        y.append(a)

print len(y)
np.save('test_y_class',np.array(y))


y = np.array(y)
c531 = np.load('c531list.npy')
c431 = np.load('c431list.npy')

web = y[:15627]
print len(web)
sv = y[15627:]
print len(sv)
sv181 = y[c431]
print len(sv181)
sv100 = y[c531]
print len(sv100)

np.save('test_y_web_class',web)
np.save('test_y_sv_class',sv)
np.save('test_y_sv181_class',sv181)
np.save('test_y_sv100_class',sv100)
