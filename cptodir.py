import os
import os.path
from shutil import copyfile

desdir = 'new_data/sv_data/image_all/train/'
srcdir = 'new_data/sv_data/image/'
with open('new_data/sv_data/train_surveillance.txt') as fp:
    cnt=1
    for f in fp:
        s = f.strip()
        copyfile(srcdir+s,desdir+str(cnt)+'.jpg')
        cnt+=1
print cnt-1

desdir = 'new_data/sv_data/image_all/test/'
srcdir = 'new_data/sv_data/image/'
with open('new_data/sv_data/test_surveillance.txt') as fp:
    cnt=1
    for f in fp:
        s = f.strip()
        copyfile(srcdir+s,desdir+str(cnt)+'.jpg')
        cnt+=1
print cnt-1
