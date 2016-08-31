import os
cnt=0
for root, dirs, files in os.walk('data/image/'):
    for tfile in files:
        if tfile.endswith('.jpg'):
            #os.path.join(root,tfile)
            cnt+=1
print cnt
