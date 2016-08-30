import numpy as np
import cPickle as pkl
import caffe
import sys
import time

mode = str(sys.argv[1])#train/test
gpu_id = int(sys.argv[2])

model = "deploy.prototxt";
weights = "mix_finetune/googlenet_finetune_mix_car_iter_20000.caffemodel"
caffe.set_mode_gpu()
caffe.set_device(gpu_id)

net = caffe.Net(model,weights, caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data',np.load('imagenet_mean.npy').mean(1).mean(1))
transformer.set_transpose('data',(2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data',255.0)
net.blobs['data'].reshape(1,3,224,224)

imagepath = "../image_224/"+mode+'/'    
if mode=='train':
    dsize = 67604
elif mode=='test':
    dsize = 28960
print 'dsize = ' + str(dsize)

with open('idmap.pkl') as fp:
    idmap = pkl.load(fp)

ans_list = []
ans5_list = np.zeros((dsize,5))
prob_list = []
stime = time.time()
sstime = time.time()
for i in range(1,dsize+1):
    if i%1000==0:
        print str(i) + ' : '+str(time.time()-stime)+'s'
        stime = time.time()
    img = caffe.io.load_image(imagepath+str(i)+'.jpg')
    net.blobs['data'].data[...] = transformer.preprocess('data',img)
    output = net.forward()
    ans_list.append(output['prob'].argmax()+1)
    prob_list.append(output['prob'].max())
    prob = output['prob'][0]
    tmp = prob.argsort()[-5:][::-1]
    tmp+=1
    ans5_list[i-1] = tmp

print 'All time : ' + str(time.time()-sstime)+'s'

print 'Length of ans_list: ' + str(len(ans_list))
ans_list = np.array(ans_list)
prob_list = np.array(prob_list)
np.save(mode+'_ans_list',ans_list)
np.save(mode+'_ans5_list',ans5_list)
np.save(mode+'_prob_list',prob_list)

y_class = np.load(mode+'_y_class.npy')
c531 = np.load("c531list.npy") 
c431 = np.load("c431list.npy") 

y_web = y_class[:15627]
y_sv = y_class[15627:]
y_sv181 = y_class[c431]
y_sv100 = y_class[c531]

ans_web = ans_list[:15627]
ans_sv = ans_list[15627:]
ans_sv181 = ans_list[c431]
ans_sv100 = ans_list[c531]

ans5_web = ans5_list[:15627]
ans5_sv = ans5_list[15627:]
ans5_sv181 = ans5_list[c431]
ans5_sv100 = ans5_list[c531]

prob_web = prob_list[:15627]
prob_sv = prob_list[15627:]
prob_sv181 = prob_list[c431]
prob_sv100 = prob_list[c531]

print weights

print 'Average prob: ' + str(np.mean(prob_list))
print str(np.mean(prob_web)) +'/'+ str(np.mean(prob_sv)) +'/'+ str(np.mean(prob_sv181)) +'/'+ str(np.mean(prob_sv100))

print 'Top-1 Accuracy: ' + str(sum(y_class==ans_list)/float(dsize))
print str(sum(y_web==ans_web)/float(15627)) +'/'+ str(sum(y_sv==ans_sv)/float(13333)) +'/'+ str(sum(y_sv181==ans_sv181)/float(9630)) +'/'+ str(sum(y_sv100==ans_sv100)/float(3703))

cnt=0
for i in range(dsize):
    if y_class[i] in ans5_list[i]:
        cnt+=1

print 'Top-5 Accuracy: ' + str(cnt/float(dsize))

cnt=0
for i in range(15627):
    if y_web[i] in ans5_web[i]:
        cnt+=1

cnt2=0
for i in range(13333):
    if y_sv[i] in ans5_sv[i]:
        cnt2+=1

cnt3=0
for i in range(9630):
    if y_sv181[i] in ans5_sv181[i]:
        cnt3+=1

cnt4=0
for i in range(3703):
    if y_sv100[i] in ans5_sv100[i]:
        cnt4+=1

print str(cnt/15627.0) +'/'+ str(cnt2/13333.0) +'/'+ str(cnt3/9630.0) +'/'+ str(cnt4/3703.0)
