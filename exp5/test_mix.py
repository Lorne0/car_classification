import numpy as np
import cPickle as pkl
import caffe
import sys
import time

#mode = str(sys.argv[1])#train/test
mode = 'test'
gpu_id = int(sys.argv[1])

#Change path if needed
model = "deploy.prototxt";
weights = "mix_finetune/googlenet_finetune_mix_car_iter_16000.caffemodel"

caffe.set_mode_gpu()
caffe.set_device(gpu_id)

net = caffe.Net(model,weights, caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data',np.load('imagenet_mean.npy').mean(1).mean(1))
transformer.set_transpose('data',(2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data',255.0)
net.blobs['data'].reshape(1,3,224,224)

#imagepath = "../image_224/"+mode+'/'    
#Change path
imagepath = /path/to/image/

if mode=='train':
    dsize = 58945
elif mode=='test':
    dsize = 25257
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
    img = caffe.io.load_image(imagepath+str(idmap[i])+'.jpg')
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

y_class = np.load('mix_'+mode+'_y_class.npy')


y_a = y_class[:15627]
y_b = y_class[15627:]
ans_a = ans_list[:15627]
ans_b = ans_list[15627:]
ans5_a = ans5_list[:15627]
ans5_b = ans5_list[15627:]
prob_a = prob_list[:15627]
prob_b = prob_list[15627:]

print 'Average prob: ' + str(np.mean(prob_list))
print str(np.mean(prob_a)) +'/'+ str(np.mean(prob_b))

print 'Top-1 Accuracy: ' + str(sum(y_class==ans_list)/float(dsize))
print str(sum(y_a==ans_a)/float(15627)) +'/'+ str(sum(y_b==ans_b)/float(9630))


cnt=0
for i in range(dsize):
    if y_class[i] in ans5_list[i]:
        cnt+=1
print 'Top-5 Accuracy: ' + str(cnt/float(dsize))

cnt=0
for i in range(15627):
    if y_a[i] in ans5_a[i]:
        cnt+=1
cnt2=0
for i in range(9630):
    if y_b[i] in ans5_b[i]:
        cnt2+=1

print str(cnt/15627.0) +'/'+ str(cnt2/9630.0)
