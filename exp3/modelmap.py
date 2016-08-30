# Map model_id -> class_id

import cPickle as pkl

model = {}
class_id = 1
with open('train_test_split/classification_train.txt') as fp:
    for f in fp:
        s = f.split('/')
        model_id = int(s[1])
        if model_id in model:
            continue
        else:
            model[int(model_id)] = class_id
            class_id += 1

print len(model)
print model[1]
print model[5]
print model[11]
print model[1993]

with open('model_class.pkl','w+') as fp:
    pkl.dump(model,fp,pkl.HIGHEST_PROTOCOL)

