# make text with label
# train.txt
# test.txt

from glob import glob

data_list = glob('./data/images/train/*/*')

train_file = open('./data/meta/train.txt', 'w')

for i in data_list:
    _id = i.split('/')[-2] + '/' + i.split('/')[-1].replace(' ', '')
    train_file.write(_id+"\n")

train_file.close()


