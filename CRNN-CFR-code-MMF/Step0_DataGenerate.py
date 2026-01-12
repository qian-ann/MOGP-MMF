import numpy as np
import torch
import pickle

# ["A C D E F G H I K L M N P Q R S T V W Y X"]
ProtT5idx = torch.tensor([3, 22, 10,  9, 15,  5, 20, 12, 14,  4, 19, 17, 13, 16,  8,  7, 11,  6, 21, 18, 23])
ProtT5idx = ProtT5idx.unsqueeze(1).numpy()

name=['traindata','validata','testdata','CASP10data','CB513data','CASP11data','CASP12data','CASP13data','CASP14data']
datalen={}
for idx in range(len(name)):
    data = np.load(f'./DataSet/{name[idx]}.npy')
    seqlen=(1-data[:,:,41]).sum(-1) #对有氨基酸的地方标记1，求和可以得到蛋白质长度
    seqidx=np.argsort(seqlen)
    data=data[seqidx,:,:]  #按长度从小到大排序
    data0=data[:,:,20:41]  #取出 #A C D E F G H I K L M  N  P  Q   R   S   T   V   W   Y X 21个氨基酸位置
    data0=np.append(data0, np.zeros([data0.shape[0],1,21]), 1)  #序列长度+1，补0
    data0mask=data0.sum(axis=2)  #根据序列长度制作掩码
    data0=np.dot(data0,ProtT5idx) #把onehot换成编号
    data1=np.append(data[:,:,41:42], np.ones([data.shape[0],1,1]), 1) #前面序列长度+1，这里序列结尾补1. 1表示没有氨基酸
    data2=np.append(np.zeros([data.shape[0],1,1]), data[:,:,41:42], 1) #前面序列长度+1，这里序列开头补0.
    data1=data1-data2  # 序列结尾的位置的后一位为1
    data0=data0+data1  #序列结尾后一位编号为1。这也是为什么要增加1位。
    data0mask=data0mask[:,:,np.newaxis]+data1 #掩码矩阵在序列结尾的后一位上补上1。否则T5识别不了结尾。
    data00 = np.append(data0,data0mask,2)
    datalen[name[idx]]=seqlen

    ff = f'./DataSet/{name[idx]}_sort.npy'
    np.save(ff, data)
    ff = f'./DataSet/{name[idx]}0.npy'
    np.save(ff,data00)
dir2 = f'./DataSet/datalen.pkl'
with open(dir2, 'wb') as f:
    pickle.dump(datalen, f)
