import pandas as pd
import torch.utils.data as data
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader

def normalize_by_l2(vector):
    # 计算每行的二范数
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    norm = np.where(norm==0, 1, norm)
    return vector / norm

class DTIDataset(data.Dataset):
    def __init__(self, list_IDs, dataset, datasetLM, featureset,targetset, seqlen_max):
        self.list_IDs = list_IDs
        self.dataset = dataset
        self.datasetLM = datasetLM
        self.featureset = featureset
        self.targetset=targetset
        self.seqlen_max= seqlen_max

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        index = self.list_IDs[index]
        if len(self.featureset[1])>0:
            datai=self.dataset[index:index+1,:self.seqlen_max,self.featureset[0]]
            suplen=max(self.seqlen_max-self.datasetLM[index].shape[0],0)
            dim=self.datasetLM[index].shape[-1]
            dataLM=np.concatenate([self.datasetLM[index][np.newaxis,:self.seqlen_max,:],
                                                  np.zeros([1,suplen,dim])],-2)
            # dataLM=normalize_by_l2(dataLM)*0
            datai=np.concatenate([datai,dataLM],-1)

        else:
            datai=self.dataset[index:index+1,:self.seqlen_max,self.featureset[0]]
        y=self.dataset[index:index+1,:self.seqlen_max,self.targetset]

        return datai, y


def collate_fn(batch):
    data, y = zip(*batch)
    data=np.concatenate(data,0)
    y=np.concatenate(y,0)
    return torch.tensor(data).float(), torch.tensor(y).float()

def dataload(datasetname,para):
    dataset0={}
    for idx in range(len(datasetname)):
        dataset0[datasetname[idx]]=np.load(fr'./DataSet//{datasetname[idx]}_sort.npy')
        if 'T5' in para['trainmode']:
            with open(fr'./dataLM//{datasetname[idx]}LM.npy', 'rb') as file:
                dataset0[datasetname[idx]+'LM'] = pickle.load(file)
        elif 'Sa' in para['trainmode']:
            with open(fr'./SatData//{datasetname[idx]}Sa.npy', 'rb') as file:
                dataset0[datasetname[idx]+'LM'] = pickle.load(file)
        else:
            dataset0[datasetname[idx] + 'LM']=[]
    dataset={}
    datasetloader={}
    for idx in range(len(datasetname)):
        dataset[datasetname[idx]] = DTIDataset(np.arange(0,dataset0[datasetname[idx]].shape[0]),
                                               dataset0[datasetname[idx]],dataset0[datasetname[idx]+'LM'],
                                               para['featureset'],para['targetset'],para['seqlen_max'])
        if idx ==0:
            datasetloader[datasetname[idx]] = DataLoader(dataset[datasetname[idx]], batch_size=para['BATCH_SIZE'], shuffle=True, collate_fn=collate_fn)
            datasetloader[datasetname[idx]+'Test'] = DataLoader(dataset[datasetname[idx]], batch_size=para['BATCH_SIZE'], shuffle=False, collate_fn=collate_fn)
        else:
            datasetloader[datasetname[idx]] = DataLoader(dataset[datasetname[idx]], batch_size=para['BATCH_SIZE'], shuffle=False, collate_fn=collate_fn)


    return datasetloader, len(dataset0['traindata']), para['trainmode']
