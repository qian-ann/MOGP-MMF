import time
from copy import deepcopy
import numpy as np

import torch
import sys
import pickle


class dataloader(object):
    def __init__(self, para, mode='train'):
        """
        """
        "load data"
        self.para = para
        self.data = {}
        if mode == 'train':
            useTVdata=False  #是否直接调用保存的训练和验证数据。训练数据比总数据少，可以快速加载
            if useTVdata:
                try:
                    filename = f"./TVdata/TVdata.pkl"
                    with open(filename,'rb') as file:
                        self.data = pickle.load(file)
                except:
                    useTVdata = False
            if not useTVdata:
                Index_Feature=np.arange(0,para.FeatureLen,1)
                self.data['Tinputs']= {}
                self.data['Ttarget']=[]
                self.data['TtargetQ3']=[]
                self.data['Ttargidx']=[]
                self.data['Vinputs']= {}
                self.data['Vtarg']=[]
                self.data['VtargQ3']=[]
                self.data['Vtargidx']=[]
                for vf in range(len(para.VF)):
                    # 274维度，feature 256, out 8, target 8, targ 1, targidx 1
                    # Tdata = self.DatasetLoad('traindataTest',para.dir, para.VF[vf])
                    Tdata = self.DatasetLoad('traindataTest',para.dir, para.VF[vf])
                    self.data['Tinputs'][para.VF[vf]]=[]
                    for jdx in range(0,len(Tdata),para.datapara):
                        Tdataj=Tdata[jdx]
                        inputs=Tdataj[:,:,Index_Feature]
                        self.data['Tinputs'][para.VF[vf]].append(self.create_tensor(inputs[0,:,:]))
                        if vf==0:
                            target=self.create_tensor(Tdataj[0,:,264:272])
                            self.data['Ttarget'].append(target)
                            targetQ3 = deepcopy(target[:, :3])
                            targetQ3[:,0] = target[:,0]+target[:,6]+target[:,7] # C,S,T
                            targetQ3[:,1] = target[:,1]+target[:,2] #B,E
                            targetQ3[:,2] = target[:,3]+target[:,4]+target[:,5] # G,I,H
                            self.data['TtargetQ3'].append(targetQ3)
                            self.data['Ttargidx'].append(Tdataj[0,:,-1])
                            if Tdataj[0,:,-1][:(Tdataj[0,:,-1]==1).sum()].sum()<(Tdataj[0,:,-1]==1).sum():
                                print('targidx should be used')
                    self.data['Tinputs'][para.VF[vf]]=\
                        self.GoGPU(torch.cat(self.data['Tinputs'][para.VF[vf]],0))
                self.data['Ttarget']=self.GoGPU(torch.cat(self.data['Ttarget'],0))
                self.data['TtargetQ3']=self.GoGPU(torch.cat(self.data['TtargetQ3'],0))
                self.data['Ttargidx']=np.concatenate(self.data['Ttargidx'],0)

                for vf in range(len(para.VF)):
                    # 274维度，feature 256, out 8, target 8, targ 1, targidx 1
                    # Vdata = self.DatasetLoad('validata',para.dir, para.VF[vf])
                    Vdata = self.DatasetLoad('validata',para.dir, para.VF[vf])
                    self.data['Vinputs'][para.VF[vf]]=[]
                    for jdx in range(0,len(Vdata)):
                        Vdataj=Vdata[jdx]
                        inputs=Vdataj[:,:,Index_Feature]
                        self.data['Vinputs'][para.VF[vf]].append(self.create_tensor(inputs[0,:,:]))
                        if vf==0:
                            target = self.create_tensor(Vdataj[0, :, 264:272])
                            targetQ3 = deepcopy(target[:, :3])
                            targetQ3[:,0] = target[:,0]+target[:,6]+target[:,7] # C,S,T
                            targetQ3[:,1] = target[:,1]+target[:,2] #B,E
                            targetQ3[:,2] = target[:,3]+target[:,4]+target[:,5] # G,I,H
                            self.data['VtargQ3'].append(targetQ3.max(dim=-1)[1])
                            self.data['Vtarg'].append(self.create_tensor(Vdataj[0,:,-2]))
                            self.data['Vtargidx'].append(Vdataj[0,:,-1])
                            if Vdataj[0,:,-1][:(Vdataj[0,:,-1]==1).sum()].sum()<(Vdataj[0,:,-1]==1).sum():
                                print('targidx should be used')
                    self.data['Vinputs'][para.VF[vf]]=\
                        self.GoGPU(torch.cat(self.data['Vinputs'][para.VF[vf]],0))
                self.data['Vtarg']=torch.cat(self.data['Vtarg'],0)
                self.data['VtargQ3']=torch.cat(self.data['VtargQ3'],0)
                self.data['Vtargidx']=np.concatenate(self.data['Vtargidx'],0)
                self.datasave()  # 用于保存数据，因为训练数据只使用了一部分，保存后便于快速加载。
        else:  ####测试集数据
            Index_Feature = np.arange(0, para.FeatureLen, 1)
            for namei in para.testsetname:
                self.data[namei] = {}
                self.data[namei]['Vinputs'] = {}
                self.data[namei]['Vtarg'] = []
                self.data[namei]['VtargQ3'] = []
                self.data[namei]['Vtargidx'] = []
                for vf in range(len(para.VF)):
                    # 274维度，feature 256, out 8, target 8, targ 1, targidx 1
                    Vdata = self.DatasetLoad(namei, para.dir, para.VF[vf])
                    self.data[namei]['Vinputs'][para.VF[vf]] = []
                    for jdx in range(0, len(Vdata)):
                        Vdataj = Vdata[jdx]
                        inputs = Vdataj[:, :, Index_Feature]
                        self.data[namei]['Vinputs'][para.VF[vf]].append(self.create_tensor(inputs[0, :, :]))
                        if vf == 0:
                            target = self.create_tensor(Vdataj[0, :, 264:272])
                            targetQ3 = deepcopy(target[:, :3])
                            targetQ3[:, 0] = target[:, 0] + target[:, 6] + target[:, 7]  # C,S,T
                            targetQ3[:, 1] = target[:, 1] + target[:, 2]  # B,E
                            targetQ3[:, 2] = target[:, 3] + target[:, 4] + target[:, 5]  # G,I,H
                            self.data[namei]['VtargQ3'].append(targetQ3.max(dim=-1)[1])
                            self.data[namei]['Vtarg'].append(self.create_tensor(Vdataj[0, :, -2]))
                            self.data[namei]['Vtargidx'].append(Vdataj[0, :, -1])
                            if Vdataj[0, :, -1][:(Vdataj[0, :, -1] == 1).sum()].sum() < (Vdataj[0, :, -1] == 1).sum():
                                print('targidx should be used')
                    self.data[namei]['Vinputs'][para.VF[vf]] = \
                        self.GoGPU(torch.cat(self.data[namei]['Vinputs'][para.VF[vf]], 0))
                self.data[namei]['Vtarg'] = torch.cat(self.data[namei]['Vtarg'], 0)
                self.data[namei]['VtargQ3'] = torch.cat(self.data[namei]['VtargQ3'], 0)
                self.data[namei]['Vtargidx'] = np.concatenate(self.data[namei]['Vtargidx'], 0)


    def datasave(self):
        filename = f"./TVdata/TVdata.pkl"
        with open(filename, 'wb') as file:
            pickle.dump(self.data, file)

    def DatasetLoad(self,datasetname, dir, vf):
        filename = fr"{dir}{datasetname}_{vf}.pkl"
        with open(filename,'rb') as file:
            data = pickle.load(file)
        return data

    def create_tensor(self, tensor):
        if tensor.dtype == 'float64' or tensor.dtype == 'float32' or tensor.dtype == 'float16':
            if self.para.USE_GPU:
                # tensor = torch.from_numpy(tensor).half()
                tensor = torch.from_numpy(tensor).float()
            else:
                tensor = torch.from_numpy(tensor).float()
        elif tensor.dtype == 'int64' or tensor.dtype == 'int32' or tensor.dtype == 'int16':
            # tensor = torch.from_numpy(tensor).to(torch.int16)
            tensor = torch.from_numpy(tensor).int()
        else:
            print("create_tensor错误！")
            sys.exit(1)  # 1 表示非正常退出，0 表示正常退出
        return tensor

    def GoGPU(self, tensor):
        if self.para.USE_GPU:
            device = torch.device("cuda:0")
            tensor = tensor.to(device)
        return tensor










