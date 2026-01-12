
import numpy as np
import torch
import torch.nn.functional as F


class FuseModel():
    def __init__(self, para):
        """
        用基础ELM初始化的时候进行一次隐藏层结点寻优
        参数solution提供隐藏层节点数、权重、偏置
        特制ELM应该有的特点：
            种群中的ELM应该具有相同的激活函数，
            用不到random_state
            train_model()函数不需要再产生随机权重和偏置
        """
        self.para = para
        self.output_weight_ = None

    def FeatureFuse(self, data, individual, func, mode='train'):
        with torch.no_grad():
            torch.cuda.empty_cache()  # 清理未被引用的显存
            if mode=='train':
                self.train_model(data, func)
                individual.output_weight_ = self.output_weight_.half().cpu()
                accuracy, failureVector, ObjValueN = self.hitting_rate(data, func)
                return accuracy, failureVector, ObjValueN
            else:
                self.output_weight_=individual.output_weight_.float()
                y_pred, accuracy, failureVector, ObjValueN = self.hitting_rate(data, func,'test')
                return y_pred, accuracy, failureVector, ObjValueN

    def train_model(self, data, func):
        """训练模型过程中需要保存的数据是输入权重，偏置以及输出权重"""
        para = self.para
        # CNN输出到隐藏层
        with torch.no_grad():  #使得计算过程中不计算梯度
            input_map_=[data['Tinputs'][vf] for vf in para.VF]
            input_map_ = [F.normalize(input_map, p=2, dim=-1, eps=1e-5) for input_map in input_map_]  # L2归一化
            input_map_ = func(*input_map_)
            ndata = input_map_.shape[0]
            # relu, 另外，特征维度增加一维度的1进而便于求Ax+b中的b
            input_map_ = torch.cat([torch.relu(input_map_), torch.ones(ndata, 1).to(input_map_.device)], -1)
            y = torch.cat([data['Ttarget'],data['TtargetQ3']], -1)
            self.output_weight_=(torch.linalg.pinv(input_map_)@y).float()
            # residual:  input_map_ @ x - y

    def hitting_rate(self, data, func, model='train'):
        para=self.para
        input_map_ = [data['Vinputs'][vf] for vf in para.VF]
        input_map_= [F.normalize(fe,p=2,dim=-1,eps=1e-5) for fe in input_map_]  #L2归一化
        input_map_ = func(*input_map_)
        ndata = input_map_.shape[0]
        input_map = torch.cat([torch.relu(input_map_), torch.ones(ndata, 1).to(input_map_.device)], -1)
        y_pred = torch.matmul(input_map, self.output_weight_.to(input_map.device))
        y_pred = y_pred.reshape(-1,y_pred.shape[-1])
        accuracy, hitting, accuracyN=self.get_accuracy(y_pred,data)
        if model=='train':
            return accuracy, hitting, accuracyN
        else:
            return y_pred, accuracy, hitting, accuracyN

    def get_accuracy(self,y_pred,data):
        # y_pred = y_pred.reshape(-1, y_pred.shape[-1]).float()
        #Q8
        y_pred0 = y_pred[:,0:8].argmax(axis=1).cpu().detach().numpy()
        y0 = data['Vtarg'].cpu().detach().numpy()
        error0 = y_pred0 - y0
        hitting = np.where(abs(error0) < 0.1, 1, 0)
        accuracy = (np.count_nonzero(hitting) / len(y0))

        #Q3
        y_pred1 = y_pred[:,8:11].argmax(axis=1).cpu().detach().numpy()
        y1 = data['VtargQ3'].cpu().detach().numpy()
        error1 = y_pred1 - y1
        hitting1 = np.where(abs(error1) < 0.1, 1, 0)
        accuracy1 = (np.count_nonzero(hitting1) / len(y1))
        if 'Q8' in self.para.QName:
            accuracy0=accuracy
            hitting0=hitting
            if 'Q3' in self.para.QName:
                accuracy0 = accuracy0+accuracy1
                hitting0=np.concatenate([hitting0,hitting1],-1)
        elif 'Q3' in self.para.QName:
            accuracy0 = accuracy1
            hitting0=hitting1
        accuracyN = ['%.4f' % accuracy, '%.4f' % accuracy1 ]

        return accuracy0, hitting0, accuracyN



class compare_funcs():
    def __init__(self, mode):
        self.mode = mode

    def compare_funcs(self,HMMCNN1,HMMCNN2,HMMRNN1,HMMRNN2,PSSMCNN1,PSSMCNN2,PSSMRNN1,PSSMRNN2,
                      T5CNN1,T5CNN2,T5RNN1,T5RNN2,SaCNN1,SaCNN2,SaRNN1,SaRNN2):
        if self.mode=='HMMCNN1':
            x= HMMCNN1
            return F.normalize(x, p=2, dim=-1, eps=1e-5)
        elif self.mode=='HMMCNN2':
            x= HMMCNN2
            return F.normalize(x, p=2, dim=-1, eps=1e-5)
        elif self.mode=='HMMRNN1':
            x= HMMRNN1
            return F.normalize(x, p=2, dim=-1, eps=1e-5)
        elif self.mode=='HMMRNN2':
            x= HMMRNN2
            return F.normalize(x, p=2, dim=-1, eps=1e-5)
        elif self.mode=='PSSMCNN1':
            x= PSSMCNN1
            return F.normalize(x, p=2, dim=-1, eps=1e-5)
        elif self.mode=='PSSMCNN2':
            x= PSSMCNN2
            return F.normalize(x, p=2, dim=-1, eps=1e-5)
        elif self.mode=='PSSMRNN1':
            x= PSSMRNN1
            return F.normalize(x, p=2, dim=-1, eps=1e-5)
        elif self.mode=='PSSMRNN2':
            x= PSSMRNN2
            return F.normalize(x, p=2, dim=-1, eps=1e-5)
        elif self.mode=='T5CNN1':
            x= T5CNN1
            return F.normalize(x, p=2, dim=-1, eps=1e-5)
        elif self.mode=='T5CNN2':
            x= T5CNN2
            return F.normalize(x, p=2, dim=-1, eps=1e-5)
        elif self.mode=='T5RNN1':
            x= T5RNN1
            return F.normalize(x, p=2, dim=-1, eps=1e-5)
        elif self.mode=='T5RNN2':
            x= T5RNN2
            return F.normalize(x, p=2, dim=-1, eps=1e-5)
        elif self.mode=='SaCNN1':
            x= SaCNN1
            return F.normalize(x, p=2, dim=-1, eps=1e-5)
        elif self.mode=='SaCNN2':
            x= SaCNN2
            return F.normalize(x, p=2, dim=-1, eps=1e-5)
        elif self.mode=='SaRNN1':
            x= SaRNN1
            return F.normalize(x, p=2, dim=-1, eps=1e-5)
        elif self.mode=='SaRNN2':
            x= SaRNN2
            return F.normalize(x, p=2, dim=-1, eps=1e-5)
        elif self.mode=='Concatation':
            x1 = HMMCNN1+HMMCNN2+HMMRNN1+HMMRNN2
            x1=F.normalize(x1, p=2, dim=-1, eps=1e-5)
            x2 = PSSMCNN1+PSSMCNN2+PSSMRNN1+PSSMRNN2
            x2=F.normalize(x2, p=2, dim=-1, eps=1e-5)
            x3 = T5CNN1+T5CNN2+T5RNN1+T5RNN2
            x3=F.normalize(x3, p=2, dim=-1, eps=1e-5)
            x4 = SaCNN1+SaCNN2+SaRNN1+SaRNN2
            x4=F.normalize(x4, p=2, dim=-1, eps=1e-5)
            x = torch.concatenate([x1,x2,x3,x4], axis=-1)
            return F.normalize(x, p=2, dim=-1, eps=1e-5)
        elif self.mode=='Concatation3':
            x1 = HMMCNN1+HMMCNN2+HMMRNN1+HMMRNN2
            x1=F.normalize(x1, p=2, dim=-1, eps=1e-5)
            x2 = PSSMCNN1+PSSMCNN2+PSSMRNN1+PSSMRNN2
            x2=F.normalize(x2, p=2, dim=-1, eps=1e-5)
            x4 = SaCNN1+SaCNN2+SaRNN1+SaRNN2
            x4=F.normalize(x4, p=2, dim=-1, eps=1e-5)
            x = torch.concatenate([x1,x2,x4], axis=-1)
            return F.normalize(x, p=2, dim=-1, eps=1e-5)
        elif self.mode=='Add':
            x=HMMCNN1+HMMCNN2+HMMRNN1+HMMRNN2+ \
              PSSMCNN1 + PSSMCNN2 + PSSMRNN1 + PSSMRNN2+ \
              T5CNN1 + T5CNN2 + T5RNN1 + T5RNN2+ \
              SaCNN1 + SaCNN2 + SaRNN1 + SaRNN2
            return F.normalize(x, p=2, dim=-1, eps=1e-5)
        elif self.mode=='Mul':
            x=HMMCNN1*HMMCNN2*HMMRNN1*HMMRNN2* \
              PSSMCNN1 * PSSMCNN2 * PSSMRNN1 * PSSMRNN2 * \
              T5CNN1 * T5CNN2 * T5RNN1 * T5RNN2 * \
              SaCNN1 * SaCNN2 * SaRNN1 * SaRNN2
            return F.normalize(x, p=2, dim=-1, eps=1e-5)
        elif self.mode=='Max':
            xx = torch.stack([HMMCNN1,HMMCNN2,HMMRNN1,HMMRNN2, \
              PSSMCNN1,PSSMCNN2 ,PSSMRNN1 , PSSMRNN2, \
              T5CNN1 , T5CNN2 , T5RNN1 , T5RNN2, \
              SaCNN1, SaCNN2 , SaRNN1 , SaRNN2])
            x,_=torch.max(xx,dim=0)
            return F.normalize(x, p=2, dim=-1, eps=1e-5)
        elif self.mode=='Min':
            xx = torch.stack([HMMCNN1,HMMCNN2,HMMRNN1,HMMRNN2, \
              PSSMCNN1,PSSMCNN2 ,PSSMRNN1 , PSSMRNN2, \
              T5CNN1 , T5CNN2 , T5RNN1 , T5RNN2, \
              SaCNN1, SaCNN2 , SaRNN1 , SaRNN2])
            x,_=torch.min(xx,dim=0)
            return F.normalize(x, p=2, dim=-1, eps=1e-5)

