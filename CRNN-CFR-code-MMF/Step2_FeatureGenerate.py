


import torch
import torch.nn as nn
import numpy as np
import pickle
import time
import math
from torch.optim.lr_scheduler import StepLR
import torch.nn.utils.rnn as rnn_utils
import sys
from dataloader import dataload


# HMM，PSSM，T5, Sa
# CNN1， CNN2， RNN1，RNN2
features=['HMM','PSSM','T5','Sa']
# features=['HMM']
viewnames=['CNN1','CNN2','RNN1','RNN2']
# viewnames=['CNN2']
for fe in features:
    for view in viewnames:
        trainmode=fe+view  #最后一位数字表示层数  # 1 HMM+Onehot, 2 Onehot+PSSM, 3 Onehot+T5

        datasetname=['traindata','validata','testdata','CASP10data','CB513data','CASP11data','CASP12data','CASP13data','CASP14data']

        dir = fr'./parameter/modelpara_{trainmode}.pth'

        if ('HMM' in trainmode) and ('PSSM' in trainmode):
            featureset = [range(0, 63), []]  # Onehot+PSSM+HMM
        elif 'ONE' in trainmode:
            featureset = [range(20,42),[]]   #Onehot
        elif 'HMM' in trainmode:
            featureset = [range(0,42),[]]   #HMM+Onehot
        elif 'PSSM' in trainmode:
            featureset = [range(20,63),[]]  #Onehot+PSSM
        elif 'T5' in trainmode:
            featureset = [range(20,42),range(0,1024)]  #Onehot+T5
        elif 'Sa' in trainmode:
            featureset = [range(20,42),range(0,480)]  #Onehot+Sa
            # featureset = [range(20,42),range(0,1280)]  #Onehot+Sa  650AF2
        else:
            print(f"trainmode: {trainmode} 错误！")
            sys.exit(1)  # 1 表示非正常退出，0 表示正常退出
        targetset = range(63,72)

        USE_GPU = True
        N_EPOCHS=0
        # N_EPOCHS=0
        # learning rate decay
        learning_rate = 1e-3
        LAM = 0
        rate_decay = 0.5
        dropout_rateLRI = 0.2
        kernel_size = 9
        BATCH_SIZE = 64
        BATCH_SIZE_TEST = 128

        seqlen_max = 700
        LABELS_NUM = 8
        INPUTS_SIZE = len(featureset[0])+len(featureset[1])

        out_channels = 256
        LRI_SIZE = 512
        LRI_SIZE1 = 256
        LRI_SIZE2 = int(LRI_SIZE1/8)
        HIDDEN_SIZE = 256
        bidirectional = True
        if 'RNN' in trainmode:
            MIDDLE_SIZE=HIDDEN_SIZE*2*bidirectional
        else:
            MIDDLE_SIZE=out_channels
        N_LAYER = int(trainmode[-1])
        pad_len = int((kernel_size-1)/2)
        dropout_rate_Conv = 0.2
        dropout_rateRNN = 0.2
        if bidirectional: dropout_rateRNN = 0


        # dropout
        dropoutLRI = nn.Dropout(p=dropout_rateLRI)
        dropoutRNN = nn.Dropout(p=dropout_rateRNN)
        dropoutConv = nn.Dropout(p=dropout_rate_Conv)

        class CNNLayer(nn.Module):
            def __init__(self):
                super(CNNLayer, self).__init__()
                self.conv = torch.nn.Sequential(
                    torch.nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=pad_len),
                    torch.nn.ReLU(),
                )
                self.layernorm = nn.LayerNorm(out_channels)
                self.ReLu = torch.nn.ReLU()

            def forward(self, x):
                # x （batch_size, seqlen_Max, out_channels）
                x = self.layernorm(x + self.conv(x.permute(0, 2, 1)).permute(0, 2, 1))
                x = self.ReLu(x)
                if self.training:
                    x = dropoutConv(x)
                return x


        class RNNLayer(nn.Module):
            def __init__(self):
                super(RNNLayer, self).__init__()
                self.n_directions = 2 if bidirectional else 1
                # input: (batchSize,seqLen,input_size)   output: (batchSize,seqLen,hiddenSize*nDirections)
                self.rnn = nn.LSTM(out_channels, HIDDEN_SIZE, N_LAYER, batch_first=True, bidirectional=bidirectional, dropout=dropout_rateRNN)

            def _init_hidden(self, batch_size):
                hidden = torch.zeros(N_LAYER * self.n_directions, batch_size, HIDDEN_SIZE)
                return (create_tensor(hidden), create_tensor(hidden))

            def forward(self, x, data_len):
                batch_size = x.size(0)
                seq_len = x.size(1)
                output = rnn_utils.pack_padded_sequence(x, data_len, batch_first=True)
                output, (_, _) = self.rnn(output, self._init_hidden(batch_size))
                output, _ = rnn_utils.pad_packed_sequence(output, batch_first=True)
                output = torch.cat([output, abs(output[:, 0, :].unsqueeze(1).repeat(1, seq_len-output.shape[1], 1)*0)], 1)
                if self.training:
                    output = dropoutRNN(output)
                return output

        class CRNNmodel(nn.Module):
            def __init__(self):
                super(CRNNmodel, self).__init__()

                self.FC0 = torch.nn.Sequential(
                    torch.nn.Linear(INPUTS_SIZE, out_channels),
                    torch.nn.ReLU(),
                )
                if 'CNN' in trainmode or 'RNN' in trainmode:
                    self.CNNlayer = nn.ModuleList([CNNLayer() for _ in range(N_LAYER)])
                    self.RNNlayer = RNNLayer()

                if 'SAT' in trainmode:
                    self.SAT = nn.ModuleList([torch.nn.MultiheadAttention(out_channels, 4, batch_first=True)
                                             for _ in range(N_LAYER)])

                self.FC1 = torch.nn.Sequential(
                     torch.nn.Linear(MIDDLE_SIZE, LRI_SIZE),
                     torch.nn.ReLU(),
                     torch.nn.Linear(LRI_SIZE, LRI_SIZE1),
                     torch.nn.ReLU(),
                )
                self.FC2 = torch.nn.Linear(LRI_SIZE1, LABELS_NUM)

            def forward(self, x, data_len):
                #  (batch_size, seqlen_max, INPUTS_SIZE)==>(batch_size, seqlen_max, out_channels)
                x = self.FC0(x)

                if 'CNN' in trainmode:
                    ## CNN (batch_size, seqlen_max, out_channels)
                    for layer in self.CNNlayer:
                        x = layer(x)
                if 'RNN' in trainmode:
                    ## RNN (batch_size, seqlen_max, hidden_size * nDirections)
                    x = self.RNNlayer(x,data_len)
                if 'SAT' in trainmode:
                    ## SAT (batch_size, seqlen_max, out_channels)
                    for layer in self.SAT:
                        x, _ = layer(x, x, x)


                x1 = self.FC1(x)
                if self.training:
                    x1 = dropoutLRI(x1)

                x = self.FC2(x1)

                return x, x1

        def create_tensor(tensor):
            if USE_GPU:
                device = torch.device("cuda:0")
                tensor = tensor.to(device)
            return tensor

        def make_tensors(data,y):

            data_len = (1-y[:,:,-1]).sum(axis=1)
            data_idx = np.argsort(-data_len)
            data_len = data_len[data_idx]
            inputs = data[data_idx,:,:]
            target = y[data_idx,:,:]

            targ = target.max(dim=2)[1]
            targidx = ~np.isin(targ, np.array([8]))

            return create_tensor(inputs), create_tensor(targ), targidx, create_tensor(target[:,:,:-1]), data_len

        def time_since(since):
            s = time.time() - since
            m = math.floor(s / 60)
            s -= m * 60
            return '%dm %ds' % (m, s)

        def trainModel(epoch):
            # training =True
            CRNNmodel.train()
            total_loss = 0
            for i, (data,y) in enumerate(datasetloader['traindata']):
                # inputs (batch_size, seqlen_max, INPUTS_SIZE)
                # target (batch_size, seqlen_max, LABELS_NUM)
                inputs, _, targidx, target,data_len = make_tensors(data,y)
                output, _ = CRNNmodel(inputs,data_len)
                batchlen=output.shape[0]
                datalen= i*BATCH_SIZE+batchlen
                loss = criterion(output.reshape(-1, LABELS_NUM)[targidx.reshape(-1),:], target.reshape(-1, LABELS_NUM)[targidx.reshape(-1),:])*100
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                if i % 10 == 0:
                    print(f'[{time_since(start)}] Epoch {epoch} ', end='')
                    print(f'[{datalen}/{traindatalen}] ', end='')
                    try:
                        print(f'loss={total_loss / (datalen)}')
                    except:
                        pass
            return total_loss


        def valiModel(dataloader,datasetname):
            # 将training =False
            CRNNmodel.eval()
            correct = 0
            total = 0
            datafeature=[]
            print("vali or test trained model in",f'{datasetname}')
            # 表示不需要求梯度
            with torch.no_grad():
                for i, (data,y) in enumerate(dataloader):
                    # inputs (batch_size, seqlen_max, INPUTS_SIZE)
                    # target (batch_size, seqlen_max, LABELS_NUM)
                    inputs, targ, targidx, _, data_len = make_tensors(data,y)
                    output, _ = CRNNmodel(inputs,data_len)
                    output = output.reshape(-1, LABELS_NUM)
                    pred = output.max(dim=1)[1][targidx.reshape(-1)].cpu()
                    targ = targ.reshape(-1).cpu()[targidx.reshape(-1)]
                    correct += pred.eq(targ.view_as(pred)).sum().item()
                    total+= len(targ)
                percent='%.3f' % (100 * correct / total)
                print(f'Validation set: Accuracy {correct}/{total} {percent}%')
            return correct / total


        def savefeature(dataloader,datasetname):
            # 将training =False
            CRNNmodel.eval()
            correct = 0
            total = 0
            filename = fr"FeatureGenerate/{datasetname}_{trainmode}.pkl"
            datafeature=[]
            print("vali or test trained model in",f'{datasetname}')
            # 表示不需要求梯度
            with torch.no_grad():
                for i, (data,y) in enumerate(dataloader):
                    # inputs (batch_size, seqlen_max, INPUTS_SIZE)
                    # target (batch_size, seqlen_max, LABELS_NUM)
                    inputs, targ, targidx, target, data_len = make_tensors(data,y)
                    output, x = CRNNmodel(inputs,data_len)
                    # feature generate
                    # 274维度，feature 256, out 8, target 8, targ 1, targidx 1
                    x=torch.cat([x,output,target,targ.unsqueeze(-1),torch.tensor(targidx).unsqueeze(-1).to(x.device)*1],-1).cpu().detach().numpy()
                    output = output.reshape(-1, LABELS_NUM)
                    pred = output.max(dim=1)[1][targidx.reshape(-1)].cpu()
                    targ = targ.reshape(-1).cpu()[targidx.reshape(-1)]
                    correct += pred.eq(targ.view_as(pred)).sum().item()
                    total += len(targ)
                    for xi in range(x.shape[0]):
                        datafeature.append(x[xi:xi+1,:int(data_len[xi]),:])
                percent = '%.4f' % (100 * correct / total)
                print(f'Validation set: Accuracy {correct}/{total} {percent}%')
                with open(filename, 'wb') as file:
                    pickle.dump(datafeature, file)
            return percent



        para={'trainmode': trainmode, 'featureset': featureset, 'targetset': targetset,
              'seqlen_max': seqlen_max, 'BATCH_SIZE': BATCH_SIZE}

        if ('trainmode1' not in globals()) or (trainmode1[:2]!=trainmode[:2]):
            datasetloader,traindatalen,trainmode1=dataload(datasetname,para)

        CRNNmodel = CRNNmodel()

        start_epoch = 1
        criterion = torch.nn.CrossEntropyLoss()
        # optimizer = torch.optim.Adam(CRNNmodel.parameters(), betas=[0.9,0.99], lr=learning_rate, weight_decay=LAM)
        optimizer = torch.optim.Adam(CRNNmodel.parameters(), lr=learning_rate, weight_decay=LAM)
        scheduler = StepLR(optimizer, step_size=1, gamma=rate_decay)

        start = time.time()
        if USE_GPU:
            device = torch.device("cuda:0")
            CRNNmodel.to(device)
            criterion.to(device)

        print("Training for %d epochs..." % N_EPOCHS)
        acc_list = []
        acc_list = list(acc_list)
        valiacc_list = []
        valiacc_list = list(valiacc_list)
        for epoch in range(start_epoch, N_EPOCHS + 1):
            # Train cycle
            trainModel(epoch)
            valiacc = valiModel(datasetloader['validata'],'validata')
            valiacc_list.append(valiacc)
            if epoch == 1: minacc = valiacc
            if np.array(valiacc_list)[-1] < minacc:
                # break
                print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
                scheduler.step()
                checkpoint = torch.load(dir)
                CRNNmodel.load_state_dict(checkpoint['net'])
            else:
                minacc = np.array(valiacc_list)[-1]
                ## save parameter
                state = {'net': CRNNmodel.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'minacc': minacc}
                torch.save(state, dir)
            if optimizer.param_groups[0]['lr'] < 0.0001:
                break
        print(dir)
        checkpoint = torch.load(dir)
        CRNNmodel.load_state_dict(checkpoint['net'])
        datasetname1=['testdata','CB513data','CASP10data','CASP11data','CASP12data','CASP13data','CASP14data']
        acc={}
        for name in datasetname1:
            acc[name]=valiModel(datasetloader[name],name)

        savemode=True
        if savemode:
            datasetname1 = ['testdata', 'CB513data', 'CASP10data', 'CASP11data', 'CASP12data', 'CASP13data',
                            'CASP14data']
            acc = {}
            for name in datasetname1:
                acc[name] = savefeature(datasetloader[name], name)
            acc1=str(acc)
            from datetime import datetime
            # 获取当前日期
            current_date = datetime.now().date()
            # 打开文件并写入字符串, "a"表示在现有内容下面增加。“w”表示覆盖源文件
            with open("output.txt", "a") as file:
                file.write(f'{current_date}: {trainmode}'+'\n')
                file.write(acc1+'\n')
