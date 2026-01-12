
from transformers import T5Tokenizer, T5Model
import torch
import numpy as np
import pickle
USE_GPU=True
def create_tensor(tensor):
    if USE_GPU:
        device = torch.device("cuda:0")
        tensor = tensor.to(device)
    return tensor

model = T5Model.from_pretrained("./prot_t5_xl_uniref50")

name=['traindata','validata','testdata','CASP10data','CB513data','CASP11data','CASP12data','CASP13data','CASP14data']
for jdx in range(len(name)):
    ff = f'./DataSet/{name[jdx]}0.npy'
    data0=np.load(ff)
    data1=torch.LongTensor(data0)
    datalen = data1.shape[0]
    seqlen=np.int32(data0[:,:,1].sum(-1)-1)
    step = 2
    dataLM = []
    for idx in range(int(np.ceil(datalen / step))):
        print(idx)
        input_ids = create_tensor(data1[idx * step:(idx + 1) * step, :, 0])
        attention_mask = create_tensor(data1[idx * step:(idx + 1) * step, :, 1])
        if USE_GPU:
            device = torch.device("cuda:0")
            model.to(device)
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=input_ids)
            # embedding = model(input_ids=input_ids,attention_mask=attention_mask,decoder_input_ids=None)
        # For feature extraction we recommend to use the encoder embedding
        encoder_embedding = embedding[2].cpu().numpy().astype('float32')
        for ii in range(step):
            if idx*step+ii<len(seqlen):
                dataLM.append(encoder_embedding[ii,:seqlen[idx*step+ii],:])
    ff = f'./dataLM/{name[jdx]}LM.npy'
    with open(ff, 'wb') as file:
        pickle.dump(dataLM, file)




