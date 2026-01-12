import torch
import numpy as np
from SaProt.model.saprot.base import SaprotBaseModel
from transformers import EsmTokenizer
import pickle
config = {
    "task": "base",
    "config_path": "./SaProt/model/seqOnly",
    "load_pretrained": True,
}
model = SaprotBaseModel(**config)
tokenizer = EsmTokenizer.from_pretrained(config["config_path"])
device = "cuda"
model.to(device)
#
# #################### Example ####################

#
# seq = "M#E#V#Q#L#V#Q#Y#K#X#"
# tokens = tokenizer.tokenize(seq)
# print(tokens)
#
# inputs = tokenizer(seq, return_tensors="pt")
# inputs = {k: v.to(device) for k, v in inputs.items()}
#
# outputs = model(**inputs)
# print(outputs.logits.shape)


onehot = range(20,41)
#'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X'

def num_to_str(num,char_map):
    return ''.join([char_map[n] for n in num])

char_map=['A#', 'C#', 'D#', 'E#', 'F#', 'G#', 'H#', 'I#', 'K#', 'L#', 'M#',
          'N#', 'P#', 'Q#', 'R#', 'S#', 'T#', 'V#', 'W#', 'Y#', 'X#']


names=['traindata','validata','testdata','CASP10data','CB513data','CASP11data','CASP12data','CASP13data','CASP14data']


for namee in names:
    Traindata001 = np.load(fr'./DataSet/{namee}_sort.npy')

    Traindata001 = Traindata001[:,:,onehot]
    lengths=(Traindata001.sum(-1).sum(-1)).astype(int)
    Traindata001=Traindata001.argmax(-1)
    TraindataSa=[]
    for idx in range(Traindata001.shape[0]):
        prot = num_to_str(Traindata001[idx,:lengths[idx]],char_map)
        tokens = tokenizer.tokenize(prot)
        inputs = tokenizer(prot, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            # embeddings = model.get_hidden_states(inputs, reduction="mean")
            embeddings = model.get_hidden_states(inputs)
            # print(embeddings[0].shape)
        TraindataSa.append(embeddings[0].half().cpu().detach().numpy())
        print([namee, idx, range(Traindata001.shape[0])])
    ff = fr'./SatData/{namee}Sa.npy'
    with open(ff, 'wb') as file:
        pickle.dump(TraindataSa, file)
