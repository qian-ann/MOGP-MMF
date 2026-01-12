import pickle
import numpy as np
from sovfun import sov_score

# file_name='y_pred_Set_799'
# file_name='y_pred_Set_8015'
file_name='y_pred_Set_8013'
with open(fr'Y_pred_and_Y_targ/{file_name}.pkl', 'rb') as f:
    y_pred_Set = pickle.load(f)

dir2 = f'/home/lu/mount/ssd_test/project/MOGP-MMF-投稿SMC/CRNN-CFR-code-MMF/DataSet/datalen.pkl'
with open(dir2, 'rb') as f:
    datalens = pickle.load(f)

# datasetname=['validata','testdata','CASP10data','CB513data','CASP11data','CASP12data','CASP13data','CASP14data']
datasetname=['testdata','CB513data','CASP10data','CASP11data','CASP12data','CASP13data','CASP14data']
name=['acc','precision','recall','F1','mcc','sov','Q8','G',
      'acc','precision','recall','F1','mcc','sov','Q8','H',
      'acc','precision','recall','F1','mcc','sov','Q8','I',
      'acc','precision','recall','F1','mcc','sov','Q8','B',
      'acc','precision','recall','F1','mcc','sov','Q8','E',
      'acc','precision','recall','F1','mcc','sov','Q8','S',
      'acc','precision','recall','F1','mcc','sov','Q8','T',
      'acc','precision','recall','F1','mcc','sov','Q8','C',
      'acc','precision','recall','F1','mcc','sov','Q8','ALL']

def calculate_mcc(tp, tn, fp, fn):
    numerator = tp * tn - fp * fn
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return numerator / denominator if denominator != 0 else 0
scoreset={}
scoreset['name']=name
accur={}
for namei in datasetname:
    y_preds = y_pred_Set[namei][:,0:8].argmax(axis=1)
    y_preds1 = y_pred_Set[namei][:,0:8]
    y_labels = y_pred_Set[namei][:,-1]


    sov=sov_score(y_preds,y_labels,datalens[namei].astype(int),return_per_state=False)
    error0 = y_preds - y_labels
    hitting = np.where(abs(error0) < 0.1, 1, 0)
    Q8 = (np.count_nonzero(hitting) / len(y_preds))
    scores=[]
    sumTP=0
    for idx in range(8):
        y_label = np.array(y_labels == idx).astype(float)
        y_pred = np.array(y_preds == idx).astype(float)
        TP=y_pred[y_label==1].sum()
        FN=(1-y_pred[y_label==1]).sum()
        FP=y_pred[y_label==0].sum()
        TN=(1-y_pred[y_label==0]).sum()
        acc=(TP+TN)/(TP+FN+FP+TN)
        precision = TP/(TP+FP+0.000001)
        recall = TP/(TP+FN+0.000001)
        F1 = 2*precision*recall/(precision+recall+0.000001)
        mcc = calculate_mcc(TP, TN, FP, FN)
        scores.append([acc,precision,recall,F1,mcc,0,0,TP+FN])
        sumTP+=TP
    num = np.array(scores)[:,-1].sum()
    dig=np.diag(np.array(scores)[:,-1])/num
    score=np.matmul(dig,np.array(scores)).sum(0)
    score[-1]=num
    score[-2]=Q8
    score[-3]=sov
    scores.append(score.tolist())
    scoreset[namei]=np.array(scores).reshape(-1)

    accur[namei]=[sumTP/num,num]



import pandas as pd

# 定义 CSV 文件路径
csv_file = fr'Y_pred_and_Y_targ/{file_name}.csv'
# 将字典保存为 CSV 文件
df = pd.DataFrame(scoreset)
df.to_csv(csv_file, index=False)