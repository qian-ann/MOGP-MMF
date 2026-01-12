# Multi-objective Genetic Programming with Multi-view Multi-level Feature for Enhanced Protein Secondary Structure Prediction

1.	Databases
<br>Precomputed databases are available for download at:
<br>https://pan.baidu.com/s/1ZHwXBJZxjkDkEajSHraQXA?pwd=wd3y
<br>Unpack this archive into the <b>DataSet</b> directory. 

2.  Models preparation
<br>Pretrained models are available for download at:
<br>https://pan.baidu.com/s/16m2Q5v6hOmMlzGu0X-8tTw?pwd=xcm5
<br>Unpack this archive into the <b>parameter</b> directory. 


3.	Protein secondary structure prediction (PSSP)
<br>•	The codes in CRNN-CFR-code-MMF generate features for GP.
<br>•	The codes in MoGP generate the final results.

Usage examples:
In the <b>CRNN-CFR-code-MMF</b> directory, run the following command:
```
python3 Step0_DataGenerate.py 
python3 Step1_LMdatagenerate.py 
python3 Step1_saprot_data_generate.py 
python3 Step2_FeatureGenerate.py 
```

In the <b>MoGP</b> directory, run the following command:
```
python3 Step0_MGP_main.py 
python3 Step1_MGP_main_evaluate.py 
```

The results of PSSP are stored in the <b>Y_pred_and_Y_targ</b> directory of <b>MoGP</b>. 


