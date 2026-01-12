
# python packages
import datetime
import random
import time
import MGP_eval as evalGP
from Fuse_dataload import dataloader
from Fuse_functions import compare_funcs
import numpy
from deap import tools
from MGP_init import init_GP
import pickle
from tree_plot import plottree
import glob

class Parameters:
    def __init__(self):
        self.datapara=10   #用于数据抽取，每隔xx个抽取一个。
        # self.dir=r'./FeatureGenerate/'
        self.dir=r'/home/lu/mount/ssd_test/project/MOGP-MMF-投稿SMC/MOGP-MMF/CRNN-CFR-code-MMF/FeatureGenerate/'
        self.testsetname = ['validata', 'testdata', 'CB513data', 'CASP10data', 'CASP11data','CASP12data','CASP13data','CASP14data']
        self.USE_GPU=True
        self.FeatureLen=256+8
        # self.Views=['HMM','PSSM','T5','Sa']
        self.Views=['HMM']
        self.Features=['CNN1','CNN2','RNN1','RNN2']
        # self.Features=['CNN2','RNN2']
        self.VF=[]
        for f in self.Views:
            for v in self.Features:
                self.VF.append(f+v)
        self.ProbName = ''
        for fe in self.Views:
            self.ProbName+=fe[0]
        if len(self.Features)<4:
            for ve in self.Features:
                self.ProbName += ve[0]+ve[-1]
        self.QName = 'Q8'  # 'Q8', 'Q3', 'Q8Q3'

class GPParameters:
    def __init__(self):
        self.population = 200 # 种群大小
        self.generation = 50   # 种群的最大迭代次数
        self.cxProb = 0.5   #交叉变异概率
        self.mutProb = 0.5  #突变变异概率
        self.elitismProb = 0.02  # 精英率
        self.initialMinDepth = 2   #初始最小深度
        self.initialMaxDepth = 6   #初始最大深度
        self.maxDepth = 8   #最大深度
        self.randomSeeds = 200
        # self.randomSeeds = random.randint(1,10000)

para = Parameters()
GPpara = GPParameters()





def GPMain(randomSeeds,toolbox,filenames=[]):
    random.seed(randomSeeds)
    pop = toolbox.population(GPpara.population)
    pop0=[]
    if len(filenames)>0:
        for filename in filenames:
            with open(filename, 'rb') as file:
                pop1 = pickle.load(file)
            if 'SGP' in filename:
                pop1[-1].fitness = pop[1].fitness
            pop0.append(pop1[-1])
            del pop1
    # plottree(pop[5])
    hof = tools.HallOfFame(10) #保留 10 个最佳个体
    log = tools.Logbook() #记录和跟踪各种信息
    stats_fit1 = tools.Statistics(key=lambda ind: ind.fitness.values[0]) #根据个体的 fitness.values 提取适应度值
    stats_fit2 = tools.Statistics(key=lambda ind: ind.fitness.values[1]) #根据个体的 fitness.values 提取适应度值
    stats_size_tree = tools.Statistics(key=len) #收集和统计个体表达式树的大小（树的节点数量)
    mstats = tools.MultiStatistics(fitness1=stats_fit1, fitness2=stats_fit2, size_tree=stats_size_tree) #用于一次性收集多个相关的进化信息，并可以用于生成综合的统计信息
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)
    # "gen"记录代数，“evals”表示适应度函数的评估次数
    log.header = ["gen", "evals"] + mstats.fields
    pop, pop2, log, hof, acc = evalGP.eaSimple(pop, pop0, toolbox, GPpara,
                               stats=mstats, halloffame=hof, verbose=True)
    return pop, pop2, log, hof, acc


def Test(pop):
    data_loader_test = dataloader(para, 'test')
    _, _, toolbox = init_GP(para, GPpara, data_loader_test.data, 'test')
    ObjValueSet, failureVectorSet, ObjValueNSet, ySet, Ensemble_result=toolbox.poptest(pop,para)
    return ObjValueSet, failureVectorSet, ObjValueNSet, ySet, Ensemble_result


if __name__ == "__main__":
    trainmode=False
    # trainmode=True
    if trainmode:
        beginTime = time.process_time()
        data_loader = dataloader(para, 'train')
        pset, _, toolbox = init_GP(para, GPpara, data_loader.data)
        # old_filename = './GPResults_partail/Q8_HPTS_0.7992.pkl'
        old_filename = None
        pop, pop2, log, hof, acc = GPMain(GPpara.randomSeeds,toolbox,old_filename)
        endTime = time.process_time()
        trainTime = endTime - beginTime

        filename = f"GPResults_partail/{para.QName}_{para.ProbName}_{round(acc,4)}.pkl"
        with open(filename, 'wb') as file:
            pickle.dump(pop, file)
        # filename = f"GPResults_partail/{para.QName}_{para.ProbName}_{round(acc, 4)}_2.pkl"
        # with open(filename, 'wb') as file:
        #     pickle.dump(pop2, file)
        print("数据已成功保存到文件中！")
        filename = f"GPResults_partail/{para.QName}_{para.ProbName}_{round(acc,4)}_{GPpara.randomSeeds}.txt"
        # 假设 logbook 是已经记录好数据的对象
        logbook_str = str(log)
        # 将 logbook 的内容写入到 txt 文件中c
        with open(filename, "w") as file:
            file.write(logbook_str)
        print("数据已成功保存到文件中！")

    testmode=True
    # testmode=False
    if testmode:
        # names=['_H_','_P_','_T_','_S_','_HP_','_TS_','_HPS_','_HPTSC1_','_HPTSC2_','_HPTSR1_','_HPTSR2_',\
        #        '_HPTSC1C2_','_HPTSR1R2_','_HPTSC1R1_','_HPTSC2R2_']
        names=['_H_']
        for name in names:
            views=[]
            features=[]
            if 'H' in name:
                views.append('HMM')
            if 'P' in name:
                views.append('PSSM')
            if 'T' in name:
                views.append('T5')
            if 'S' in name:
                views.append('Sa')
            if 'C1' in name:
                features.append('CNN1')
            if 'C2' in name:
                features.append('CNN2')
            if 'R1' in name:
                features.append('RNN1')
            if 'R2' in name:
                features.append('RNN2')
            if len(features)==0:
                features=['CNN1','CNN2','RNN1','RNN2']
            VF = []
            for f in views:
                for v in features:
                    VF.append(f + v)
            para.yset = 0
            if name == '_H_':
                del para.yset
                para.acc='HMM'
            para.Views = views
            para.Features=features
            para.VF=VF
            file_list = glob.glob(f"./GPResults_partail/*{name}*.pkl")
            filename=file_list[0]
            with open(filename, 'rb') as file:
                pop = pickle.load(file)
            pop2=[]
            for ind in pop:
                if ind not in pop2:
                    pop2.append(ind)
            pop=pop2
            dic_num={}  #分析各个函数使用次数
            for ind in pop:
                for item in ind:
                    if item.name not in dic_num.keys() and item.name[:3]!='ARG':
                        dic_num[item.name]=1
                    elif item.name[:3]!='ARG':
                        dic_num[item.name] += 1

            ## [pop[idx].fitness.values for idx in range(len(pop))]
            ObjValueSet, failureVectorSet, ObjValueNSet, ySet, Ensemble_result = Test(pop)

            text=filename+str(datetime.datetime.now())+'\n'+\
                 str(dic_num)+'\n'
            for idx in range(1, len(pop) + 1):
                text = text + str(idx) + ':' + str(pop[-idx]) + '\n' + \
                       'Vali:' + str(ObjValueNSet[0][-idx]) + ', Test:' + str(ObjValueNSet[1][-idx]) + \
                       ', CB513:' + str(ObjValueNSet[2][-idx]) + ', CASP10:' + str(ObjValueNSet[3][-idx]) + \
                       ', CASP11' + str(ObjValueNSet[4][-idx]) + '\n'
                if idx == 1:
                    text_file = filename[:-4]+'_result_best.txt'
                    with open(text_file, "a") as file:
                        file.write(text + '\n\n')
            print(text)
            text_file = filename[:-4]+'_result.txt'
            with open(text_file, "a") as file:
                file.write(text+ '\n\n')




