
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


class Parameters:
    def __init__(self):
        self.datapara=10   #用于数据抽取，每隔xx个抽取一个。
        # self.dir=r'./FeatureGenerate/'
        self.dir=r'/home/lu/mount/ssd_test/project/MOGP-MMF-投稿SMC/MOGP-MMF/CRNN-CFR-code-MMF/FeatureGenerate/'
        self.testsetname = ['validata', 'testdata', 'CB513data', 'CASP10data', 'CASP11data','CASP12data','CASP13data','CASP14data']
        self.USE_GPU=True
        self.FeatureLen=256+8
        self.Views=['HMM','PSSM','T5','Sa']
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
        self.QName = 'LQ8'  # 'Q8', 'Q3', 'Q8Q3'

class GPParameters:
    def __init__(self):
        self.population = 200 # 种群大小
        self.generation = 100   # 种群的最大迭代次数
        self.cxProb = 0.5   #交叉变异概率
        self.mutProb = 0.5  #突变变异概率
        self.elitismProb = 0.02  # 精英率
        self.initialMinDepth = 2   #初始最小深度
        self.initialMaxDepth = 6   #初始最大深度
        self.maxDepth = 8   #最大深度
        self.randomSeeds = 10
        # self.randomSeeds = random.randint(1,10000)

para = Parameters()
GPpara = GPParameters()

def count_used_inputs(individual, pset):
    used_inputs = 0
    dic=pset.arguments
    dic0=[]
    dic1=[]
    for node in individual:
        # 如果节点是一个终端（即输入变量或常量），且是变量（例如 X1, X2, X3）
        if 'ARG' in node.name:
            if node.name not in dic1:
                dic1.append(node.name)
                used_inputs+=1
                idx = int(node.name[3:])
                if dic[idx][:2] not in dic0:
                    dic0.append(dic[idx][:2])
                    used_inputs += 10
    return round(used_inputs,2)



def GPMain(randomSeeds,toolbox,filenames=[]):
    random.seed(randomSeeds)
    pop = toolbox.population(GPpara.population)
    pop0=[]
    if len(filenames)>0:
        for filename in filenames:
            with open(filename, 'rb') as file:
                pop1 = pickle.load(file)
            for idx in range(1,100,10):
                if 'SGP' in filename:
                    pop1[-idx].fitness = pop[idx].fitness
                pop0.append(pop1[-idx])
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
        old_filename = []
        # old_filename = ['./GPResults/SGP_Q8_HPTS_0.8.pkl','./GPResults/SGP_Q8_HPTS_0.8001.pkl',
        #                 './GPResults/SGP_Q8_HPTS_0.8007.pkl','./GPResults/SGP_Q8_HPTS_0.8007_1.pkl',
        #                 './GPResults/SGP_NQ8_HPTS_0.8008.pkl']
        if len(old_filename)>0:
            para.QName='Prior_'+para.QName
        pop, pop2, log, hof, acc = GPMain(GPpara.randomSeeds,toolbox,old_filename)
        endTime = time.process_time()
        trainTime = endTime - beginTime

        filename = f"GPResults/{para.QName}_{para.ProbName}_{round(acc,4)}.pkl"
        with open(filename, 'wb') as file:
            pickle.dump(pop, file)
        filename = f"GPResults/{para.QName}_{para.ProbName}_{round(acc, 4)}_2.pkl"
        with open(filename, 'wb') as file:
            pickle.dump(pop2, file)
        print("数据已成功保存到文件中！")
        filename = f"GPResults/{para.QName}_{para.ProbName}_{round(acc,4)}_{GPpara.randomSeeds}.txt"
        # 假设 logbook 是已经记录好数据的对象
        logbook_str = str(log)
        # 将 logbook 的内容写入到 txt 文件中
        with open(filename, "w") as file:
            file.write(logbook_str)
        print("数据已成功保存到文件中！")

    testmode=True
    # testmode=False
    if testmode:
        # filename = './GPResults/LQ8_HPTS_0.799.pkl'
        # filename = './GPResults/Prior_Q8_HPTS_0.8012.pkl'
        filename = './GPResults/Prior_Q8_HPTS_0.8013.pkl'
        # filename = './GPResults/Prior_Q8_HPTS_0.8015.pkl'
        # filename = './GPResults_partail/Q8_T_0.7933.pkl'
        # filename = './GPResults/Q8_HPTS_0.7996.pkl'
        para.acc=filename[-8:-4]
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
        if 0:
            for idx in range(len(pop)):
                plottree(pop[-1-idx],f'pop{idx+1}_{round(pop[-1-idx].fitness.values[0],4)}_'
                                     f'{round(pop[-1-idx].fitness.values[1])}',view=False)

        data_loader = dataloader(para, 'train')
        pset, _, toolbox = init_GP(para, GPpara, data_loader.data)
        if 1:
            import matplotlib.pyplot as plt

            x = []
            y = []
            for idx in range(len(pop)):
                y.append(-pop[-1 - idx].fitness.values[0])
                x.append(count_used_inputs(pop[-1 - idx], pset))
                # 创建散点图
            plt.scatter(x, y, s=20, cmap='viridis', edgecolor='b')
            plt.xlabel('Fitness 2')
            plt.ylabel('Fitness 1 (Negative)')
            # 显示图表
            # plt.show()

        ## [pop[idx].fitness.values for idx in range(len(pop))]
        ObjValueSet, failureVectorSet, ObjValueNSet, ySet, Ensemble_result = Test(pop)

        text=filename+str(datetime.datetime.now())+'\n'+\
             str(dic_num)+'\n'+'Ensemble\n'+ \
             'Vali:'+str(Ensemble_result[1][0])+', Test:'+ str(Ensemble_result[1][1])+\
        ', CB513:'+ str(Ensemble_result[1][2])+ ', CASP10:'+ str(Ensemble_result[1][3])+\
             ', CASP11'+ str(Ensemble_result[1][4])+'\n'
        for idx in range(1,len(pop)+1):
             text=text+str(idx)+':'+str(pop[-idx])+'\n' + \
             'Vali:' +str(ObjValueNSet[0][-idx]) + ', Test:' + str(ObjValueNSet[1][-idx]) + \
             ', CB513:' + str(ObjValueNSet[2][-idx]) + ', CASP10:' + str(ObjValueNSet[3][-idx]) + \
             ', CASP11' + str(ObjValueNSet[4][-idx])+'\n'
        print(text)
        text_file = filename[:-4]+f'_result.txt'
        with open(text_file, "a") as file:
            file.write(text+ '\n\n')


        filename1 = f"GPResults/compare_exist.pkl"
        para.acc = filename1[-9:-4]
        # para.idx=0; para.acc='HMMCNN1'
        # para.idx=17; para.acc='ADD'

        with open(filename1, 'rb') as file:
            pop = pickle.load(file)
        ObjValueSet, failureVectorSet, ObjValueNSet, ySet, Ensemble_result = Test(pop)
        dic = pset.arguments
        dic=dic+['Concatation','Add','Mul','Max','Min','Concatation3']

        text='Exist\n'
        for idx in range(len(pop)):
             text=text+str(dic[idx])+':'+str(idx)+'\n' + \
             'Vali:' +str(ObjValueNSet[0][idx]) + ', Test:' + str(ObjValueNSet[1][idx]) + \
             ', CB513:' + str(ObjValueNSet[2][idx]) + ', CASP10:' + str(ObjValueNSet[3][idx]) + \
             ', CASP11' + str(ObjValueNSet[4][idx])+'\n'
        print(text)
        text_file = filename[:-4]+f'_result.txt'
        with open(text_file, "a") as file:
            file.write(text+ '\n\n')





    # compare_mode=True
    compare_mode = False
    if compare_mode:
        data_loader = dataloader(para, 'train')
        pset, _, toolbox = init_GP(para, GPpara, data_loader.data)
        pop2 = toolbox.population(GPpara.population)
        pop = []
        dic = pset.arguments
        dic = dic + ['Concatation', 'Add', 'Mul', 'Max', 'Min','Concatation3']
        for idx in range(len(dic)):
            pop2[idx].func = compare_funcs(dic[idx]).compare_funcs
            pop2[idx].name = dic[idx]
            pop.append(pop2[idx])
        GPpara.generation = 0
        hof = tools.HallOfFame(10)  # 保留 10 个最佳个体
        log = tools.Logbook()  # 记录和跟踪各种信息
        stats_fit1 = tools.Statistics(key=lambda ind: ind.fitness.values[0])  # 根据个体的 fitness.values 提取适应度值
        stats_fit2 = tools.Statistics(key=lambda ind: ind.fitness.values[1])  # 根据个体的 fitness.values 提取适应度值
        stats_size_tree = tools.Statistics(key=len)  # 收集和统计个体表达式树的大小（树的节点数量)
        mstats = tools.MultiStatistics(fitness1=stats_fit1, fitness2=stats_fit2,
                                       size_tree=stats_size_tree)  # 用于一次性收集多个相关的进化信息，并可以用于生成综合的统计信息
        mstats.register("avg", numpy.mean)
        mstats.register("std", numpy.std)
        mstats.register("min", numpy.min)
        mstats.register("max", numpy.max)
        # "gen"记录代数，“evals”表示适应度函数的评估次数
        log.header = ["gen", "evals"] + mstats.fields
        pop, pop2, log, hof, acc = evalGP.eaSimple(pop, toolbox, GPpara,
                                                   stats=mstats, halloffame=hof, verbose=True)
        filename = f"GPResults/compare_exist.pkl"
        with open(filename, 'wb') as file:
            pickle.dump(pop, file)

    compare_mode2=False
    # compare_mode2=True
    if compare_mode2:
        data_loader = dataloader(para, 'train')
        # para.yset = 1
        pset, _, toolbox = init_GP(para, GPpara, data_loader.data)
        filename1 = f"GPResults/compare_exist.pkl"
        with open(filename1, 'rb') as file:
            pop = pickle.load(file)
        ObjValueSet, failureVectorSet, ObjValueNSet, ySet, Ensemble_result = Test(pop)
        dic = pset.arguments
        dic = dic + ['Concatation', 'Add', 'Mul', 'Max', 'Min', 'Concatation3']

        text = 'Exist\n'
        for idx in range(len(pop)):
            text = text + str(dic[idx]) + ':' + str(idx) + '\n' + \
                   'Vali:' + str(ObjValueNSet[0][idx]) + ', Test:' + str(ObjValueNSet[1][idx]) + \
                   ', CB513:' + str(ObjValueNSet[2][idx]) + ', CASP10:' + str(ObjValueNSet[3][idx]) + \
                   ', CASP11' + str(ObjValueNSet[4][idx]) + '\n'
        print(text)
        text_file = filename[:-4] + f'_result.txt'
        with open(text_file, "a") as file:
            file.write(text + '\n\n')





