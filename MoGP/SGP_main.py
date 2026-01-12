
# python packages
import random
import time
import SGP_eval as evalGP
from Fuse_dataload import dataloader
import numpy
from deap import tools
from SGP_init import init_GP
import pickle
from tree_plot import plottree


class Parameters:
    def __init__(self):
        self.datapara=10   #用于数据抽取，每隔xx个抽取一个。
        # self.dir=r'./FeatureGenerate/'
        self.dir=r'/home/lu/mount/ssd_test/project/第一篇论文-智能优化/CRNN-CFR-code/FeatureGenerate/'
        self.testsetname = ['validata', 'testdata', 'CB513data', 'CASP10data', 'CASP11data']
        self.USE_GPU=True
        self.FeatureLen=256+8
        self.Features=['HMM','PSSM','T5','Sa']
        self.Views=['CNN1','CNN2','RNN1','RNN2']
        # self.Views=['CNN2','RNN2']
        self.VF=[]
        for f in self.Features:
            for v in self.Views:
                self.VF.append(f+v)
        self.ProbName = ''
        for fe in self.Features:
            self.ProbName+=fe[0]
        self.QName = 'LQ8'  # 'Q8', 'Q3', 'Q8Q3'

class GPParameters:
    def __init__(self):
        self.population = 200 # 种群大小
        self.generation = 100   # 种群的最大迭代次数
        self.cxProb = 0.8   #交叉变异概率
        self.mutProb = 0.2  #突变变异概率
        self.elitismProb = 0.05  # 精英率
        self.initialMinDepth = 2   #初始最小深度
        self.initialMaxDepth = 6   #初始最大深度
        self.maxDepth = 8   #最大深度
        self.randomSeeds = 200
        # self.randomSeeds = random.randint(1,10000)

para = Parameters()
GPpara = GPParameters()



def GPMain(randomSeeds,toolbox,filename=None):
    random.seed(randomSeeds)
    pop = toolbox.population(GPpara.population)
    if filename!=None:
        with open(filename, 'rb') as file:
            pop1 = pickle.load(file)
        idx=-1
        for ind in pop1:
            if ind not in pop:
                pop[idx]=ind
                del pop[idx].fitness.values
                idx-=1
        del pop1
    # plottree(pop[5])
    hof = tools.HallOfFame(10) #保留 10 个最佳个体
    log = tools.Logbook() #记录和跟踪各种信息
    stats_fit1 = tools.Statistics(key=lambda ind: ind.fitness.values[0]) #根据个体的 fitness.values 提取适应度值
    # stats_fit2 = tools.Statistics(key=lambda ind: ind.fitness.values[1]) #根据个体的 fitness.values 提取适应度值
    stats_size_tree = tools.Statistics(key=len) #收集和统计个体表达式树的大小（树的节点数量)
    mstats = tools.MultiStatistics(fitness1=stats_fit1, size_tree=stats_size_tree) #用于一次性收集多个相关的进化信息，并可以用于生成综合的统计信息
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)
    # "gen"记录代数，“evals”表示适应度函数的评估次数
    log.header = ["gen", "evals"] + mstats.fields
    pop, log, hof, acc = evalGP.eaSimple(pop, toolbox, GPpara,
                               stats=mstats, halloffame=hof, verbose=True)
    return pop, log, hof, acc


def Test(pop):
    data_loader_test = dataloader(para, 'test')
    _, _, toolbox = init_GP(para, GPpara, data_loader_test.data, 'test')
    ObjValueSet, failureVectorSet, ObjValueNSet, ySet, Ensemble_result=toolbox.poptest(pop,para)
    return ObjValueSet, failureVectorSet, ObjValueNSet, ySet, Ensemble_result


if __name__ == "__main__":
    # trainmode=False
    trainmode=True
    if trainmode:
        beginTime = time.process_time()
        data_loader = dataloader(para, 'train')
        _, _, toolbox = init_GP(para, GPpara, data_loader.data)
        # old_filename = './GPResults/SGP_Q8_HPTS_0.7997.pkl'
        old_filename = None
        pop, log, hof, acc = GPMain(GPpara.randomSeeds,toolbox,old_filename)
        endTime = time.process_time()
        trainTime = endTime - beginTime

        filename = f"GPResults/SGP_{para.QName}_{para.ProbName}_{round(acc,4)}.pkl"
        with open(filename, 'wb') as file:
            pickle.dump(pop, file)
        print("数据已成功保存到文件中！")
        filename = f"GPResults/SGP_{para.QName}_{para.ProbName}_{round(acc,4)}_{GPpara.randomSeeds}.txt"
        # 假设 logbook 是已经记录好数据的对象
        logbook_str = str(log)
        # 将 logbook 的内容写入到 txt 文件中
        with open(filename, "w") as file:
            file.write(logbook_str)
        print("数据已成功保存到文件中！")

    # testmode=True
    testmode=False
    if testmode:
        filename = 'GPResults/SGP_Q8_HPTS_0.799.pkl'
        # filename = 'GPResults/SGP_Q8_HPTS_0.7978.pkl'
        # filename = f"GPResults/{para.QName}_{para.ProbName}_{round(acc,3)}.pkl"
        with open(filename, 'rb') as file:
            pop = pickle.load(file)
        # plottree(pop[-1],'pop1')
        # plottree(pop2[-1],'pop2')
        # [pop[idx].fitness.values for idx in range(0, len(pop))]
        dic_num={}
        for ind in pop:
            for item in ind:
                if item.name not in dic_num.keys():
                    dic_num[item.name]=1
                else:
                    dic_num[item.name] += 1
        [pop[idx].fitness.values for idx in range(len(pop))]
        ObjValueSet, failureVectorSet, ObjValueNSet, ySet, Ensemble_result = Test(pop)

        print(filename)
        print('Ensemble')
        print('Vali', Ensemble_result[1][0], 'Test', Ensemble_result[1][1],
              'CB513', Ensemble_result[1][2], 'CASP10', Ensemble_result[1][3], 'CASP11', Ensemble_result[1][4])
        # print('Bestc=',cc[len(cc)-2]['c_ViewVariable'].transpose(1,0))
        print('Vali', ObjValueNSet[0][-1], 'Test', ObjValueNSet[1][-1],
              'CB513', ObjValueNSet[2][-1], 'CASP10', ObjValueNSet[3][-1], 'CASP11', ObjValueNSet[4][-1])

        # filename='GAResults\AllViewNum_Q8Q3_bc_1.6507.pkl'
        # filename='GAResults\AllViewNum_Q8_bc_1.6497.pkl'
        # VF=cc[0]['para']['VF']
        # print(filename)
        # print('from long to short')
        # for idx in range(30):
        #     list=''
        #     num=int(cc[idx]['a_ViewVariable'][0])
        #     for jdx in range(num):
        #         name=VF[int(cc[idx]['a_ViewVariable'][num-jdx])]
        #         list=list+f', {name}'
        #     print(f'Individual{idx}', cc[idx]['a_ViewVariable'])
        #     print('Features:', list)
        #     print('Vali', ObjValueNSet[0][idx], 'Test', ObjValueNSet[1][idx],
        #           'CB513', ObjValueNSet[2][idx], 'CASP10', ObjValueNSet[3][idx], 'CASP11', ObjValueNSet[4][idx])
        #     print(' ')

        # filename='GAResults\Q8Q3_All_bc_1.5375.pkl'
        # # datasetname = ['validata', 'testdata', 'CB513data', 'CASP10data', 'CASP11data']
        # for idx in range(12):
        #     print(para.VF[idx])
        #     print('Vali  ', ObjValueNSet[0][idx])
        #     print('Test  ', ObjValueNSet[1][idx])
        #     print('CB513 ', ObjValueNSet[2][idx])
        #     print('CASP10 ', ObjValueNSet[3][idx])
        #     print('CASP11', ObjValueNSet[4][idx])

        # 对于基础几个方法的看前四个个体。0-3分别对应CB513,TS115,CASP12,Test。0-5对应前五个个体分别只使用了ADD,MUL,Max,Min,Tay
        # filename='GAResults\Q8Q3_All_bc_1.6335.pkl'
        # # datasetname = ['validata', 'testdata', 'CB513data', 'CASP10data', 'CASP11data']
        # print(filename)
        # for idx in range(5):
        #     print('#',para.ViewFuse[idx],'Q8','Q3')
        #     print('Vali  ', ObjValueNSet[0][idx])
        #     print('Test  ', ObjValueNSet[1][idx])
        #     print('CB513 ', ObjValueNSet[2][idx])
        #     print('CASP10 ', ObjValueNSet[3][idx])
        #     print('CASP11', ObjValueNSet[4][idx])
        #
        #
        # print('1')





