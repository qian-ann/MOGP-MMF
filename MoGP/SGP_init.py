from deap import base, creator, tools, gp
from Fuse_functions import *
from GP_functions import Fe1, Fe2, OFe1, Int1, Int2, Int3, Float1
import GP_functions as fe_fs
import functools
import random
import GP_restrict as gp_restrict
from deap.tools.emo import selNSGA2
import operator
import pickle

def count_used_inputs(individual, pset):
    used_inputs = 0
    dic=pset.arguments
    dic0=[]
    for node in individual:
        # 如果节点是一个终端（即输入变量或常量），且是变量（例如 X1, X2, X3）
        if 'ARG' in node.name:
            used_inputs+=1
            idx = int(node.name[3:])
            if dic[idx][:2] not in dic0:
                dic0.append(dic[idx][:2])
                used_inputs += 10
    return used_inputs


def init_GP(para,GPpara,data,mode='train'):
    new_names = para.VF
    input_types = []
    for fea in para.Features:
        for view in para.Views:
            input_types.append(Fe2)
    pset = gp.PrimitiveSetTyped('MAIN', input_types, OFe1)
    # 创建字典，将 ARG0, ARG1, ARG2 分别映射到新名称 xx, xx,  xx
    rename_dict = {f'ARG{i}': new_name for i, new_name in enumerate(new_names)}
    # 利用 renameArguments 批量重命名
    pset.renameArguments(**rename_dict)
    # feature concatenation
    pset.addPrimitive(fe_fs.root_con, [Fe2], OFe1, name='Root1')
    pset.addPrimitive(fe_fs.root_con, [Fe2, Fe2], OFe1, name='Root2')
    pset.addPrimitive(fe_fs.root_con, [Fe2, Fe2, Fe2], OFe1, name='Root3')
    # filtering
    pset.addPrimitive(fe_fs.mixconadd, [Fe2, Float1, Fe2, Float1], Fe2, name='W_Add')
    pset.addPrimitive(fe_fs.mixconsub, [Fe2, Float1, Fe2, Float1], Fe2, name='W_Sub')

    # con maxPooling
    pset.addPrimitive(fe_fs.root_con_maxP, [Fe2, Fe2], Fe2, name='MaxPF2')
    # # feature extraction
    pset.addPrimitive(fe_fs.mul, [Fe2, Fe2], Fe2, name='Mul')
    pset.addPrimitive(fe_fs.grt, [Fe2, Fe2], Fe2, name='GRT')  # 元素对比取大的值
    pset.addPrimitive(fe_fs.sqrt, [Fe2], Fe2, name='Sqrt')  # 绝对值开根号，再加上符号
    pset.addPrimitive(fe_fs.log, [Fe2], Fe2, name='Log')  # log函数
    pset.addPrimitive(fe_fs.exp, [Fe2], Fe2, name='Exp')  # 指数函数
    pset.addPrimitive(fe_fs.relu, [Fe2], Fe2, name='ReLU')
    pset.addPrimitive(fe_fs.gaussian_laplace, [Fe2, Int1], Fe2, name='LoGF')  # 高斯-拉普拉斯
    pset.addPrimitive(fe_fs.fft1, [Fe2, Int3], Fe2, name='FFT')  # 一维傅里叶变换

    # Terminals
    pset.addEphemeralConstant('Singma', functools.partial(random.randint, 1, 4), Int1)
    pset.addEphemeralConstant('KernelSize', functools.partial(random.randrange, 3, 8, 2), Int2)  # 生成3-8之间的数，间隔2，这里只能是3，5,7
    pset.addEphemeralConstant('FFTmode', functools.partial(random.randint, 1, 2), Int3)
    pset.addEphemeralConstant('n', functools.partial(random.randint, 1, 10), Float1)


    # 使用 creator.create 函数来创建一个名为 "FitnessMulti" 的自定义类，该类继承自 DEAP 库中的 base.Fitness 类。weight=1表示越大越好
    creator.create("Fitness", base.Fitness, weights=(1.0,))
    # 创建了一个名为 "Individual" 的自定义个体类，这个类将用于表示遗传编程中的个体
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness)
    # 为 Individual 类新增 weight 属性，默认为 None
    creator.Individual.output_weight_ = None
    creator.Individual.generation = 0
    creator.Individual.generation1 = 0

    toolbox = base.Toolbox()
    # 注册了一个名为 "expr" 的遗传编程个体生成函数。这个函数用于生成遗传编程中的表达式树（expression tree）或个体。
    # gp_restrict.genHalfAndHalfMD限制了长度不超过60
    toolbox.register("expr", gp_restrict.genHalfAndHalfMD, pset=pset, min_=GPpara.initialMinDepth,
                     max_=GPpara.initialMaxDepth)
    # 注册了一个名为 "individual" 的个体初始化函数，用于创建遗传编程中的个体
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    # 注册了一个名为 "population" 的种群初始化函数，用于创建一个个体的集合，也就是一个种群。
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # 注册了一个名为 "compile" 的编译函数，用于将遗传编程中的个体（表达式树）编译成可执行的函数或程序。
    toolbox.register("compile", gp.compile, pset=pset)
    # 注册了一个名为 "mapp" 的函数，它似乎是为了处理并行计算任务而注册的。
    toolbox.register("mapp", map)

    def evalTrain(individual):
        try:
            # 将个体编译成可运行的函数
            func = toolbox.compile(expr=individual)
            model = FuseModel(para)
            accuracy, failureVector, ObjValueN = model.FeatureFuse(data, individual, func)
        except:
            accuracy = 0
        return accuracy, failureVector, ObjValueN

    # genetic operator
    toolbox.register("evaluate", evalTrain)
    # 随机选N个个体，从中选出最优的
    toolbox.register("select", tools.selTournament, tournsize=5)
    # 从当前种群中选择适应度值最高的个体，以确保这些个体能够在下一代中继续存在
    toolbox.register("selectElitism", tools.selBest)
    # 交叉操作在遗传算法中用于产生新个体，通常是通过将两个父代个体的遗传信息进行组合来创建新的后代个体。
    # 点交叉（One-Point Crossover）操作,随机选择一个交叉点，并交换两个父代的子树。
    toolbox.register("mate", gp.cxOnePoint)
    # 突变操作在遗传算法中用于引入新的遗传变异，以帮助算法搜索更广的解空间。
    # 用于生成全树（Full Tree）结构的表达式树，该函数根据参数 min_ 和 max_ 来确定树的深度范围。
    toolbox.register("expr_mut", gp_restrict.genFull, min_=0, max_=2)
    # 均匀突变是一种突变策略，它会随机选择表达式树中的一个节点，并用一个随机生成的新节点替换它。
    # "expr_mut" 的函数，用于生成新节点
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    # 交叉操作 "mate" 被装饰以添加了一个树高度的限制。这意味着在交叉操作中生成的后代个体的树高度将受到限制，不会超过预定义的最大值 maxDepth
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=GPpara.maxDepth))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=GPpara.maxDepth))


    if mode!='train':
        def evalTest(population, para):
            torch.cuda.empty_cache()  # 清理未被引用的显存
            ObjValueSetPop = []
            failureVectorSetPop = []
            ObjValueNSetPop = []
            ySetPop = []
            hittingNSet = []
            accuracyNSet = []
            accuracySet = []
            y_pred_Set={}
            data1 = {}
            datasetname = para.testsetname
            model = FuseModel(para)
            for namei in datasetname:
                data1['Vinputs'] = data[namei]['Vinputs']
                data1['Vtarg'] = data[namei]['Vtarg']
                data1['VtargQ3'] = data[namei]['VtargQ3']
                ObjValueSet = []
                failureVectorSet = []
                ObjValueNSet = []
                ySet = []
                with torch.no_grad():
                    for ipop in range(len(population) - 1):
                        individual = population[ipop]
                        func = toolbox.compile(expr=individual)
                        y_pred, accuracy, failureVector, ObjValueN=model.FeatureFuse(data1, individual, func, mode='test')
                        ySet.append(y_pred)
                        ObjValueSet.append(accuracy)
                        failureVectorSet.append(failureVector)
                        ObjValueNSet.append(ObjValueN)
                        print(f"ind{ipop}")
                    ObjValueSetPop.append(ObjValueSet)
                    failureVectorSetPop.append(failureVectorSet)
                    ObjValueNSetPop.append(ObjValueNSet)
                    ySetPop.append(ySet)

                    y_pred = ySet[-1]
                    y_pred_Set[namei] = np.concatenate([y_pred.cpu().detach().numpy(),
                                                        data1['Vtarg'].unsqueeze(-1).cpu().detach().numpy()], -1)
                    # for yidx in range(len(ySet)-1):
                    for yidx in range(len(ySet) // 2 - 1):  # 集成
                        y_pred += ySet[-2 - yidx]
                    y_pred = y_pred / len(ySet)
                    accuracy, hitting, accuracyN = model.get_accuracy(y_pred, data1)
                    accuracySet.append(accuracy)
                    accuracyNSet.append(accuracyN)
                    hittingNSet.append(hitting)
                with open(r'Y_pred_and_Y_targ\y_pred_Set.pkl', 'wb') as f:
                    pickle.dump(y_pred_Set, f)
            return ObjValueSetPop, failureVectorSetPop, ObjValueNSetPop, ySetPop, [accuracySet, accuracyNSet, hittingNSet]
        toolbox.register("poptest", evalTest)

    return pset, creator, toolbox


