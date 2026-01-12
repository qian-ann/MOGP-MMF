import random
from deap import tools
from collections import defaultdict
import numpy
from tree_plot import plottree
import pickle

def pop_compare(ind1, ind2):
    # List all available primitive types in each individual
    types1 = defaultdict(list)
    types2 = defaultdict(list)
    for idx, node in enumerate(ind1[1:], 1):
        types1[node.ret].append(idx)
    for idx, node in enumerate(ind2[1:], 1):
        types2[node.ret].append(idx)
    return types1 == types2




def varAnd(population, pop0, mu, toolbox, cxProb, mutProb):  #执行变异和交叉
    offspring = [toolbox.clone(ind) for ind in population]
    new_cxProb = cxProb / (cxProb + mutProb)
    i = 0
    mu1=1
    mu2=1
    while i < len(offspring):
        if random.random() < new_cxProb:  #交叉变异，如果相同就对两者进行变异操作
            if random.random() < mu or len(pop0) == 0:
                ii = i + 1
                while ii < len(offspring) and (
                        (offspring[i] == offspring[ii]) or pop_compare(offspring[i], offspring[ii])):
                    ii += 1
                if ii == len(offspring):
                    offspring[i], = toolbox.mutate(offspring[i])
                else:
                    mu2 = mu2 + 1
                    offspring[i], _ = toolbox.mate(offspring[i], offspring[ii])
            else:
                offidx = random.randint(0, len(pop0) - 1)
                if (offspring[i] == pop0[offidx]) or pop_compare(offspring[i], pop0[offidx]):
                    offspring[i], = toolbox.mutate(offspring[i])
                else:
                    mu1 = mu1 + 1
                    offspring[i], _ = toolbox.mate(offspring[i], pop0[offidx])
            if offspring[i] == population[i]:
                offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values
            i = i + 1
        else:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values
            i = i + 1
    #   [offspring[idx]==population[idx] for idx in range(0,len(population))]
    return offspring, 1/2+mu1/(mu1+mu2)


def eaSimple(population, pop0, toolbox, GPpara, stats=None,
             halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    bestacc = 0
    pop_num = len(population)
    # Evaluate the individuals with an invalid fitness。
    # fitness.valid 通常是一个布尔值，表示个体的适应度是否已经被计算或评估。
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.mapp(toolbox.evaluate, invalid_ind)
    index=1
    for ind, fit in zip(invalid_ind, fitnesses):
        if numpy.mod(index,10)==0:
            print(index)
        index += 1
        ind.fitness.values = fit[0]  # 更新适应度函数
        if fit[0][0]>bestacc:
            bestacc=fit[0][0]
    # plottree(population[-1]) # [population[idx].fitness.values for idx in range(0, len(population))]
    halloffame.update(population)  # 将当前代的种群 population 中的个体与名人堂 halloffame 进行比较，并更新名人堂中的个体
    hof_store = tools.HallOfFame(5 * pop_num)  # 创建了一个新的名人堂 hof_store，其最大容量是当前代种群的个体数的 5 倍
    hof_store.update(population)  #更新名人堂，会移除重复的
    # 调用 stats.compile(population) 方法来编译种群 population 的统计信息。这将计算和汇总与种群相关的统计数据，并将其存储在 record 变量中。
    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=pop_num, **record)
    print(logbook.stream)
    pop2=[]
    for gen in range(1, GPpara.generation + 1):
        # Select the next generation individuals by elitism
        elitismNum = int(numpy.ceil(GPpara.elitismProb * pop_num))
        population_for_eli = [toolbox.clone(ind) for ind in population]
        offspringE = toolbox.selectElitism(population_for_eli, k=elitismNum)
        # plottree(offspringE[0])
        # Vary the pool of individuals
        # 接受父代个体 offspring、遗传算法工具箱 toolbox、交叉概率 GPpara.cxProb 和变异概率 GPpara.mutProb 作为参数。
        # 它将根据这些概率进行交叉和变异操作，生成新的后代个体，并将它们添加到 offspring 列表中。
        offspring = [toolbox.clone(ind) for ind in population]
        offspring = offspring_select(offspring,pop_num,random_num=5)
        if gen==1:
            mu=1/2
        offspring, mu = varAnd(offspring, pop0, mu, toolbox, GPpara.cxProb, GPpara.mutProb)
        # [offspring[i]==population[i] for i in range(0,len(population))]
        # add offspring from elitism into current offspring
        # generate the next generation individuals
        # Evaluate the individuals with an invalid fitness，从名人堂里面找计算过的适应度函数，多目标优化中下面循环取消
        for i in offspring:
            ind = 0
            while ind < len(hof_store):
                if i == hof_store[ind]:
                    i.fitness.values = hof_store[ind].fitness.values
                    i.output_weight_ = hof_store[ind].output_weight_
                    ind = len(hof_store)
                else:
                    ind += 1
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid] #确定还有未计算适应度函数的个体集合
        fitnesses = toolbox.mapp(toolbox.evaluate, invalid_ind)
        index=1
        for ind, fit in zip(invalid_ind, fitnesses):
            if numpy.mod(index, 20) == 0:
                print(index)
            index += 1
            ind.fitness.values = fit[0]
            ind.generation = gen
            if fit[0][0] > bestacc:
                bestacc = fit[0][0]
                # plottree(ind)

        # Update the hall of fame with the generated，保留10个最佳个体
        halloffame.update(offspring)
        # cop_po = offspring.copy()
        for ind in offspring:
            ind.generation1 = gen
        hof_store.update(offspring) #更新名人堂
        # 将原始种群 插入到后代个体 offspring 的开头（索引 0 的位置）
        offspring[0:0] = population
        offspring = pop_del_copy(offspring, GPpara.generation)  #去掉重复项
        # Select the next generation individuals
        population[:] = toolbox.select(offspring, pop_num-len(offspringE))
        # 将精英个体 offspringE 插入到后代个体 offspring 的开头（索引 0 的位置）
        population[0:0] = offspringE
        population=sort_pop(population)
        pop2.append(population[-1])
        pop2.append(population[-2])
        print(population[-1].fitness.values)
        # plottree(population[-1])
        # [population[idx].fitness.values for idx in range(0, len(population))]
        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=pop_num, **record)
        print(logbook.stream)
    # pop2 = pop_del_copy(hof_store,GPpara.generation)
    return population, pop2, logbook, halloffame, bestacc

def sort_pop(population):
    population_sort = [population[0]]
    for idx in range(1, len(population)):
        for jdx in range(0, len(population_sort)):
            if population[idx].fitness.values[0] > population_sort[jdx].fitness.values[0]:
                if jdx == len(population_sort) - 1 or \
                        population[idx].fitness.values[0] < population_sort[jdx + 1].fitness.values[0]:
                    population_sort[jdx + 1:jdx + 1] = [population[idx]]
                    break
            else:
                if jdx == 0 or \
                        population[idx].fitness.values[0] > population_sort[jdx - 1].fitness.values[0]:
                    population_sort[jdx:jdx] = [population[idx]]
                    break
    return population_sort

def pop_del_copy(pop,generation):
    pop2 = []
    if generation > 0:
        fit_list = []
        for ind in pop:
            if ind not in pop2 and ind.fitness.values not in fit_list:
                pop2.append(ind)
                fit_list.append(ind.fitness.values)
        # [pop2[idx].fitness for idx in range(len(pop2))]
        pop2 = sort_pop(pop2)
    return pop2

def offspring_select(pop,num,random_num=5):
    popnew=[]
    for idx in range(num):
        pop_sel=random.sample(pop,random_num)
        pop_sel_fit=numpy.array([popi.fitness.values[0] for popi in pop_sel])
        popnew.append(pop_sel[pop_sel_fit.argmax()])
    return popnew
