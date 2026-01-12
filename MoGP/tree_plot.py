import graphviz
from deap import gp


def plottree(individual,name=None,view=True,dir=None):
    # 创建节点和边的列表
    nodes, edges, labels = gp.graph(individual)

    # 创建Graphviz图形对象
    g = graphviz.Digraph(format='png')

    # 添加节点
    for node in nodes:
        g.node(str(node), label=str(labels[node]))

    # 添加边
    for edge in edges:
        g.edge(str(edge[0]), str(edge[1]))

    # 渲染并保存图形
    if name != None:
        if dir==None:
            g.render(filename=name, directory='./figures', cleanup=True)
        else:
            g.render(filename=name, directory=dir, cleanup=True)
    if view==True:
        g.view()
