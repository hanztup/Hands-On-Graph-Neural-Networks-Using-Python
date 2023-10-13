# coding:utf-8

'''
date: 2023/10/13
content: implement biased random walk in Node2Vec.
'''

import os
import random
import numpy as np
import networkx as nx

# set random seed
random.seed(0)
np.random.seed(0)


def select_next_node(G, previous, current, p, q):
    '''
    根据前一节点(previous)、当前节点(current)、以及参数p/q来随机选择下一个节点
    ----
    - Inputs:
      - G: networkx graph, whole graph
      - previous: 
      - current:
      - p:
      - q:
    - Outputs:
    '''

    # 1.找到当前节点的邻居列表
    neighbors = list(G.neighbors(current))

    # 2.计算每个邻居节点neighbor的转移概率（非归一化）
    #   - neighbor为previous
    #   - neighbor与previous直接相连
    #   - neighbor与previous不相连
    alphas = []
    for neighbor in neighbors:
        if neighbor == previous:
            alpha = 1 / p
        elif G.has_edge(neighbor, previous):
            alpha = 1
        else:
            alpha = 1 / q
        alphas.append(alpha)
    
    # 3.计算current到neighbor的归一化转移概率
    probs = [alpha / sum(alphas) for alpha in alphas]

    # 4.从上述邻居节点中随机选择一个节点
    next_node = np.random.choice(neighbors, size=1, p=probs)[0]

    return next_node


def generate_random_walk(G, start, length, p, q):
    '''
    biased random walk
    ----
    - Inputs:
      - G:
      - start:
      - length:
      - p:
      - q:
    '''

    walk = [start]

    for i in range(length):
        current = walk[-1]
        previous = walk[-2] if len(walk) >= 2 else None
        next_node = select_next_node(G=G, previous=previous, current=current, p=p, q=q)
        walk.append(next_node)

    return walk


def main():

    # 1. 加载graph
    G = nx.erdos_renyi_graph(10, 0.3, seed=1, directed=False)
    # G = nx.gnp_random_graph(10, 0.3, seed=1, directed=False)  # the same as erdos_renyi_graph function
    # print(G)
    # print(G.nodes)
    # print(G.edges)

    # 2. 生成random walks
    a = generate_random_walk(G, 0, 8, p=1, q=1)
    b = generate_random_walk(G, 0, 8, p=1, q=10)
    c = generate_random_walk(G, 0, 8, p=10, q=1)
    print("p==1 and q==1: {}".format(a))
    print("p==1 and q==10: {}".format(b))
    print("p==10 and q==1: {}".format(c))




if __name__ == "__main__":
    main()



"""
record points:
1. nx.erdos_renyi_graph是什么, 一般怎么使用?
用来生成随机图，等价于nx.gnp_random_graph，官方文档（https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.erdos_renyi_graph.html#networkx.generators.random_graphs.erdos_renyi_graph）

2.
"""