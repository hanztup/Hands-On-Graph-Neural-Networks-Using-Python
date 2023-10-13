# coding:utf-8

'''
date: 2023/10/13
content: implement vanilla node2vec model.
'''

import os
from tqdm import tqdm


import numpy as np
import networkx as nx

from gensim.models.word2vec import Word2Vec

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from biased_random_walk import generate_random_walk


def load_data():
    '''
    加载图数据集
    '''
    G = nx.karate_club_graph()
    
    # 将标签club（'Mr. Hi' or 'Officer'）转换为0/1
    labels = []
    for node in G.nodes:
        label = G.nodes[node]['club']
        if label == 'Mr. Hi':
            labels.append(0)
        else:
            labels.append(1)
    return G, labels


def generate_random_walks(G, walk_num, length, p, q):

    walks = []
    for node in tqdm(G.nodes):
        for _ in range(walk_num):
            walks.append(generate_random_walk(G, node, length, p, q))
    
    return walks


def train_node2vec(random_walks, mode=1):
    
    if mode == 1:
        # 使用Hierarchical Softmax
        node2vec_model = Word2Vec(sentences=random_walks,
                                hs=1,  # hierarchical softmax
                                sg=1,  # skip gram
                                vector_size=100, 
                                window=10,
                                workers=2,
                                min_count=1, 
                                seed=0)
    else:
        # 使用negative sampling
        node2vec_model = Word2Vec(sentences=random_walks,
                            hs=0,  # negative sampliing
                            sg=1,  # skip gram
                            negative=10, # negative samples number
                            vector_size=100, 
                            window=10,
                            workers=2,
                            min_count=1, 
                            seed=0)
        
    
    node2vec_model.train(random_walks, 
                         total_examples=node2vec_model.corpus_count,
                         epochs=30,
                         report_delay=1)

    return node2vec_model


def train_and_test_classifier(node2vec_model, labels):
    
    # 划分训练、测试集
    train_mask = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
    test_mask = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33]
    labels = np.array(labels)

    # 分类器训练
    clf = RandomForestClassifier(random_state=0)
    clf.fit(node2vec_model.wv[train_mask], labels[train_mask])

    # 预测
    prediction = clf.predict(node2vec_model.wv[test_mask])
    acc = accuracy_score(labels[test_mask], prediction)

    print('Node2Vec accuracy: {}'.format(acc))


def main():

    # 1.加载图数据集
    G, labels = load_data()

    # 2.生成random walks
    walk_num = 80
    length = 10
    p, q = 3, 2
    random_walks = generate_random_walks(G, walk_num, length, p, q)

    # 3.训练node2vec
    node2vec_model = train_node2vec(random_walks, mode=0)

    # 4.训练randomforest分类器
    train_and_test_classifier(node2vec_model, labels)
    
    


if __name__ == '__main__':
    main()