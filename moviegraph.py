import pandas as pd
import numpy as np
import math as math
import networkx as nx
import time 
from sklearn.metrics.pairwise import linear_kernel
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [14,14]

def find_similar(tfidf_matrix, index, top_n = 5):
    cosine_similarities = linear_kernel(tfidf_matrix[index:index+1], tfidf_matrix).flatten()
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]
    return [index for index in related_docs_indices][0:top_n]  

def generate_graph(moviesdf,tfidf):
    moviesdf['directors'] = moviesdf['director'].apply(lambda l: [] if pd.isna(l) else [i.strip() for i in l.split(",")])
    moviesdf['categories'] = moviesdf['listed_in'].apply(lambda l: [] if pd.isna(l) else [i.strip() for i in l.split(",")])
    moviesdf['actors'] = moviesdf['cast'].apply(lambda l: [] if pd.isna(l) else [i.strip() for i in l.split(",")])
    moviesdf['countries'] = moviesdf['country'].apply(lambda l: [] if pd.isna(l) else [i.strip() for i in l.split(",")])

    G = nx.Graph(label="MOVIE")
    start_time = time.time()
    for i, rowi in moviesdf.iterrows():
        if (i%1000==0):
            print(" iter {} -- {} seconds --".format(i,time.time() - start_time))
        G.add_node(rowi['title'],key=rowi['show_id'],label="MOVIE",mtype=rowi['type'],rating=rowi['rating'])
        for element in rowi['actors']:
            G.add_node(element,label="PERSON")
            G.add_edge(rowi['title'], element, label="ACTED_IN")
        for element in rowi['categories']:
            G.add_node(element,label="CAT")
            G.add_edge(rowi['title'], element, label="CAT_IN")
        for element in rowi['directors']:
            G.add_node(element,label="PERSON")
            G.add_edge(rowi['title'], element, label="DIRECTED")
        for element in rowi['countries']:
            G.add_node(element,label="COU")
            G.add_edge(rowi['title'], element, label="COU_IN")
        
        indices = find_similar(tfidf, i, top_n = 5)
        snode="Sim("+rowi['title'][:15].strip()+")"        
        G.add_node(snode,label="SIMILAR")
        G.add_edge(rowi['title'], snode, label="SIMILARITY")
        for element in indices:
            G.add_edge(snode, moviesdf['title'].loc[element], label="SIMILARITY")
    print(" finish -- {} seconds --".format(time.time() - start_time)) 
    return G

def get_all_adj_nodes(list_in, G):
    sub_graph=set()
    for m in list_in:
        sub_graph.add(m)
        for e in G.neighbors(m):        
                sub_graph.add(e)
    return list(sub_graph)
    
def draw_sub_graph(sub_graph, G):
    subgraph = G.subgraph(sub_graph)
    colors=[]
    for e in subgraph.nodes():
        if G.nodes[e]['label']=="MOVIE":
            colors.append('blue')
        elif G.nodes[e]['label']=="PERSON":
            colors.append('red')
        elif G.nodes[e]['label']=="CAT":
            colors.append('green')
        elif G.nodes[e]['label']=="COU":
            colors.append('yellow')
        elif G.nodes[e]['label']=="SIMILAR":
            colors.append('orange')    
        elif G.nodes[e]['label']=="CLUSTER":
            colors.append('orange')

    nx.draw(subgraph, with_labels=True, font_weight='bold',node_color=colors)
    plt.show()

def get_recommendation(root, G):
    commons_dict = {}
    for e in G.neighbors(root):
        for e2 in G.neighbors(e):
            if e2==root:
                continue
            if G.nodes[e2]['label']=="MOVIE":
                commons = commons_dict.get(e2)
                if commons==None:
                    commons_dict.update({e2 : [e]})
                else:
                    commons.append(e)
                    commons_dict.update({e2 : commons})
    movies=[]
    weight=[]
    for key, values in commons_dict.items():
        w=0.0
        for e in values:
            w=w+1/math.log(G.degree(e))  # Adamic Adar measure
        movies.append(key) 
        weight.append(w)
    
    result = pd.Series(data=np.array(weight),index=movies)
    result.sort_values(inplace=True,ascending=False)        
    return result;