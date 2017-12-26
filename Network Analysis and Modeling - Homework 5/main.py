
### import libraries
import networkx as nx
from copy import deepcopy
from numpy import random
import numpy as np
from random import shuffle, choice
import itertools
import operator
import matplotlib.pyplot as plt
from scipy.integrate import simps
from collections import Counter

### helper functions
def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

def most_common(L):
  SL = sorted((x, i) for i, x in enumerate(L))
  groups = itertools.groupby(SL, key=operator.itemgetter(0))
  def _auxfun(g):
    item, iterable = g
    count = 0
    min_index = len(L)
    for _, where in iterable:
      count += 1
      min_index = min(min_index, where)
    return count, -min_index
  return max(groups, key=_auxfun)[0]

def collect_labels(G,label):
    labels = []
    for node in G.nodes():
        labels.append(G.node[node][label])
    return labels

def find_min_nonetype_nieghbor(G,label, missing_nodes):

    max = -1
    best_choice = None

    for node in missing_nodes:
        surrounding_nodes = G.neighbors(node)
        number_of_known = len([1 for neighbors in surrounding_nodes if G.node[neighbors][label] != -1])
        if number_of_known > max:
            max = number_of_known
            best_choice = node

    return best_choice

def GbA_heuristic_naive(trueG, p, label):

    # create graph with missing labels
    parsedG = deepcopy(trueG)
    for node in parsedG.nodes():
        # simulate whether node has missing label
        if random.uniform(0,1) < p:
            # replace label with null qualifier
            parsedG.node[node][label] = "None"

    # run GbA heuristic over graph with missing edges
    not_done = 1
    old_nodes = []
    while not_done:

        # iterate through nodes and add labels
        nodes_ = parsedG.nodes()
        nodes_ = [node for node in nodes_ if parsedG.node[node][label] == "None"]
        shuffle(nodes_)

        for node in nodes_:

            # find mode of label of neigbors
            surrounding_nodes = parsedG.neighbors(node)
            surrounding_labels = []
            for node_n in surrounding_nodes:
                surrounding_labels.append(parsedG.node[node_n][label])

            # remove nonetype
            surrounding_labels[:] = [x for x in surrounding_labels if x != "None"]

            # if no neighbors have label, reshuffle and start again
            if surrounding_labels:
                # find mode of labels and set current node to it
                nx.set_node_attributes(parsedG, label, {node: most_common(surrounding_labels)})
            else:
                break

        # check if any nodes have none-type as label any longer
        remaining_nodes = []
        for node in nodes_:
            remaining_nodes.append(parsedG.node[node][label])
        if "None" not in remaining_nodes:
            not_done = 0

        # if graph is not fully connected
        if not nx.is_connected(parsedG):
            # check if the nodes havent changed (island of un-labled nodes)
            if nodes_ == old_nodes:
                # pick total mode
                labels = [parsedG.node[node][label] for node in nodes_ if parsedG.node[node][label] == "None"]
                c = Counter(labels)
                for node_ in nodes_:
                    nx.set_node_attributes(parsedG, label, {node_: c.most_common()[0]})

        old_nodes = nodes_

    # calculate accuracy of heuristic
    correct = 0.
    for node in parsedG.nodes():
        if parsedG.node[node][label] == trueG.node[node][label]:
            correct += 1

    return correct/len(parsedG.nodes())

def GbA_heuristic(trueG, percent_hidden, label):

    # create graph with missing labels
    parsedG = deepcopy(trueG)
    testing_nodes = []

    for node in parsedG.nodes():
        # simulate whether node has missing label
        if random.uniform(0,1) > percent_hidden:
            # replace label with null qualifier
            testing_nodes.append(node)
            parsedG.node[node][label] = -1

    # check if graph is fully connected

    if not nx.is_connected(parsedG):

        # check if the nodes havent changed (island of un-labled nodes)
        sub_graphs = list(sorted(nx.connected_components(parsedG), key = len, reverse=True))

        # find highest degree node and connext a random node from the subgrach to it
        # using centrality
        max_degree_node = sorted(nx.degree_centrality(parsedG).items(), key=operator.itemgetter(1))
        # max_degree_node = sorted(nx.betweenness_centrality(parsedG).items(), key=operator.itemgetter(1))
        # max_degree_node = sorted(nx.eigen(parsedG).items(), key=operator.itemgetter(1))
        # assign best candidate
        max_degree_node = max_degree_node[-1][0]
        # add connection to this node if any of the subgraphs are all None
        for graph in sub_graphs:
            nodes = list(graph)
            if len([parsedG.node[node][label] for node in nodes if parsedG.node[node][label] != -1]) == 0:
                parsedG.add_edge(max_degree_node,choice(nodes))


    # run GbA heuristic over graph with missing edges
    not_done = 1

    while not_done:

        # set current nodes to impute
        missing_nodes = [x for x in parsedG.nodes() if parsedG.node[x][label] == -1]
        observed_labels = [x for x in parsedG.nodes() if parsedG.node[x][label] != -1]

        # check if the graph does not have any labels at all
        if len(observed_labels) == 0:
            lucky_node = choice(parsedG.nodes())
            new_label = choice([trueG.node[node_][label] for node_ in trueG.nodes()])
            parsedG.node[lucky_node][label] = new_label

        # otherwise
        elif len(missing_nodes) > 0:

            # pick node with fewest unknown neighbors
            node = find_min_nonetype_nieghbor(parsedG, label, missing_nodes)

            # don't include non-types in consideration
            surrounding_labels = [int(parsedG.node[x][label]) for x in parsedG.neighbors(node) if parsedG.node[x][label] != -1]

            # find mode of labels and set current node to it
            if len(surrounding_labels) == 1:
                parsedG.node[node][label] = surrounding_labels[0]
            elif len(surrounding_labels) > 1:
                parsedG.node[node][label] = most_common(surrounding_labels)

        # if there are no remaining un-labeled nodes
        else:
            not_done = 0

    # calculate accuracy of heuristic
    correct = 0.
    for node in testing_nodes:
        if parsedG.node[node][label] == trueG.node[node][label]:
            correct += 1
    if testing_nodes:
        return correct/len(testing_nodes)
    else:
        return 1

def sf_degree_product(node_1, node_2, graph):
    return graph.degree(node_1)*graph.degree(node_2)

def sf_normalized_common_neighbors(node_1, node_2, graph):
    gamma_intersect = list(nx.common_neighbors(graph, node_1, node_2))
    gamma_union = []
    if graph.neighbors(node_1):
        gamma_union.append(graph.neighbors(node_1))
    if graph.neighbors(node_2):
        gamma_union.append(graph.neighbors(node_2))
    if gamma_union:
        gamma_union = [item for sublist in gamma_union for item in sublist]
        return float(len(gamma_intersect) / len(list(set(gamma_union))))
    else:
        return 0

def sf_shortest_path(node_1, node_2, graph):
    # check that path exists
    if nx.has_path(graph, node_1, node_2):
        # return shortest path length
        return float(1/nx.shortest_path_length(graph, source=node_1, target=node_2))
    else:
        return 0

def calc_true_posatives(proposed_edges, actual_edges):
    return float(len(set(proposed_edges).intersection(actual_edges)))

def calc_false_posatives(proposed_edges, actual_missing_edges):
    return float(len(set(proposed_edges).intersection(actual_missing_edges)))

def calc_false_negatives(proposed_non_edges, actual_edges):
    return float(len(set(proposed_non_edges).intersection(actual_edges)))

def calc_true_negatives(proposed_non_edges, actual_missing_edges):
    return float(len(set(proposed_non_edges).intersection(actual_missing_edges)))

def AUC_info(score_data, actual_edges, actual_missing_edges, resolution):

    ROC_data = []

    # iterate through all thresholds
    val = len(score_data)
    for threshold in range(1, val, resolution):
        print('current threshold: ', threshold/val)
        # add edges to first l node pairs in data
        proposed_edges = score_data[:threshold]
        proposed_non_edges = score_data[threshold+1:]

        true_positives = calc_true_posatives(proposed_edges, actual_edges)
        false_positives = threshold - true_positives
        true_negatives = calc_true_negatives(proposed_non_edges, actual_missing_edges)
        false_negatives = len(proposed_non_edges) - true_negatives
        sensitivity = true_positives/(true_positives + false_negatives)
        specificity = true_negatives/(true_negatives + false_positives)

        # create ROC data
        ROC_data.append((sensitivity,1-specificity))

    # integrate over each of the roc plots
    x = [data_point[1] for data_point in ROC_data]
    y = [data_point[0] for data_point in ROC_data]

    # return thresholds with AUC values
    return simps(y,x)

def Edge_imputation(trueG, p, scoreingfxn):

    # create graph with missing edges
    parsedG = deepcopy(trueG)
    unobserved_edges = []
    for edge in trueG.edges():
        # simulate whether edge will be observed
        if random.uniform(0, 1) > p:
            parsedG.remove_edge(edge[0], edge[1])
            unobserved_edges.append(edge)

    # iterate through all possible edges and score them
    nodes = parsedG.nodes()
    scores = []
    true_data = []
    for index_1 in range(len(nodes)-1):
        for index_2 in range(index_1+1, len(nodes)):
            node_1 = nodes[index_1]
            node_2 = nodes[index_2]

            if not parsedG.has_edge(node_1, node_2):
                score = scoreingfxn(node_1, node_2, parsedG)
                scores.append((node_1,node_2,score))
                true_data.append((node_1,node_2,trueG.has_edge(node_1,node_2)))

    # sort these scores
    scores.sort(key=lambda x: -x[2])
    score_data = [(scores[ii][0], scores[ii][1]) for ii in range(len(scores))]
    actual_edges = [(true_data[ii][0], true_data[ii][1]) for ii in range(len(score_data)) if true_data[ii][2] == 1]
    actual_missing_edges = [(true_data[ii][0], true_data[ii][1]) for ii in range(len(score_data)) if true_data[ii][2] != 1]
    # run AUC calculation
    if actual_edges:
        return AUC_info(score_data, actual_edges, actual_missing_edges, 10000)
    else:
        return 1

load_data = 1
check_algorithm1 = 0
check_algorithm2 = 0
part_a = 0
part_b = 1

### create graphs
G_1 = nx.Graph()
G_2 = nx.Graph()

if load_data:
    ### initialize datasets
    file_1 = []
    file_2 = []
    file_3 = []
    file_4 = []

    with open("/Users/wilder/PycharmProjects/HW5_NAM/data_people.txt") as f:
        next(f)
        for line in f:
            file_1.append(line.replace("\n", ""))

    with open("/Users/wilder/PycharmProjects/HW5_NAM/net1m_2011-08-01.txt") as f:
        for line in f:
            file_2.append(line.replace("\n", ""))

    with open("/Users/wilder/PycharmProjects/HW5_NAM/HVR_5.txt") as f:
        for line in f:
            file_3.append(line.replace("\n", ""))

    with open("/Users/wilder/PycharmProjects/HW5_NAM/metadata_CysPoLV.txt") as f:
        for line in f:
            file_4.append(line.replace("\n", ""))

    nodes1 = []
    for line in file_2:
        current = line.split(" ")
        nodes1.append(current[0])
        nodes1.append(current[1])
    nodes1 = list(set(nodes1))

    for line in file_1:
        name_ = find_between(line, '"', '"' )
        current = line.replace(name_,'')
        current = current.replace('"', '')
        current = current.split(" ")
        if current[0] in nodes1:
            G_1.add_node(int(current[0]), gender=int(current[2]))

    for line in file_2:
        current = line.split(" ")
        G_1.add_edge(int(current[0]), int(current[1]))

    counter = 0
    for line in file_4:
        counter+=1
        G_2.add_node(counter,name=int(line))

    for line in file_3:
        current = line.split("\t")
        G_2.add_edge(int(current[0]), int(current[1]))

if check_algorithm1:
    ### run GbA heuristic
    percent_hidden = 0.5
    label = "gender"
    print(GbA_heuristic(G_1,percent_hidden,label))

    percent_hidden = 0.5
    label = "name"
    print(GbA_heuristic(G_2,percent_hidden,label))

if part_a:
    ### plot GbA heuristic
    plot1 = []
    plot2 = []
    probability = np.linspace(0.01, 0.99, num=100, endpoint=True, retstep=False, dtype=None)
    av = 1
    for percent_hidden in probability:
        plot1.append(sum([GbA_heuristic(G_1, percent_hidden, "gender") for ii in range(av)])/av)
        plot2.append(sum([GbA_heuristic(G_2, percent_hidden, "name") for ii in range(av)])/av)

    plt.plot(probability, plot1, label = "condition")
    plt.xlabel('percent observed')
    plt.ylabel('percent correct')
    plt.title('GbA Heuristic: Malaria Network')
    plt.legend()
    plt.grid(True)
    plt.savefig("test1a.png")
    plt.show()

    plt.plot(probability, plot2, label = "gender")
    plt.xlabel('percent observed')
    plt.ylabel('percent correct')
    plt.title('GbA Heuristic: Board of Directors')
    plt.legend()
    plt.grid(True)
    plt.savefig("test2a.png")
    plt.show()

if check_algorithm2:
    #print(Edge_imputation(G_2, .01, sf_normalized_common_neighbors))
    print(Edge_imputation(G_2, .01, sf_degree_product))
    print(Edge_imputation(G_2, .3, sf_shortest_path))
    print(Edge_imputation(G_1, .3, sf_normalized_common_neighbors))
    print(Edge_imputation(G_1, .3, sf_degree_product))
    print(Edge_imputation(G_1, .3, sf_shortest_path))

if part_b:
    # impute edges using the three scoring functions over different amounts of missing edges
    plot1a = []; plot1b = []; plot1c = []
    plot2a = []; plot2b = []; plot2c = []
    probability = np.linspace(0.01, .99, num=10, endpoint=True, retstep=False, dtype=None)
    av = 1
    for p in probability:
        # over the first graph
        print('probability: ', p)
        # plot1a.append(sum([Edge_imputation(G_2, p, sf_normalized_common_neighbors) for ii in range(av)])/av)
        # plot1b.append(sum([Edge_imputation(G_2, p, sf_degree_product)for ii in range(av)])/av)
        # plot1c.append(sum([Edge_imputation(G_2, p, sf_shortest_path)for ii in range(av)])/av)
        # over the second graph
        plot2a.append(sum([Edge_imputation(G_1, p, sf_normalized_common_neighbors) for ii in range(av)])/av)
        plot2b.append(sum([Edge_imputation(G_1, p, sf_degree_product)for ii in range(av)])/av)
        plot2c.append(sum([Edge_imputation(G_1, p, sf_shortest_path)for ii in range(av)])/av)

    # sf_normalized_common_neighbors = plt.plot(probability, plot1a, label ="normalized common neighbors")
    # sf_degree_product = plt.plot(probability, plot1b, label ="degree product")
    # sf_shortest_path = plt.plot(probability, plot1c, label ="shortest path")
    # plt.legend()
    # plt.xlabel('percentage of observed edges')
    # plt.ylabel('percent of correctly imputed edges')
    # plt.title('Edge Imputation: Malaria Network')
    # plt.grid(True)
    # plt.savefig("test1b.png")
    # plt.show()

    sf_normalized_common_neighbors = plt.plot(probability, plot2a, label ="normalized common neighbors")
    sf_degree_product = plt.plot(probability, plot2b,  label ="degree product")
    sf_shortest_path = plt.plot(probability, plot2c, label ="shortest path")
    plt.legend()
    plt.xlabel('percentage of observed edges')
    plt.ylabel('percent of correctly imputed edges')
    plt.title('Edge Imputation: Board of Directors')
    plt.grid(True)
    plt.savefig("test2b.png")
    plt.show()