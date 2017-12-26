
# import libraries
import matplotlib.pyplot as plt
import networkx as nx
import os
from copy import deepcopy
from sklearn.metrics.cluster import normalized_mutual_info_score

def draw_graph(G,title):

    #graph_pos = nx.spring_layout(G)
    graph_pos = nx.spring_layout(G)
    # draw nodes, edges and labels
    nx.draw_networkx_nodes(G, graph_pos, node_size=1000, node_color='blue', alpha=0.3)
    nx.draw_networkx_edges(G, graph_pos)
    nx.draw_networkx_labels(G, graph_pos, font_size=12, font_family='sans-serif')

    plt.title(title)
    plt.show()

def draw_graph_label(G,title):

    # draw nodes, edges and labels
    nx.draw(G, labels = nx.get_node_attributes(G, 'group'),  node_size=1000)
    plt.title(title)
    plt.show()

def e_rr(G, label):
    m = G.number_of_edges()

    attributes = nx.get_node_attributes(G, label)
    types = set(attributes.values())

    edge_count = {t: 0.0 for t in types}

    for edge in G.edges():
        if attributes[edge[0]] == attributes[edge[1]]:
            edge_count[attributes[edge[0]]] += 1

    e_rr = {}

    for key, value in edge_count.items():
        e_rr[key] = value / m

    return e_rr

def a_r(G,label):
    m = G.number_of_edges()

    attributes = nx.get_node_attributes(G, label)

    types = set(attributes.values())

    attr_count = {t: 0.0 for t in types}

    for edge in G.edges():
        attr_count[attributes[edge[0]]] += 1
        attr_count[attributes[edge[1]]] += 1

    a_r = {}

    for key, value in attr_count.items():
        a_r[key] = value / (2 * m)

    return a_r

def modularity_ef(G, label):

    # using definition from 1
    a = a_r(G, label)
    e = e_rr(G, label)

    attributes = nx.get_node_attributes(G, label)

    types = set(attributes.values())
    assort = 0.0

    for t in types:
        assort += (e[t] - (a[t] ** 2))

    return assort

def regen_new_grouping(G,groups):

    # initialize new graph
    G_new = nx.Graph()

    # add original edges
    edges = G.edges()
    G_new.add_edges_from(edges)

    # Add attributes from groups
    for idx, val in enumerate(groups):
        for node in val:
            G_new.add_node(node+1, group=idx)

    return G_new

def merge_groups(groups, i,j):
    group_test = deepcopy(groups)
    a, b = min(i, j), max(i, j)
    del group_test[b]
    del group_test[a]
    group_test.append(groups[i] + groups[j])
    return group_test

def greedy_algo(G):

    num_nodes = nx.number_of_nodes(G)
    groups = [[i] for i in range(num_nodes)]

    G_prev = regen_new_grouping(G, groups)
    Q_prev = modularity_ef(G_prev, 'group')

    plot_Q = []

    while len(groups) > 2:

        dQ = []
        plot_Q.append(Q_prev)

        for i in range(len(groups)):
            for j in range(len(groups)):
                if i != j:

                    groups_test = merge_groups(groups, i, j)
                    temp_G = regen_new_grouping(G, groups_test)
                    Q_new = modularity_ef(temp_G, 'group')
                    dQ.append((i, j, Q_new - Q_prev))

        merge_i, merge_j, max_dQ = sorted(dQ, key=lambda x: x[2], reverse=True)[0]

        if max_dQ > 0:
            #print(merge_i,merge_j)
            best_groups = merge_groups(groups, merge_i, merge_j)
            best_G = regen_new_grouping(G, groups)
            best_Q = modularity_ef(best_G, 'group')

        groups = merge_groups(groups, merge_i, merge_j)
        last_G = regen_new_grouping(G, groups)
        Q_prev = modularity_ef(last_G, 'group')

        plot_Q.append(Q_prev)

    return groups, best_groups, plot_Q, best_G, last_G, Q_prev, best_Q

def main():

    num3 = 0
    num4 = 1

    ##### 3
    if num3:

        # initialize graph
        G = nx.Graph()

        edge_path = "/Users/wilder/PycharmProjects/NAM_Homework_3/karate_edges_77.txt"
        attr_path = "/Users/wilder/PycharmProjects/NAM_Homework_3/karate_groups.txt"
        true_labeling = []

        # add edges as ints
        f = open(edge_path)
        edges = []
        for line in f.readlines():
            edge_pair = [int(v) for v in line.split("\t")]
            edges.append((edge_pair[0], edge_pair[1]))
        G.add_edges_from(edges)

        # Add attributes
        f = open(attr_path)
        #f.readline()  # skip the initial line

        for index, line in enumerate(f.readlines()):
            nodes = index + 1
            labeling = [int(i) for i in line.strip('\n').split('\t')]
            true_labeling.append(labeling[1])
            G.add_node(nodes, group=labeling[0])

        # call greedy algorithm for modularity maximimization on it

        groups, best_groups, plot_Q, best_G, last_G, Q_prev, best_Q = greedy_algo(G)

        # final modularity (after final merge)
        print("forced 2 group labeling: " + str(Q_prev))
        # best modularity
        print("maximal graph for group labeling: " + str(best_Q))

        # normalized_mutual_info_scores (forced to two groups and optimal)
        label_check_1 = [0 for ii in range(0, 34)]
        label_check_2 = [0 for ii in range(0, 34)]

        # forced labeling
        for jj in groups[0]:
            label_check_1[jj] = 2
        for jj in groups[1]:
            label_check_1[jj] = 1

        #print("True Labeling: " + str(true_labeling) + "(forced 2 group labeling)")
        #print("Generated Label Set: " + str(label_check_1) + "(forced 2 group labeling)")
        score = normalized_mutual_info_score(label_check_1, true_labeling)
        print("Normalized mutual info score: " + str(score) + " (forced 2 group labeling)")

        # forced labeling
        for jj in best_groups[0]:
            label_check_2[jj] = 3
        for jj in best_groups[1]:
            label_check_2[jj] = 2
        for jj in best_groups[2]:
            label_check_2[jj] = 1

        #print("True Labeling: " + str(true_labeling) + "(maximal graph for group labeling)")
        #print("Generated Label Set: " + str(label_check_2) + "(maximal graph for group labeling)")
        score = normalized_mutual_info_score(label_check_2, true_labeling)
        print("Normalized mutual info score: " + str(score) + " (maximal graph for group labeling)")
        print()

        # modularity maximization
        f, ax1 = plt.subplots()
        ax1.set_title("Modularity Maximization")
        ax1.plot(plot_Q)
        ax1.axhline(0, color='black',)
        ax1.set_ylabel('Modularity')
        ax1.set_xlabel('Number of Merges')
        #plt.show()

        # draw final graph
        #draw_graph(G, "Original Graph:")
        #draw_graph_label(last_G, "Maximization Graph: grouping (forced 2 group labeling)")
        #draw_graph_label(best_G, "Maximization Graph: grouping (maximal graph for group labeling)")

    ##### 4
    if num4:

        # find names of the edge list text files
        edge_lists = os.listdir("/Users/wilder/PycharmProjects/NAM_Homework_3/facebook100txt")

        # remove attribute files
        for string in edge_lists:
            if ("_attr" in string) or ("readme" in string) or ("pdf" in string) or (".txt" not in string):
                edge_lists.remove(string)

        # initialize the attributes you want to store
        Q_m = []; Q_d = []; Q_g = []; N = []; Q_s = []

        # now iterate through list of files and calculate relevent info
        for files in edge_lists:
            # break
            print(files)
            edge_path = "/Users/wilder/PycharmProjects/NAM_Homework_3/facebook100txt" + "/" + files
            attr_path = edge_path[:-4] + '_attr.txt'

            # initialize graph
            G = nx.Graph()

            # add edges as ints
            f = open(edge_path)
            edges = []
            for line in f.readlines():
                edge_pair = [int(v) for v in line.split("\t")]
                edges.append((edge_pair[0], edge_pair[1]))
            G.add_edges_from(edges)

            # Add attributes
            f = open(attr_path)
            f.readline()  # skip the initial line

            for index, line in enumerate(f.readlines()):
                nodes = index + 1
                labeling = [int(i) for i in line.strip('\n').split('\t')]
                G.add_node(nodes, status=labeling[0], gender=labeling[1], major=labeling[2], degree=G.degree(nodes))

            status_assort = modularity_ef(G, 'status')
            Q_s.append(status_assort)
            major_assort = modularity_ef(G, 'major')
            Q_m.append(major_assort)
            degree_assort = modularity_ef(G, 'degree')
            Q_d.append(degree_assort)
            gender_assort = modularity_ef(G, 'gender')
            Q_g.append(gender_assort)

            N.append(G.number_of_nodes())

        # now plot everything

        # gender
        # f1, ax1 = plt.subplots()
        # ax1.set_title("Modularity: Gender v Network Size")
        # ax1.scatter(N, Q_g)
        # ax1.axhline(0, color='black',)
        # ax1.set_ylabel('Modularity: Gender')
        # ax1.set_xlabel('Network Size, n')
        # ax1.set_xscale('log')
        # plt.show()
        # f2, ax2 = plt.subplots()
        # ax2.axvline(0, color='black')
        # ax2.hist(Q_g, 14, histtype='bar')
        # ax2.set_title('Gender Assortativity')
        # ax2.set_xlabel('Assortativity, Gender')
        # ax2.set_ylabel('Count, density')
        # plt.show()

        # # major
        # f3, ax1 = plt.subplots()
        # ax1.set_title("Modularity: Major v Network Size")
        # ax1.scatter(N, Q_m)
        # ax1.axhline(0, color='black',)
        # ax1.set_ylabel('Modularity: Major')
        # ax1.set_xlabel('Network Size, n')
        # ax1.set_xscale('log')
        # plt.show()
        # f4, ax2 = plt.subplots()
        # ax2.axvline(0, color='black')
        # ax2.hist(Q_m, 14, histtype='bar')
        # ax2.set_title('Major Assortativity')
        # ax2.set_xlabel('Assortativity, Major')
        # ax2.set_ylabel('Count, density')
        # plt.show()

        # status
        f5, ax1 = plt.subplots()
        ax1.set_title("Modularity: Status v Network Size")
        ax1.scatter(N, Q_s)
        ax1.axhline(0, color='black', )
        ax1.set_ylabel('Modularity: Status')
        ax1.set_xlabel('Network Size, n')
        ax1.set_xscale('log')
        plt.show()
        f6, ax2 = plt.subplots()
        ax2.axvline(0, color='black')
        ax2.hist(Q_s, 14, histtype='bar')
        ax2.set_title('Status Assortativity')
        ax2.set_xlabel('Assortativity, Status')
        ax2.set_ylabel('Count, density')
        plt.show()

        # degree
        f7, ax1 = plt.subplots()
        ax1.set_title("Modularity: Degree v Network Size")
        ax1.scatter(N, Q_d)
        ax1.axhline(0, color='black', )
        ax1.set_ylabel('Modularity: Degree')
        ax1.set_xlabel('Network Size, n')
        ax1.set_xscale('log')
        plt.show()
        f8, ax2 = plt.subplots()
        ax2.axvline(0, color='black')
        ax2.hist(Q_d, 14, histtype='bar')
        ax2.set_title('Degree Assortativity')
        ax2.set_xlabel('Assortativity, Degree')
        ax2.set_ylabel('Count, density')
        plt.show()


if __name__ == '__main__':
    main()
