
# network analysis and modeling homework 2

import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy
import operator
from random import shuffle, uniform
import numpy as np
# =================================================================================
# Graph vizualization using networkX
# =================================================================================

def draw_graph(G,title):

    #graph_pos = nx.spring_layout(G)
    graph_pos = nx.spring_layout(G)
    # draw nodes, edges and labels
    nx.draw_networkx_nodes(G, graph_pos, node_size=1000, node_color='blue', alpha=0.3)
    nx.draw_networkx_edges(G, graph_pos)
    nx.draw_networkx_labels(G, graph_pos, font_size=12, font_family='sans-serif')

    plt.title(title)
    plt.show()


# =================================================================================
# Configuration model as was described in class, as well as a seemingly better version
# =================================================================================

def perm_configuration_model(degree_dist):

    # generate vector of degrees
    degree_vector_ = [[jj for ii in range(0,degree_dist[jj])] for jj in range(0,len(degree_dist))]
    degree_vector = [item for sublist in degree_vector_ for item in sublist]

    # deep copy current simulation
    current_simulation = deepcopy(degree_vector)
    # shuffle current_simulation
    shuffle(current_simulation)

    # iteratively pull two vertices and then concatinate them to create graph
    incomplete = 1; counter = 0; edges = []
    max_iterations = 100000
    while incomplete:
        counter += 1
        # check if the first start end is not a self loop, or a multi-edge
        if current_simulation[0] != current_simulation[1] \
                and (current_simulation[0], current_simulation[1]) not in edges \
                and (current_simulation[1], current_simulation[0]) not in edges \
                and current_simulation:
            # if so add edge to edge list
            edges.append((current_simulation[0], current_simulation[1]))
            # remove first two elements chosen
            del current_simulation[0]
            del current_simulation[0]
            # check if you are out of edges
            if not current_simulation:
                incomplete = 0
        # if not, re-shuffle degree list and try again
        else:
            shuffle(current_simulation)
        if counter > max_iterations:
            incomplete = 0
    # add simulated graph
    G_ = nx.Graph()
    G_.add_nodes_from([ii for ii in range(0,len(degree_dist))])
    G_.add_edges_from(edges)

    return G_

def p_configuration_model(degree_dist):

    # generate vector of degrees
    degree_vector_ = [[jj for ii in range(0, degree_dist[jj])] for jj in range(0, len(degree_dist))]
    degree_vector = [item for sublist in degree_vector_ for item in sublist]

    # create list to store graphs in
    edges = []
    # simulate graphs using degree distribution and configuration model
    for node in degree_vector:
        for other_node in degree_vector:
            p_of_edge = (node*other_node)/(2*sum(degree_dist)-1)
            pass_val = uniform(0, 1)
            print(p_of_edge)
            print(pass_val)
            if p_of_edge > pass_val\
                and node != other_node \
                and (node, other_node) not in edges \
                and (other_node, node) not in edges:
                edges.append((node,other_node))
    G_ = nx.Graph()
    G_.add_edges_from(edges)
    return G_

# =================================================================================
# Home-brew centrality calculations
# =================================================================================

def hb_degree_centrality(graph):

    degree_cetrality = {}
    edge_list = graph.edges()

    vertices = graph.nodes()
    for nodes in vertices:
        degree = sum([1 if nodes in ii else 0 for ii in edge_list])
        degree_cetrality[nodes] = degree

    return degree_cetrality

def hb_harmonic_centrality(graph):

    vertices = graph.nodes()
    harmonic_centrality = {}
    n = len(vertices)

    for vertex_i in vertices:
        C_i = 0
        for vertex_j in vertices:
            if vertex_i != vertex_j:
                if nx.has_path(graph, vertex_i, vertex_j):
                    d_ij = nx.shortest_path_length(graph,vertex_i,vertex_j)
                    C_i = C_i + 1/d_ij
        C_i = C_i/(n-1)
        harmonic_centrality[vertex_i] = C_i

    return harmonic_centrality

def hb_eigen_vector_centrality(graph):

    adj = nx.to_numpy_matrix(graph)
    x_c = np.random.rand(graph.number_of_nodes(),1)
    Tol = .001
    max_it = 10000
    norm_dif = 1.
    iter = 0

    while norm_dif > Tol and iter < max_it:
        iter += 1
        norm_c = max(x_c)
        x_c = np.dot(adj,x_c)
        x_c = np.multiply(x_c,1/norm_c)
        norm_dif = abs(max(x_c) - norm_c)

    eigen_vector_centrality = {}
    for nodes in range(0,graph.number_of_nodes()):
        eigen_vector_centrality[nodes] = x_c[nodes]

    return eigen_vector_centrality

def hb_betweenness_centrality(graph):

    # initialize vertices and centrality scoring
    vertices = graph.nodes()
    nodes = graph.nodes()
    betweenness_centrality = {}
    total_paths = 0

    #iterate through all central nodes
    for middle_node in nodes:
        betweenness_centrality[middle_node] = 0
        total = 0
        # iterate through all paths
        for nodes_start in vertices:
            for nodes_end in vertices:
                if nx.has_path(graph, nodes_start, nodes_end):
                    paths = [p for p in nx.all_shortest_paths(G, nodes_start, nodes_end, weight=None)]
                    for ii in range(0,len(paths)):
                        scale = 1. / float(len(paths))
                        current_path = paths[ii]
                        if middle_node in current_path:
                            betweenness_centrality[middle_node] = betweenness_centrality[middle_node] + scale


    # normalize by n^2
    for key in betweenness_centrality.keys():
        betweenness_centrality[key] = betweenness_centrality[key]/((len(vertices))**2.0)
    return betweenness_centrality

# =================================================================================
# read in the data as dicitonary
# =================================================================================

file = open('/Users/wilder/PycharmProjects/Homework2_NAM/medici_network.txt', 'r')
data = file.readlines()
final_data = {}
for line in data:
    temp = deepcopy(line.split(" "))
    final_data[temp[0]] = {}
    final_data[temp[0]]['Family'] = temp[1].replace(",","")
    final_data[temp[0]]['Degree'] = temp[3]
    connections = [i for i, s in enumerate(temp) if '(' in s]
    edges = []
    for start in connections:

        temp1 = temp[start].replace("(","")
        temp1 = temp1.replace(",", "")
        temp1 = int(temp1)
        temp2 = temp[start+1].replace(")","")
        temp2 = int(temp2)
        edges.append((temp1,temp2))

    final_data[temp[0]]['Connections'] = edges

# =================================================================================
# Create graph in NetworkX
# =================================================================================
G=nx.Graph()

# add nodes
for nodes in final_data.keys():
    G.add_node(int(nodes))

# add connections
for nodes in final_data.keys():
    for connection in final_data[nodes]["Connections"]:
        G.add_edge(int(nodes), int(connection[0]))

# check that graph was set up
print(G.nodes())
print(G.edges())

# draw graph
#draw_graph(G.edges())

# =================================================================================
# Calculate then sort centrality scores
# =================================================================================

# find degree centrality (built in)
print("Degree Centrality (built in)")
dc = nx.degree_centrality(G)
sorted_dc = sorted(dc.items(), key=operator.itemgetter(1))[::-1]
print(sorted_dc)
# find degree centrality (built home-brew)
print("Degree Centrality (built home-brew)")
dc_h = hb_degree_centrality(G)
sorted_dc_h = sorted(dc_h.items(), key=operator.itemgetter(1))[::-1]
print(sorted_dc_h)

# find betweeness centrality (built in)
print("Betweenness Centrality (built in)")
bc = nx.betweenness_centrality(G)
sorted_bc = sorted(bc.items(), key=operator.itemgetter(1))[::-1]
print(sorted_bc)
# find betweeness centrality (built home-brew)
print("Betweenness Centrality (built home-brew)")
bc_h = hb_betweenness_centrality(G)
sorted_bc_h = sorted(bc_h.items(), key=operator.itemgetter(1))[::-1]
print(sorted_bc_h)

# find eigenvector centrality (built in)
print("Eigenvector Centrality")
ec = nx.eigenvector_centrality(G)
sorted_ec = sorted(ec.items(), key=operator.itemgetter(1))[::-1]
print(sorted_ec)
# find eigenvector centrality (built home-brew)
print("Eigenvector Centrality")
ec_h = hb_eigen_vector_centrality(G)
sorted_ec_h = sorted(ec_h.items(), key=operator.itemgetter(1))[::-1]
print(sorted_ec)

# find harmonic centrality (built in)
print ("Harmonic Centrality: (built in)")
hc = nx.harmonic_centrality(G)
sorted_hc = sorted(hc.items(), key=operator.itemgetter(1))[::-1]
print(sorted_hc)
# find harmonic centrality (built home-brew)
print ("Harmonic Centrality: (built home-brew)")
hc_h = hb_harmonic_centrality(G)
sorted_hc_h = sorted(hc_h.items(), key=operator.itemgetter(1))[::-1]
print(sorted_hc_h)

# =================================================================================
# mess around with generative model to determine structural importance of Medicis
# =================================================================================

# generate degree distribution from original graph
degree_dist = []
for nodes in G.nodes():
    degree_dist.append(G.degree(nodes))

# plot original network
draw_graph(G, "Graph of Family Connections")

# plot one of the generated networks
for ii in range(0,3):
    G_sim = perm_configuration_model(degree_dist)
    draw_graph(G_sim,"Simulated Graph of Family Connections")

# =================================================================================
# use generative model to determine structural importance of Medicis
# =================================================================================

# generate degree distribution from original graph
degree_dist = []
for nodes in G.nodes():
    degree_dist.append(G.degree(nodes))

# run it a bunch and find the average harmonic degree distribution
harmonic_centrality = []
total = 1000
for ii in range(0,total):
    graph_sim = perm_configuration_model(degree_dist)
    new_centrality = hb_harmonic_centrality(graph_sim)
    harmonic_centrality.append(new_centrality)

# now find the sample mean of these values
avg = {}
q_75 = {}
q_25 = {}
for key in harmonic_centrality[0].keys():
    current_sum = 0
    q_ = []
    for realization in range(0,total):
        current_realization = harmonic_centrality[realization]
        current_sum = current_sum + current_realization[key]
        q_.append(current_realization[key])
    q_75[key] = np.percentile(q_,75)
    q_25[key] = np.percentile(q_, 25)
    avg[key] = current_sum/total

# display final sample mean
print ("Average Simulated Harmonic Centrality")
avg_ = sorted(avg.items(), key=operator.itemgetter(1))[::-1]
print(avg_)

# calculate difference of the two
diff_calc_1 = dict(avg)
diff_calc_2 = dict(hc_h)
diff_calc = {}
diff_calc_q1 = {}
diff_calc_q2 = {}
for key in diff_calc_1.keys():
    diff_calc[key] = diff_calc_1[key] - diff_calc_2[key]
    diff_calc_q1[key] = q_75[key] - diff_calc_2[key]
    diff_calc_q2[key] = q_25[key] - diff_calc_2[key]
#diff_calc_ = sorted(diff_calc.items(), key=operator.itemgetter(1))[::-1]

# plot the the two values
plt.plot([avg[val] for val in avg.keys()],'r')
plt.plot([hc_h[val] for val in hc_h.keys()],'b')
plt.ylabel('Harmonic Centrality Score')
plt.xlabel('Vertex Number of Family')
plt.title('Centrality Scores')
plt.show()
plt.plot([diff_calc[val] for val in diff_calc.keys()], 'b')
plt.plot([diff_calc_q1[val] for val in q_25.keys()], 'r')
plt.plot([diff_calc_q2[val] for val in q_75.keys()], 'r')
plt.title('Difference of Simulated and True Centrality Scores')
plt.ylabel('Harmonic Centrality Score Difference')
plt.xlabel('Vertex Number of Family')
plt.show()




