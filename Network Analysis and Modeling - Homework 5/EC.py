
import numpy as np
import networkx as nx
from random import uniform
from copy import deepcopy

def simulate_epidemic(G,p,start):

    # initialize network
    nodes = G.nodes()
    infected = [0 for ii in range(len(nodes))]
    susceptible = [1 for ii in range(len(nodes))]
    t = 1
    l = 1

    # set first infected

    infected[start] = 1
    susceptible[start] = 0

    # simulate epidemic
    while 1 in susceptible:

        # add a check to make sure the amount of carriers continues to change
        current_susceptible = deepcopy(susceptible)

        # iterate through carriers
        infected_nodes = deepcopy([i for i, x in enumerate(infected) if x == 1])

        # iterate through infected nodes
        for carriers in infected_nodes:

            # iterate through nieghbors of carriers
            neighbors = G.neighbors(carriers)
            susceptible_neighbors = [i for i in neighbors if susceptible[i] == 1]

            # check to see if they become infected or not
            for possible_infected in susceptible_neighbors:
                if uniform(0,1) < p:
                    infected[possible_infected] = 1
                    susceptible[possible_infected] = 0
                    l = t

        t += 1

        if current_susceptible == susceptible:
            break

    s = sum(infected)/len(infected)

    return l

def new_centrality(G, resolution):

    # average epidemic length for each node over different probilities
    probabilities = np.linspace(0,1,resolution)

    # iterate over all nodes
    centrality_score = []
    for nodes in G.nodes():
        avg = 0.0
        for p in probabilities:
            avg = avg + simulate_epidemic(G, p, int(nodes))
        centrality_score.append((nodes,avg/resolution))
        print(nodes)
    # sort on score
    centrality_score.sort(key=lambda x: x[1])
    centrality_score = centrality_score[::-1]
    # return sorted list
    return centrality_score

def build_grid(size):

    # initialize graph
    G = nx.Graph()

    # initialize nodes
    labels = [[str((i,j)) for i in range(0,size+1)] for j in range(0,size+1)]
    labels = [item for sublist in labels for item in sublist]

    # initialize nodes
    for label in labels:
        G.add_node(label)

    # add gridded edges
    for x in range(0, size+1):
        for y in range(0, size+1):
            ### corners
            if x == 0 and y == 0:
                G.add_edge(str((x,y)), str((x,y+1)))
                G.add_edge(str((x,y)), str((x+1,y)))
            elif x == size and y == size:
                G.add_edge(str((x,y)), str((x,y-1)))
                G.add_edge(str((x,y)), str((x-1,y)))
            elif  x == size and y == 0:
                G.add_edge(str((x, y)), str((x,y+1)))
                G.add_edge(str((x, y)), str((x-1,y)))
            elif x == 0 and y == size:
                G.add_edge(str((x, y)), str((x,y-1)))
                G.add_edge(str((x, y)), str((x+1,y)))

            ### walls
            elif x == size:
                G.add_edge(str((x, y)), str((x, y + 1)))
                G.add_edge(str((x, y)), str((x, y - 1)))
                G.add_edge(str((x, y)), str((x - 1, y)))
            elif x == 0:
                G.add_edge(str((x, y)), str((x, y + 1)))
                G.add_edge(str((x, y)), str((x, y - 1)))
                G.add_edge(str((x, y)), str((x + 1, y)))
            elif y == 0:
                G.add_edge(str((x, y)), str((x,y+1)))
                G.add_edge(str((x, y)), str((x+1,y)))
                G.add_edge(str((x, y)), str((x-1,y)))
            elif y == size:
                G.add_edge(str((x, y)), str((x,y-1)))
                G.add_edge(str((x, y)), str((x+1,y)))
                G.add_edge(str((x, y)), str((x-1,y)))

            ### interior
            else:
                G.add_edge(str((x, y)), str((x,y+1)))
                G.add_edge(str((x, y)), str((x,y-1)))
                G.add_edge(str((x, y)), str((x+1,y)))
                G.add_edge(str((x, y)), str((x-1,y)))

    # relabel nodes 0 to n**2 - 1
    mapping = dict(zip(G.nodes(), range(0, len(G.nodes()))))
    G_norm = nx.relabel_nodes(G, mapping)

    return G_norm

def add_edges(G,q):

    # iterate through
    nodes = G.nodes()

    for ii in range(0,len(nodes)-1):
        for jj in range(ii+1,len(nodes)):
            if ~G.has_edge(nodes[ii],nodes[jj]) and uniform(0,1) < q:
                G.add_edge(nodes[ii],nodes[jj])

    return G

# read in graph
#G_1 = nx.read_edgelist('data_set_1.txt')
#G_2 = nx.read_edgelist('data_set_2.txt')

# planted partition
n = 500.
q = 2. # number of groups
c = 5.
epsilon = 0

# planted partition
p_in = (1 / n) * (c + epsilon / 2)
p_out = (1 / n) * (c - epsilon / 2)
G_1 = nx.planted_partition_graph(int(q), int(n / q), p_in, p_out, seed=42)

#gridded partion
n = 23
G_2 = build_grid(n - 1)

# score_1 = new_centrality(G_1, 10)
# print()
# print('Network 1: ')
# for index in range(10):
#     print('Network: 1, ', 'Node: ', score_1[index][0], ', Average Score: ', score_1[index][1])
print()
print()
score_2 = new_centrality(G_2, 10)
print('Network 2: ')
for index in range(10):
    print('Network: 2, ', 'Node: ', score_2[index][0], ', Average Score: ', score_2[index][1])
