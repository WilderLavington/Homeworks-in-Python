### Homework 4
### Jonathan Wilder Lavington
import networkx as nx
import matplotlib.pyplot as plt
from random import randint
from random import uniform
from copy import deepcopy
from numpy import linspace
from random import uniform
import scipy.fftpack
import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import dill

# =================================================================================
# Helper Fxns
# =================================================================================

def draw_graph(G,title_):

    graph_pos = nx.spring_layout(G, dim=2, k=None, pos=None, fixed=None, iterations=1000, weight='weight', scale=len(G.nodes()))
    #graph_pos = nx.shell_layout(G)
    # draw nodes, edges and labels
    nx.draw_networkx_nodes(G, graph_pos, node_size=100, node_color='blue', alpha=0.3)
    nx.draw_networkx_edges(G, graph_pos)
    nx.draw_networkx_labels(G, graph_pos, font_size=12, font_family='sans-serif')

    plt.title(title_)
    plt.show()

def simulate_epidemic(G,p):

    # initialize network
    nodes = G.nodes()
    infected = [0 for ii in range(len(nodes))]
    susceptible = [1 for ii in range(len(nodes))]
    start = np.random.randint(0, len(infected))
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

    return t,s,l

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

# =================================================================================
# Question 1
# =================================================================================

a1 = 0
b1 = 0
c1 = 0
plt.interactive(False)

### Question 1 (a) general community structure
if a1:
    n = 50.
    q = 2. # number of groups
    c = 5.
    epsilon = [0,4,8]

    # generate graphs
    p_in = (1/n)*(c + epsilon[0]/2)
    p_out = (1/n)*(c - epsilon[0]/2)
    G_0 = nx.planted_partition_graph(int(q), int(n/q), p_in, p_out, seed=42)
    p_in = (1/n)*(c + epsilon[1]/2)
    p_out = (1/n)*(c - epsilon[1]/2)
    G_1 = nx.planted_partition_graph(int(q), int(n/q), p_in, p_out, seed=42)
    p_in = (1/n)*(c + epsilon[2]/2)
    p_out = (1/n)*(c - epsilon[2]/2)
    G_2 = nx.planted_partition_graph(int(q), int(n/q), p_in, p_out, seed=42)

    # plot graphs
    draw_graph(G_0, "Planted Partition: epsilon = 0")
    draw_graph(G_1, "Planted Partition: epsilon = 4")
    draw_graph(G_2, "Planted Partition: epsilon = 8")

### Question 1 (b) measure average lenth of epidemic in network
if b1:
    n = 1000
    c = 8
    epsilon = 0
    q = 2.  # number of groups
    p_in = (1/n)*(c + epsilon/2)
    p_out = (1/n)*(c - epsilon/2)

    probabilities = linspace(0.0, 1.0, num=100)
    stored_stats = [() for ii in range(0,len(probabilities))]
    counter = -1

    for p in probabilities:

        # simulate graph
        G = nx.planted_partition_graph(int(q), int(n / q), p_in, p_out)
        counter += 1

        # simulate epidemic over graph
        t_avg = 0.0; s_avg = 0.0; l_avg = 0.0

        # sample graph statistics
        samples = 100
        for ii in range(0,samples):
            t, s, l = simulate_epidemic(G, p)
            t_avg = t_avg + t
            s_avg = s_avg + s
            l_avg = l_avg + l

        # calculate sample mean
        t_avg = t_avg/samples
        s_avg = s_avg/samples
        l_avg = l_avg/samples

        # store value
        stored_stats[counter] = (t_avg, s_avg, l_avg)
        print(p, (t_avg, s_avg, l_avg))

    p_crit1 = 0
    for idx, element in enumerate(stored_stats):
        if element[2] > 7:
            p_crit1 = probabilities[idx]
            break

    p_crit2 = 0
    for idx, element in enumerate(stored_stats):
        if element[1] > .32:
            p_crit2 = probabilities[idx]
            break

    f, ((ax1, ax2)) = plt.subplots(1, 2)

    ax1.set_title("Epidemic Length vs. Probability of Infection")
    ax1.plot(probabilities, [ element[2] for element in stored_stats ])
    ax1.axhline(math.log(n), color='black', linestyle='dashed')
    ax1.axvline(p_crit1, color='black', linestyle='dashed')
    ax1.set_ylabel('Epidemic Length (iterations)')
    ax1.set_xlabel('probability of infection transmission')

    ax2.set_title("Epidemic Size vs. Probability of Infection")
    ax2.plot(probabilities, [ element[1] for element in stored_stats ])
    ax2.axvline(p_crit2, color='black', linestyle='dashed')
    ax2.set_ylabel('Percent Infected')
    ax2.set_xlabel('probability of infection transmission')

    plt.show()

### Question 1 (c) see how community structure affects spread of virus
if c1:
    n = 200
    c = 8.
    q = 2.  # number of groups
    epsilons = linspace(0.0, 2*c, num=25)
    stored_stats = []
    probabilities = linspace(0.0, 1.0, num=25)

    # vary combinations of p and epsilon
    for epsilon in epsilons:

        p_in = (1 / n) * (c + epsilon / 2)
        p_out = (1 / n) * (c - epsilon / 2)

        stored_stats_p = [() for ii in range(0,len(probabilities))]
        counter = -1

        for p in probabilities:

            counter += 1

            # simulate epidemic over graph
            t_avg = 0.0; s_avg = 0.0; l_avg = 0.0

            # sample graph statistics
            samples = 50
            for ii in range(0,samples):
                # simulate graph
                G = nx.planted_partition_graph(int(q), int(n / q), p_in, p_out)
                t, s, l = simulate_epidemic(G, p)
                t_avg = t_avg + t
                s_avg = s_avg + s
                l_avg = l_avg + l

            # calculate sample mean
            t_avg = t_avg/samples
            s_avg = s_avg/samples
            l_avg = l_avg/samples

            # store value
            stored_stats_p[counter] = (t_avg, s_avg, l_avg)
            print(epsilon, p, (t_avg, s_avg, l_avg))

        stored_stats.append(deepcopy(stored_stats_p))


    ### plot change over epsilon and p on grid wrt epidemic size
    data = np.array([[stat[2] for stat in higher_stats] for higher_stats in stored_stats])
    length = data.shape[0]
    width = data.shape[1]
    x, y = np.meshgrid(np.array(probabilities), np.array(epsilons))

    fig1 = plt.figure()
    ax = fig1.add_subplot(1,1,1, projection='3d')
    ax.plot_surface(x, y, data)
    plt.title("Average length of epidemic in network as a function of p")
    plt.xlabel("Probability")
    plt.ylabel("Epsilon")
    plt.show()

    # data = np.array([[stat[1] for stat in higher_stats] for higher_stats in stored_stats])
    # length = data.shape[0]
    # width = data.shape[1]
    # x, y = np.meshgrid(np.array(probabilities), np.array(epsilons))
    #
    # fig1 = plt.figure()
    # ax = fig1.add_subplot(1,1,1, projection='3d')
    # ax.plot_surface(x, y, data)
    # plt.title("Average epidemic size  as a function of p")
    # plt.xlabel("Probability")
    # plt.ylabel("Epsilon")
    # plt.show()

# =================================================================================
# Question 2
# =================================================================================

a2 = 0
b2 = 1

#### question 2 (a)
if a2:

    # generate graph structure
    n = 50
    G = build_grid(n-1)
    #draw_graph(G, "grid model without added edges")

    probabilities = linspace(0.0, 1.0, num=100)
    stored_stats = [() for ii in range(0,len(probabilities))]
    counter = -1

    # simulate and average epidemics for different grid structures
    for p in probabilities:

        # simulate a bunch of epidemics over graph
        t_avg = 0.0; s_avg = 0.0; l_avg = 0.0

        # sample graph statistics
        samples = 50
        for ii in range(0,samples):
            t, s, l = simulate_epidemic(G, p)
            t_avg = t_avg + t
            s_avg = s_avg + s
            l_avg = l_avg + l

        # calculate sample mean
        t_avg = t_avg/samples
        s_avg = s_avg/samples
        l_avg = l_avg/samples

        # store value
        counter += 1
        stored_stats[counter] = (t_avg, s_avg, l_avg)
        print(p, (t_avg, s_avg, l_avg))

    p_crit1 = 0
    for idx, element in enumerate(stored_stats):
        if element[2] > 7:
            p_crit1 = probabilities[idx]
            break

    p_crit2 = 0
    for idx, element in enumerate(stored_stats):
        if element[1] > .32:
            p_crit2 = probabilities[idx]
            break
    print(p_crit2, p_crit2)

    f, ((ax1, ax2)) = plt.subplots(1, 2)

    ax1.set_title("Grid structure: average epidemic length as a function of p")
    ax1.plot(probabilities, [element[2] for element in stored_stats])
    ax1.axhline(math.log(n), color='black', linestyle='dashed')
    ax1.axvline(p_crit1, color='black', linestyle='dashed')
    ax1.set_ylabel('Epidemic Length (iterations)')
    ax1.set_xlabel('probability of infection transmission')

    ax2.set_title("Grid structure: average epidemic length as a function of p")
    ax2.plot(probabilities, [element[1] for element in stored_stats])
    ax2.axvline(p_crit2, color='black', linestyle='dashed')
    ax2.set_ylabel('Percent Infected')
    ax2.set_xlabel('probability of infection transmission')

    plt.show()

#### question 2 (b)
if b2:

    ### generate graph structure
    n = 25
    q = 0
    G = build_grid(n)
    probabilities = linspace(0.0, 1.0, num=20)
    q_grid = []

    ### iterate through different values of q
    for q in probabilities:

        ### generate new graph using q
        G = add_edges(G, q)
        # draw_graph(G, "grid model with added edges")
        stored_stats = []

        ### simulate and average epidemics for different grid structures
        for p in probabilities:

            ### simulate a bunch of epidemics over graph
            t_avg = 0.0; s_avg = 0.0; l_avg = 0.0

            ### sample graph statistics
            samples = 20
            for ii in range(0,samples):
                t, s, l = simulate_epidemic(G, p)
                t_avg = t_avg + t
                s_avg = s_avg + s
                l_avg = l_avg + l

            # calculate sample mean
            t_avg = t_avg/samples
            s_avg = s_avg/samples
            l_avg = l_avg/samples

            # store value
            stored_stats.append((t_avg, s_avg, l_avg))
            print(p, q, (t_avg, s_avg, l_avg))

        q_grid.append(deepcopy(stored_stats))

    ### plot change over p,q on grid
    # data = np.array([[stat[2] for stat in higher_stats] for higher_stats in q_grid])
    # length = data.shape[0]
    # width = data.shape[1]
    # x, y = np.meshgrid(np.array(probabilities), np.array(probabilities))
    #
    # fig1 = plt.figure()
    # ax = fig1.add_subplot(1, 1, 1, projection='3d')
    # ax.plot_surface(x, y, data)
    # plt.title("Average epidemic size  as a function of p,q")
    # plt.xlabel("Probability of infection (p)")
    # plt.ylabel("Probability of long range connection (q)")
    # plt.show()

    data = np.array([[stat[1] for stat in higher_stats] for higher_stats in q_grid ])
    length = data.shape[0]
    width = data.shape[1]
    x, y = np.meshgrid(np.array(probabilities), np.array(probabilities))

    fig2 = plt.figure()
    ax = fig2.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(x, y, data)
    plt.title("Average epidemic size  as a function of p,q")
    plt.xlabel("Probability of infection (p)")
    plt.ylabel("Probability of long range connection (q)")
    plt.show()
