### =============================================================================
### Import Libraries
### =============================================================================

from numpy.random import uniform
from numpy import linspace
import matplotlib.pyplot as plt
from random import randint
from datetime import datetime
from collections import defaultdict
from math import floor
from numpy.random import randint

### =============================================================================
### Helper fxns
### =============================================================================

def simulate_network_plotting(c,n,r,seed_graph):

    # initialization
    x = [-1 for i in range(0, c * n)]  # stores the graph
    p = c / (c + r)  # attachment probability
    x[0:len(seed_graph)] = seed_graph

    # growth
    for t in range(3, n):
        for j in range(0, c):  # for each out - edge
            if uniform(0, 1) < p:  # choose by preferential attachment
                d = x[randint(0, c * (t - 1))]
            else:  # choose by uniform attachment
                d = randint(0, t - 1)
            x[c * (t - 1) + j] = d  # record the attachment
    x = [x_ for x_ in x if x_ != -1]

    ### create histogram dictionary for quicker access
    c = {}
    for val in x:
        if val not in c.keys():
            c[val] = 1
        else:
            c[val] += 1

    return c

def simulate_network_average(c,n,r,seed_graph_size):

    edges = []
    p = (c / (c + r))
    existing_edges = set()
    in_degrees = defaultdict(int)

    for i in range(seed_graph_size):
        j = 0
        while j < c:
            random_node = randint(seed_graph_size)
            if not random_node == i and (i, random_node) not in existing_edges:
                edges.append((i, random_node))
                existing_edges.update((i, random_node))
                in_degrees[random_node] += 1
                j += 1

    nodes = seed_graph_size

    while nodes < n:
        print(int(100*nodes/n))
        in_degrees[nodes] = 0
        k = 0
        while k < c:
            if uniform(0,1) < p:
                random_node = edges[randint(len(edges))][1]
            else:
                random_node = randint(nodes)
            if not random_node == nodes and (nodes, random_node) not in existing_edges:
                edges.append((nodes, random_node))
                existing_edges.update((nodes, random_node))
                in_degrees[random_node] += 1
                k += 1
        nodes += 1

    # ### find the number of nodes with zero degree
    # z = 0
    # for deg in in_degrees.keys():
    #     if in_degrees[deg] == 0:
    #         z += 1
    # print('percent zero degree')
    # print(z/len(in_degrees.keys()))

    return in_degrees

def find_top_bottom(degrees):

    print(degrees)
    ordered_degrees = sorted(degrees)
    # get first and last dates
    partition = floor(.1 * len(ordered_degrees))
    back = ordered_degrees[-partition - 1:-1]
    front = ordered_degrees[0:partition-1]

    # Calculate average citation for first and last 10
    av_front_dates = 0.
    av_back_dates = 0.
    for node in front:
        av_front_dates += degrees[node]
    for node in back:
        av_back_dates += degrees[node]

    # return values
    return av_front_dates / partition, av_back_dates / partition

def plot_ccdf(c, n, r, samples, include = 0):

    if not include:
        downsample = 1000
        max_degree = -1
        min_degree = 0
        for r_val in r:
            print(r_val)
            frac_geq = [0 for i in range(downsample)]
            for sample in range(samples):
                degrees = simulate_network_plotting(c,n,r_val,[1, 2, 1, 2, 3, 3])
                if min_degree > min(degrees.values()):
                    min_degree = min(degrees.values())
                if max_degree < max(degrees.values()):
                    max_degree = max(degrees.values())
                sampled_in_degree = linspace(min_degree, max_degree, downsample)
                for i, x in enumerate(sampled_in_degree):
                    print(i)
                    count = 0
                    for k in degrees.values():
                        if k >= x:
                            count += 1
                    frac_geq[i] += (count / len(degrees.values()))
            frac_geq = [i/samples for i in frac_geq]
            plt.loglog(sampled_in_degree, frac_geq)

        plt.title('CCDF of Pr(K > k_in) for network in-degree')
        plt.ylabel('p <= Fraction of vertices with in-degree')
        plt.xlabel('In-degree p')
        plt.legend(['r = ' + str(i) for i in r])
        plt.show()

    else:
        ### using PA
        downsample = 1000
        for r_val in r:
            frac_geq = [0 for i in range(downsample)]
            for sample in range(samples):
                degrees = simulate_network_plotting(c,n,r_val,[1, 2, 1, 2, 3, 3])
                min_degree = min(degrees.values())
                max_degree = max(degrees.values())
                sampled_in_degree = linspace(min_degree, max_degree, downsample)
                for i, x in enumerate(sampled_in_degree):
                    count = 0
                    for k in degrees.values():
                        if k >= x:
                            count += 1
                    frac_geq[i] += (count / len(degrees.values()))
            frac_geq = [i/samples for i in frac_geq]
            plt.loglog(sampled_in_degree, frac_geq)

        ### not using PA
        frac_geq = [0 for i in range(downsample)]
        for sample in range(samples):
            degrees, x = simulate_network_sans_PA(c,n,r,[1, 2, 1, 2, 3, 3])
            min_degree = min(degrees.values())
            max_degree = max(degrees.values())
            sampled_in_degree = linspace(min_degree, max_degree, downsample)
            for i, x in enumerate(sampled_in_degree):
                count = 0
                for k in degrees.values():
                    if k >= x:
                        count += 1
                frac_geq[i] += (count / len(degrees.values()))
        frac_geq = [i/samples for i in frac_geq]
        plt.loglog(sampled_in_degree, frac_geq)

        lab = ['r = ' + str(i) + ", PA" for i in r]
        lab.append("Non-PA")
        plt.title('CCDF of Pr(K > k_in) for network in-degree')
        plt.ylabel('p <= Fraction of vertices with in-degree')
        plt.xlabel('In-degree p')
        plt.legend(lab)
        plt.show()

def return_averages(c, n, r, seed_graph_size, sims):

    tops = []
    bottoms = []

    for sim in range(sims):
        degrees = simulate_network_average(c, n, r, seed_graph_size)
        top_av, bottom_av = find_top_bottom(degrees)
        tops.append(top_av)
        bottoms.append(bottom_av)

    return sum(tops) / len(tops), sum(bottoms) / len(bottoms)

def read_networks(edge_data, date_data):

    in_degree = defaultdict(int)
    with open(edge_data) as f:
        for line in f.readlines():
            edge = line.strip('\r\n').split('\t')
            if int(edge[1]) in in_degree.keys():
                in_degree[int(edge[1])] += 1
            else:
                in_degree[int(edge[1])] = 1
            if int(edge[0]) not in in_degree.keys():
                in_degree[int(edge[0])] = 0

    dates = {}
    with open(date_data) as f:
        f.readline()
        for line in f.readlines():
            line = line.strip('\r\n').split('\t')
            if int(line[0]) in in_degree.keys():
                dates[int(line[0])] = datetime.strptime(line[1], '%Y-%m-%d')

    ordered_dates = sorted(dates.items(), key=lambda x: x[1])

    # get first and last dates
    partition = floor(.1*len(ordered_dates))
    back_dates = ordered_dates[-partition-1:-1]
    back_dates = [b[0] for b in back_dates]
    front_dates = ordered_dates[1:partition]
    front_dates = [f[0] for f in front_dates]

    # Calculate average citation for first and last 10
    av_front_dates = 0.
    av_back_dates = 0.
    for node in front_dates:
        av_front_dates += in_degree[node]

    for node in back_dates:
        av_back_dates += in_degree[node]
    # return values
    return av_front_dates/partition, av_back_dates/partition

def simulate_network_sans_PA(c,n,r,seed_graph):

    # initialization
    x = [-1 for i in range(0, c * n)]  # stores the graph
    x[1:len(seed_graph)] = seed_graph

    # growth
    for t in range(4, n):
        for j in range(0, c):  # for each out - edge
            d = randint(0, t - 1)
            x[c * (t - 1) + j] = d  # record the attachment
    x = [x_ for x_ in x if x_ != -1]

    ### create histogram dictionary for quicker access
    c = {}
    for val in x:
        if val not in c.keys():
            c[val] = 1
        else:
            c[val] += 1

    return c, x

### =============================================================================
### main
### =============================================================================

def main():

    parta = 1
    partb = 1
    partc = 1
    partd = 1
    parte = 1

    if parta:
        ### initialize paramaters
        c = 3  # k_out
        n = 10 ** 4 # size of the network
        r = [1, 2, 3, 4] # uniform attachment contribution
        seed_graph_size = 10
        samples = 1
        ### simulate network growth over time for different r
        plot_ccdf(c, n, r, samples, 0)
        ### calculate degree info
        print(return_averages(c, n, r[0], seed_graph_size, 1))

    if partb:
        c = 12 # k_out
        r = [5] # uniform attachment contribution
        n = 10 ** 6 # size of the network
        seed_graph_size = 10
        sims = 1 # number of simulations to average over
        samples = 1
        ### simulate network growth over time for different r
        plot_ccdf(c, n, r, samples, 0)

        ### calculate degree info
        print(return_averages(c, n, r[0], seed_graph_size, sims))

    if partc:
        top_ten, bottom_ten = read_networks("cit-HepPh.txt", "cit-HepPh-dates.txt")
        print('part c')
        print('top_ten', top_ten)
        print('bottom_ten', bottom_ten)

    if partd:
        done = True

    if parte: # becuase its the extra credit
        r = [1, 4]
        c = 12
        n = 10 ** 6
        seed_graph_size = 10
        samples = 1
        sims = 1
        plot_ccdf(c, n, r, samples, 1)

        ### calculate degree info
        print(return_averages(c, n, r[0], seed_graph_size, sims))

if __name__ == "__main__":
    main()