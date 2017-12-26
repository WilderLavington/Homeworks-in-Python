
# import networks package, os and helpers
import networkx as nx
import os
from HW1_helper_fncs import mean_degree, mean_neighbor_degree, return_diameter, \
    return_mean_geodesic, largest_subgraph, approximations
import matplotlib.pyplot as plt
import datetime
import time

#=======================================================
# problem 6 b
#=======================================================

# find names of the edge list text files
edge_lists = os.listdir("/Users/wilder/PycharmProjects/HW1_NAM/facebook100txt")

# remove attribute files
for string in edge_lists:
    if ("_attr" in string) or ("readme" in string) or ("pdf" in string) or (".txt" not in string):
        edge_lists.remove(string)

# find ratio's of mean neighbor degree to mean degree of each network
ratios = {}
mds = {}
for files in edge_lists:
    #break
    print(files)
    path = "/Users/wilder/PycharmProjects/HW1_NAM/facebook100txt" + "/" + files
    current_graph = nx.read_edgelist(path)
    # call helper function for MD and MND
    MD = mean_degree(current_graph)
    MND = mean_neighbor_degree(current_graph)
    ratio = MND / MD
    ratios[files] = float(ratio)
    mds[files] = float(MD)

# create lists
z = list(mds.values())
y = list(ratios.values())
n = list(ratios.keys())

# create plot
fig, ax = plt.subplots()
ax.scatter(z, y)

# add labels, axis, title
for i in range(0, len(n)):
    ax.annotate(n[i], (z[i], y[i]), size = 'xx-small')
ax.set_title('The Minority Paradox Problem: ')
ax.set_xlabel('Mean Degree')
ax.set_ylabel('Mean Neighbor Degree / Mean Degree')

# add horozontal line
ax.axhline(y=1, c="blue", linewidth=0.5, zorder=0)

# plot
plt.show()

#=======================================================
# extra credit
#=======================================================

# (i) find the diameter of the largest component
diameter = {}
# (ii) find the mean geodesic distance between pairs of
#      vertices in the largest component of the network
mean_geodesic = {}
counter = 0
approx = [0 for ii in range(0,len(edge_lists))]
graph_size = [0 for ii in range(0,len(edge_lists))]
sub_graph_size = [0 for ii in range(0,len(edge_lists))]
# iterate and calculate
for files in edge_lists:

    # create timestamp
    ts = time.time()
    print('Time stamp', datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))

    # read edgelist
    path = "/Users/wilder/PycharmProjects/HW1_NAM/facebook100txt" + "/" + files
    current_graph = nx.read_edgelist(path)

    # find largest component
    new_sub_graph = largest_subgraph(current_graph)

    # approximate MG and D
    approx[counter] = approximations(new_sub_graph, 30*len(new_sub_graph.nodes()))
    # include the size of the total graph
    graph_size[counter] = len(current_graph.nodes())
    sub_graph_size[counter] = len(new_sub_graph.nodes())
    counter = counter + 1
    print(files + ": ", approx[counter-1], graph_size[counter-1], sub_graph_size[counter-1])

# create lists
var = map(list, zip(*approx))

print(var)
print(graph_size)
print(sub_graph_size)

# plot 1
z = list(edge_lists)
y = list(var[0])
n = list(graph_size)

# create plot
fig, ax = plt.subplots()
ax.scatter(z, y)

# add labels, axis, title
for i in range(0, len(n)):
    ax.annotate(n[i], (z[i], y[i]), size = 'xx-small')
ax.set_title('Diameter vs Network Size: ')
ax.set_xlabel('Diameter')
ax.set_ylabel('Network Size')

# plot 1
z = list(edge_lists)
y = list(var[1])
n = list(sub_graph_size)

# create plot
fig, ax = plt.subplots()
ax.scatter(z, y)

# add labels, axis, title
for i in range(0, len(n)):
    ax.annotate(n[i], (z[i], y[i]), size = 'xx-small')
ax.set_title('Average Geodesic Distance vs Largest Connected Sub-Graph Size: ')
ax.set_xlabel('Average Geodesic Distance')
ax.set_ylabel('Largest Connected Sub-graph Size')

