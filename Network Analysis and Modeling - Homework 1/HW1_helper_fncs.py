from networkx import connected_component_subgraphs, diameter, average_shortest_path_length, \
    shortest_path_length
from math import floor
from random import choice

def mean_degree(graph):
    dist = degree_dist(graph)
    squared_val = 0
    for nodes in dist:
        squared_val = squared_val + dist[nodes]
    sqr = float(squared_val) / len(dist.keys())
    return sqr

def mean_degree_squared(graph):
    dist = degree_dist(graph)
    squared_val = 0
    for nodes in dist:
        squared_val = squared_val + dist[nodes]*dist[nodes]
    sqr = squared_val / len(dist.keys())
    return sqr

def nodes_connected(u, v, graph):
    return u in graph.neighbors(v)

def degree_dist(graph):
    nodes = graph.nodes()
    dict = {}
    for node in nodes:
        n = graph.__getitem__(node)
        dict[node] = len(n.keys())
    return dict

def mean_neighbor_degree(graph):
    mean_squared = mean_degree_squared(graph)
    mean = mean_degree(graph)
    return mean_squared/mean

# extra credit
def largest_subgraph(current_graph):
    sub_graphs = connected_component_subgraphs(current_graph)
    for stuff in sub_graphs:
        largest_component = stuff
        break
    return largest_component

def return_diameter(sub_graph):
    # calculate diameter
    return diameter(sub_graph)

def return_mean_geodesic(sub_graph):
    # calculate mean geodesic
    return average_shortest_path_length(sub_graph)

def approximations(sub_graph,max_evals):
    # approximate diameter, as well as mean geodesic
    nodes = sub_graph.nodes()
    path_lengths = 0
    total_iterations = 0

    mean_geo = 0
    mean_geo_sqr = 0
    diameter_ = -1

    for iterations in range(1,max_evals):

        # pick start and end at random
        current_start = choice(nodes)
        current_end = choice(nodes)

        # check they are not the same
        if current_start == current_end:
            current_end = choice(nodes)

        # calculate path length
        paths = shortest_path_length(sub_graph, source = current_start, target = current_end )

        # add info to averages
        mean_geo = mean_geo + paths
        mean_geo_sqr = mean_geo_sqr + (paths)**2
        total_iterations = total_iterations + 1

        # update diameter
        if paths > diameter_:
            diameter_ = paths

    return diameter_, float(mean_geo)/total_iterations, (float(mean_geo_sqr)/total_iterations - (float(mean_geo)/total_iterations)**2)**(1/2.0)



