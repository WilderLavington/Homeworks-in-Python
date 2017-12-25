# -*- coding: utf-8 -*-
"""
Jon W. Lavington 
AI Assignment 3
September 2016
"""
      ###=========================
        #GRAPH CLASS [node,connected nodes, weights, visited, Hueristic]
      ###=========================
import sys
class graph:
    def __init__(self):
        self.verticies = []
        self.edges = []
        self.Nodes = []
        self.edgAdd = 0
        self.vertAdd = 0
    def __edg__(self):
        return self.edgAdd
    def __ver__(self):
        return self.vertAdd
    def addVertex(self, value):
        if value in self.verticies:
            print ("Vertex already exists")
        else:
            self.verticies.append(value)
            self.Nodes.append(self.findVertex(value))
            self.vertAdd = self.vertAdd+1
    def addHueristic(self,node,value):
        if node in self.verticies:
            [self.Nodes[x].append(int(value)) for x in range(0,len(verticies)) if self.Nodes[x][0] == node]
        else:
            pass #node not found
    def findVertex(self,value):
        temp1 = [x for x in range(0,len(self.edges)) if self.edges[x][0] == value]
        temp2 = [self.edges[x][1] for x in temp1]
        wieght = map(int, [self.edges[x][2] for x in temp1])
        visited = 1 #unvisited is 1 visited is 0 
        val = [value,wieght,temp2,visited]
        return val
    def addEdge(self,value1,value2,value3):
        if value1 in self.verticies and value2 in self.verticies:
            # check for duplicates
            self.edges.append([value1,value2,int(value3)])
            self.edgAdd = self.edgAdd+1
            temp1 = [x for x in range(0,len(self.edges)) if self.edges[x][0] == value1]
        else:
            pass #One or more vertices not found.
        self.Nodes = [];
        for i in range(0,self.vertAdd):
            self.Nodes.append(self.findVertex(self.verticies[i]))
    def PrintNodes(self):
        temp = [self.Nodes[x] for x in range(0,self.vertAdd)]
        for x in range(0,len(verticies)):
            print(temp[x])
    def n_edges(self,node):
        vert = [self.Nodes[x][1] for x in range(0,len(verticies)-1) if self.Nodes[x][0] == node]
        edgew = [self.Nodes[x][2] for x in range(0,len(verticies)-1) if self.Nodes[x][0] == node]
        visited = [self.Nodes[x][3] for x in range(0,len(verticies)-1) if self.Nodes[x][0] == node]
        #shortestPath = ['inf' for x in range(0,len(verticies)-1) if self.Nodes[x][0] == node]
        huer = [self.Nodes[x][4] for x in range(0,len(verticies)-1) if self.Nodes[x][0] == node]
        q = []
        #returns tuples for each connected node with that nodes 
        #name, distance to from, shortest distance, hueristic
        [q.append(zip(edgew[x],vert[x],huer)) for x in range(0,len(vert))]
        q = q[0]
        return q
    def n_vertex(self,node):
        vert = [self.Nodes[x][0] for x in range(0,len(verticies)-1) if self.Nodes[x][0] == node]
        return vert[0]
    
g = graph()
#need command line argument
filename = raw_input()
#read in values from txt file 
#read in txt file
f = open(filename)
#print(f.read())
txt = f.read()
with open(filename) as f:
    content = txt.splitlines()
#Build graph
j = 0
while content[j] !='':
    v = content[j].split(',')
    for i in range(0,len(v)):
        v[i] = v[i].replace("[", "")
        v[i] = v[i].replace("]", "")
    content[j] = v
    j = j+1
for i in range(j+1,len(content)):
    v = content[i].split('=')
    content[i] = v
#print(content)
temp = [ content[x][0] for x in range(0,j)]
temp2 = [content[x][1] for x in range(0,j)]
[temp.extend(temp2[x]) for x in range(0,j)]
vals = [ [content[x][0],content[x][1],content[x][2]] for x in range(0,j)]
hs = [[content[x][0],content[x][1]] for x in range(j+1,len(content))]
verticies = list(set(temp))
for i in range(0,len(verticies)):
    g.addVertex(verticies[i])
for i in range(0,j):
        g.addEdge(vals[i][0],vals[i][1],vals[i][2])
        g.addEdge(vals[i][1],vals[i][0],vals[i][2])
        b_set = set(map(tuple,g.edges))  #need to convert the inner lists to tuples so they are hashable
        g.edges = map(list,b_set)
[g.addHueristic(hs[x-j-1][0],hs[x-j-1][1])for x in range(j,len(content)-1)]
G = {}
for i in range(0,len(g.verticies)):
    G[g.Nodes[i][0]] = {}
    for x in range(0,len(g.Nodes[i][1])):
        G[g.Nodes[i][0]][g.Nodes[i][2][x]] = g.Nodes[i][1][x]
###=========================
     #Dijkstra takes dict of dicts, returns shortest path
###=========================
def Dijkstra(graph,start,end,count,visited=[],distances={},predecessors={}):
    if start==end:
        path=[]
        while end != None:
            path.append(end)
            end=predecessors.get(end,None)
        return distances[start], path[::-1], count+1
    if not visited: distances[start]=0
    for neighbor in graph[start]:
        if neighbor not in visited:
            neighbordist = distances.get(neighbor,sys.maxint)
            tentativedist = distances[start] + graph[start][neighbor]
            if tentativedist < neighbordist:
                distances[neighbor] = tentativedist
                predecessors[neighbor]=start
    visited.append(start)
    unvisiteds = dict((k, distances.get(k,sys.maxint)) for k in graph if k not in visited)
    closestnode = min(unvisiteds, key=unvisiteds.get)
    count  = count+1
    return Dijkstra(graph,closestnode,end,count, visited,distances,predecessors)
#need to reformualte G for A star
def A_star(graph,start,end,count,visited=[],distances={},predecessors={}):
    if start==end:
        path=[]
        while end != None:
            path.append(end)
            end=predecessors.get(end,None)
        return distances[start], path[::-1], count+1
    if not visited: distances[start]=0
    for neighbor in graph[start]:
        if neighbor not in visited:
            neighbordist = distances.get(neighbor,sys.maxint) 
            tentativedist = distances[start] + graph[start][neighbor][0]
            if tentativedist < neighbordist:
                distances[neighbor] = tentativedist
                predecessors[neighbor]=start
    visited.append(start)
    unvisiteds = dict((k, distances.get(k,sys.maxint)) for k in graph if k not in visited)
    temp = unvisiteds
    for word in temp:
        for x in graph:
            if word in graph[x].keys():
                t = graph[x][word][1]
        temp[word] = temp[word]+t
    closestnode = min(temp, key=temp.get)
    count  = count+1
    return A_star(graph,closestnode,end,count,visited,distances,predecessors)
print("Built Graph:")
g.PrintNodes()
print 'Dijkstras Algorithm: (Path Cost, [Path], Total Nodes Evaluated, [Nodes Evaluated]) Total Nodes in graph]'
count = 0
print Dijkstra(G,'F','S',count),len(g.verticies)
temp = {}
for i in range(0,len(g.verticies)):
    temp[g.Nodes[i][0]] = g.Nodes[i][4]
#print(temp)
G = {}
for i in range(0,len(g.verticies)):
    G[g.Nodes[i][0]] = {}
    for x in range(0,len(g.Nodes[i][1])):
        #need to find node hueristics for each neighber
        G[g.Nodes[i][0]][g.Nodes[i][2][x]] = (g.Nodes[i][1][x],temp[g.Nodes[i][2][x]]) 
print 'A^* Algorithm: (Path Cost, [Path], Total Nodes Evaluated, [Nodes Evaluated]) Total Nodes in graph]'
count = 0
print A_star(G,'F','S',count),len(g.verticies)

