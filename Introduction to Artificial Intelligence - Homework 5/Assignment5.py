# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 20:09:02 2016

@author: wilder
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 10:53:30 2016

@author: wilder
"""
import pandas as p
import random as r
from random import randrange
import sys

def Initialize(poolsize, districts, matsize):
    tiles = matsize**2
    pool = 0
    genepool = []
    while pool < poolsize:
        appen = True
        applicant = [[0 for i in range(0,matsize)] for i in range(0,matsize)]
        assigned = 0
        pool = pool+1
        seq2 = [i for i in range(0,districts-1)]
        r.shuffle(seq2)
        applicant2 = seq2
        #randomize start
        index111 = r.randint(0,len(applicant)-1)
        index222 = r.randint(0,len(applicant)-1)
        next_start = [index222,index111]
        while assigned < districts-1: 
            if next_start == (100,100):
                pool = pool-1
                appen = False
                break
            else:
                applicant[next_start[0]][next_start[1]] = applicant2[assigned]
                number = r.randint(1,tiles-2*districts)
                for i in range(0,number):
                    current_positions = []
                    for k in range(0,matsize):
                        for p in range(0,matsize):
                            if applicant[k][p] == applicant2[assigned]:
                                current_positions.append((k,p))
                    temp = applicant
                    applicant = add_district(current_positions, applicant)
                    if applicant == temp:
                        i = i-1
                    elif not still_viable(applicant):
                        i = i-1
                        applicant = temp
                    else:
                        continue 
                #printmat(applicant)
                #find next position to start district
                next_start = next_district(applicant)
                assigned = assigned+1
        
        if appen:
            # fill the rest of the open space
            for i in range(0,len(applicant)-1):
                while 0 in applicant[i]:
                   currentindex = applicant[i].index(0)
                   picks = zero_fil(applicant,(i,currentindex))
                   pick = r.randint(0,len(picks)-1)
                   applicant[i][currentindex] = applicant[picks[pick][0]][picks[pick][1]]
            #printmat(applicant)
            #push applicant onto pool
            genepool.append(applicant)
    return genepool
    
def print_genepool(genepool):
    for i in range(0,len(genepool)):
        print("organism: ", i)
        printmat(genepool[i])
        
def _zero_fil(applicant,current_position):
    possible_adds = []
    #corners
    if  current_position[0] == 0 and current_position[1] == 0:
        if applicant[1][0] != 0:
            possible_adds.append((1,0))
        if applicant[1][1] != 0:
            possible_adds.append((1,1))
        if applicant[0][1] != 0:
            possible_adds.append((0,1))
        if possible_adds:
            return possible_adds
        else:
            return 100
            
    elif  current_position[0] == len(applicant)-1 and current_position[1] == 0:
        if applicant[current_position[0]-1][current_position[1]] != 0:
            possible_adds.append((current_position[0]-1,current_position[1]))
        if applicant[current_position[0]-1][current_position[1]+1] != 0:
            possible_adds.append((current_position[0]-1,current_position[1]+1))
        if applicant[current_position[0]][current_position[1]+1] != 0:
            possible_adds.append((current_position[0],current_position[1]))
        if possible_adds:
            return possible_adds
        else:
            return 100
        
    elif  current_position[0] == len(applicant)-1 and current_position[1] != len(applicant)-1:
        if applicant[current_position[0]-1][current_position[1]] != 0:
            possible_adds.append((current_position[0]-1,current_position[1]))
        if applicant[current_position[0]-1][current_position[1]-1] != 0:
            possible_adds.append((current_position[0]-1,current_position[1]-1))
        if applicant[current_position[0]][current_position[1]-1] != 0:
            possible_adds.append((current_position[0],current_position[1]-1))
        if possible_adds:
            return possible_adds
        else:
            return 100
            
    elif  current_position[0] == 0 and current_position[1] == len(applicant)-1:
        if applicant[current_position[0]+1][current_position[1]] != 0:
            possible_adds.append((current_position[0]+1,current_position[1]))
        if applicant[current_position[0]+1][current_position[1]-1] != 0:
            possible_adds.append((current_position[0]+1,current_position[1]-1))
        if applicant[current_position[0]][current_position[1]-1] != 0:
            possible_adds.append((current_position[0],current_position[1]-1))
        if possible_adds:
           return possible_adds
        else:
            return 100
    #sides 
    elif  current_position[0] == 0:
        if applicant[current_position[0]+1][current_position[1]] != 0:
            possible_adds.append((current_position[0]+1,current_position[1]))
        if applicant[current_position[0]+1][current_position[1]-1] != 0:
            possible_adds.append((current_position[0]+1,current_position[1]-1))
        if applicant[current_position[0]][current_position[1]-1] != 0:
            possible_adds.append((current_position[0],current_position[1]-1))
        if applicant[current_position[0]][current_position[1]+1] != 0:
            possible_adds.append((current_position[0],current_position[1]+1))
        if applicant[current_position[0]+1][current_position[1]+1] != 0:
            possible_adds.append((current_position[0]+1,current_position[1]+1))
        if possible_adds:
            return possible_adds
        else:
            return 100
            
    elif  current_position[0] == len(applicant)-1:
        if applicant[current_position[0]-1][current_position[1]] != 0:
            possible_adds.append((current_position[0]-1,current_position[1]))
        if applicant[current_position[0]-1][current_position[1]-1] != 0:
            possible_adds.append((current_position[0]-1,current_position[1]-1))
        if applicant[current_position[0]][current_position[1]-1] != 0:
            possible_adds.append((current_position[0],current_position[1]-1))
        if applicant[current_position[0]][current_position[1]+1] != 0:
            possible_adds.append((current_position[0],current_position[1]+1))
        if applicant[current_position[0]-1][current_position[1]+1] != 0:
            possible_adds.append((current_position[0]-1,current_position[1]+1))
        if possible_adds:
            return possible_adds
        else:
            return 100
            
    elif  current_position[1] == len(applicant)-1:
        if applicant[current_position[0]][current_position[1]-1] != 0:
            possible_adds.append((current_position[0],current_position[1]-1))
        if applicant[current_position[0]-1][current_position[1]-1] != 0:
            possible_adds.append((current_position[0]-1,current_position[1]-1))
        if applicant[current_position[0]+1][current_position[1]-1] != 0:
            possible_adds.append((current_position[0]+1,current_position[1]-1))
        if applicant[current_position[0]+1][current_position[1]] != 0:
            possible_adds.append((current_position[0]+1,current_position[1]))
        if applicant[current_position[0]-1][current_position[1]] != 0:
            possible_adds.append((current_position[0]-1,current_position[1]))
        if possible_adds:
            return possible_adds
        else:
            return 100
    elif  current_position[1] == 0:
        if applicant[current_position[0]][current_position[1]+1] != 0:
            possible_adds.append((current_position[0],current_position[1]+1))
        if applicant[current_position[0]-1][current_position[1]+1] != 0:
            possible_adds.append((current_position[0]-1,current_position[1]+1))
        if applicant[current_position[0]+1][current_position[1]+1] != 0:
            possible_adds.append((current_position[0]+1,current_position[1]+1))
        if applicant[current_position[0]+1][current_position[1]] != 0:
            possible_adds.append((current_position[0]+1,current_position[1]))
        if applicant[current_position[0]-1][current_position[1]] != 0:
            possible_adds.append((current_position[0]-1,current_position[1]))
        if possible_adds:
            return possible_adds
        else:
            return 100
    #interior
    else:
        if applicant[current_position[0]+1][current_position[1]+1] != 0:
            possible_adds.append((current_position[0]+1,current_position[1]+1))
        if applicant[current_position[0]][current_position[1]+1] != 0:
            possible_adds.append((current_position[0],current_position[1]+1))
        if applicant[current_position[0]+1][current_position[1]] != 0:
            possible_adds.append((current_position[0]+1,current_position[1]))
        if applicant[current_position[0]-1][current_position[1]-1] != 0:
            possible_adds.append((current_position[0]-1,current_position[1]-1))
        if applicant[current_position[0]-1][current_position[1]] != 0:
            possible_adds.append((current_position[0]-1,current_position[1]))
        if applicant[current_position[0]][current_position[1]-1] != 0:
            possible_adds.append((current_position[0],current_position[1]-1))
        if applicant[current_position[0]-1][current_position[1]+1] != 0:
            possible_adds.append((current_position[0]-1,current_position[1]+1))
        if applicant[current_position[0]+1][current_position[1]-1] != 0:
            possible_adds.append((current_position[0]+1,current_position[1]-1))
        if possible_adds:
            return possible_adds
        else:
            return 100

    
def zero_fil(applicant,current_position):
    non_zero_positions = _zero_fil(applicant,current_position)
    #print(non_zero_positions)
    return non_zero_positions
    
def printmat(matrix):
    for i in range(0,len(matrix)-1):
        print(matrix[i])
        
def next_district(applicant):
    for i in range(0,len(applicant)-1):
        for j in range(0,len(applicant)-1):
            if applicant[i][j] == 0:
                return (i,j)
    return (100,100)
    
def still_viable(mat):
    #check that all zeros 
    isval = False
    #upper left
    if mat[0][0] == mat[1][0]:
        isval = True
    elif mat[0][0] == mat[0][1]:
        isval = True
    elif mat[0][0] == mat[1][1]:
        isval = True
    else: 
        return False
    #upper right
    if mat[0][len(mat)-1] == mat[1][len(mat)-1]:
        isval = True
    elif mat[0][len(mat)-1] == mat[0][len(mat)-2]:
        isval = True
    elif mat[0][len(mat)-1] == mat[1][len(mat)-2]:
        isval = True
    else: 
        return False
    #lower right
    if mat[len(mat)-1][len(mat)-1] == mat[len(mat)-2][len(mat)-1]:
        isval = True
    elif mat[len(mat)-1][len(mat)-1] == mat[len(mat)-2][len(mat)-2]:
        isval = True
    elif mat[len(mat)-1][len(mat)-1] == mat[len(mat)-1][len(mat)-2]:
        isval = True
    else: 
        return False
    #lower right
    if mat[len(mat)-1][0] == mat[len(mat)-2][0]:
        isval = True
    elif mat[len(mat)-1][0] == mat[len(mat)-1][1]:
        isval = True
    elif mat[len(mat)-1][0] == mat[len(mat)-2][1]:
        isval = True
    else: 
        return False 
    #interior
    for i in range(1, len(mat)-1):
        for j in range(1, len(mat)-1): 
            if mat[i][j] == mat[i-1][j]:
                isval = True
            elif mat[i][j] == mat[i+1][j]:
                isval = True
            elif mat[i][j] == mat[i][j-1]:
                isval = True
            elif mat[i][j] == mat[i][j+1]:
                isval = True
            elif mat[i][j] == mat[i-1][j-1]:
                isval = True
            elif mat[i][j] == mat[i+1][j-1]:
                isval = True
            elif mat[i][j] == mat[i-1][j+1]:
                isval = True
            elif mat[i][j] == mat[i+1][j+1]:
                isval = True
            else:
                return False
    return isval
def not_surrounded(current_position, applicant):
    surrounded = True
    #corners
    if  current_position[0] == 0 and current_position[1] == 0:
        if applicant[1][0] == 0:
            return surrounded
        elif applicant[1][1] == 0:
            return surrounded
        elif applicant[0][1] == 0:
            return surrounded
        else:
            return False
    elif  current_position[0] == len(applicant)-1 and current_position[1] == 0:
        if applicant[current_position[0]-1][current_position[1]] == 0:
            return surrounded
        elif applicant[current_position[0]-1][current_position[1]+1] == 0:
            return surrounded
        elif applicant[current_position[0]][current_position[1]+1] == 0:
            return surrounded
        else:
            return False
    elif  current_position[0] == len(applicant)-1 and current_position[1] == len(applicant)-1:
        if applicant[current_position[0]-1][current_position[1]] == 0:
            return surrounded
        elif applicant[current_position[0]-1][current_position[1]-1] == 0:
            return surrounded
        elif applicant[current_position[0]][current_position[1]-1] == 0:
            return surrounded
        else:
            return False
    elif  current_position[0] == 0 and current_position[1] == len(applicant)-1:
        if applicant[current_position[0]+1][current_position[1]] == 0:
            return surrounded
        elif applicant[current_position[0]+1][current_position[1]-1] == 0:
            return surrounded
        elif applicant[current_position[0]][current_position[1]-1] == 0:
            return surrounded
        else:
            return False
    #sides 
    elif  current_position[0] == 0:
        if applicant[current_position[0]+1][current_position[1]] == 0:
            return surrounded
        elif applicant[current_position[0]+1][current_position[1]-1] == 0:
            return surrounded
        elif applicant[current_position[0]][current_position[1]-1] == 0:
            return surrounded
        elif applicant[current_position[0]][current_position[1]+1] == 0:
            return surrounded
        elif applicant[current_position[0]+1][current_position[1]+1] == 0:
            return surrounded
        else:
            return False
    elif  current_position[0] == len(applicant)-1:
        if applicant[current_position[0]-1][current_position[1]] == 0:
            return surrounded
        elif applicant[current_position[0]-1][current_position[1]-1] == 0:
            return surrounded
        elif applicant[current_position[0]][current_position[1]-1] == 0:
            return surrounded
        elif applicant[current_position[0]][current_position[1]+1] == 0:
            return surrounded
        elif applicant[current_position[0]-1][current_position[1]+1] == 0:
            return surrounded
        else:
            return False
    elif  current_position[1] == len(applicant)-1:
        if applicant[current_position[0]][current_position[1]-1] == 0:
            return surrounded
        elif applicant[current_position[0]-1][current_position[1]-1] == 0:
            return surrounded
        elif applicant[current_position[0]+1][current_position[1]-1] == 0:
            return surrounded
        elif applicant[current_position[0]+1][current_position[1]] == 0:
            return surrounded
        elif applicant[current_position[0]-1][current_position[1]] == 0:
            return surrounded
        else:
            return False
    elif  current_position[1] == 0:
        if applicant[current_position[0]][current_position[1]+1] == 0:
            return surrounded
        elif applicant[current_position[0]-1][current_position[1]+1] == 0:
            return surrounded
        elif applicant[current_position[0]+1][current_position[1]+1] == 0:
            return surrounded
        elif applicant[current_position[0]+1][current_position[1]] == 0:
            return surrounded
        elif applicant[current_position[0]-1][current_position[1]] == 0:
            return surrounded
        else:
            return False
    #interior
    else:
        if applicant[current_position[0]+1][current_position[1]+1] == 0:
            return surrounded
        elif applicant[current_position[0]][current_position[1]+1] == 0:
            return surrounded
        elif applicant[current_position[0]+1][current_position[1]] == 0:
            return surrounded
        elif applicant[current_position[0]-1][current_position[1]-1] == 0:
            return surrounded
        elif applicant[current_position[0]-1][current_position[1]] == 0:
            return surrounded
        elif applicant[current_position[0]][current_position[1]-1] == 0:
            return surrounded
        elif applicant[current_position[0]-1][current_position[1]+1] == 0:
            return surrounded
        elif applicant[current_position[0]+1][current_position[1]-1] == 0:
            return surrounded
        else:
            return False
            
def open_to_add(current_position, applicant):
    possible_adds = []
    #corners
    if  current_position[0] == 0 and current_position[1] == 0:
        if applicant[1][0] == 0:
            possible_adds.append((1,0))
        if applicant[1][1] == 0:
            possible_adds.append((1,1))
        if applicant[0][1] == 0:
            possible_adds.append((0,1))
        if possible_adds:
            return possible_adds
        else:
            return 100
            
    elif  current_position[0] == len(applicant)-1 and current_position[1] == 0:
        if applicant[current_position[0]-1][current_position[1]] == 0:
            possible_adds.append((current_position[0]-1,current_position[1]))
        if applicant[current_position[0]-1][current_position[1]+1] == 0:
            possible_adds.append((current_position[0]-1,current_position[1]+1))
        if applicant[current_position[0]][current_position[1]+1] == 0:
            possible_adds.append((current_position[0],current_position[1]))
        if possible_adds:
            return possible_adds
        else:
            return 100
        
    elif  current_position[0] == len(applicant)-1 and current_position[1] == len(applicant)-1:
        if applicant[current_position[0]-1][current_position[1]] == 0:
            possible_adds.append((current_position[0]-1,current_position[1]))
        if applicant[current_position[0]-1][current_position[1]-1] == 0:
            possible_adds.append((current_position[0]-1,current_position[1]-1))
        if applicant[current_position[0]][current_position[1]-1] == 0:
            possible_adds.append((current_position[0],current_position[1]-1))
        if possible_adds:
            return possible_adds
        else:
            return 100
            
    elif  current_position[0] == 0 and current_position[1] == len(applicant)-1:
        if applicant[current_position[0]+1][current_position[1]] == 0:
            possible_adds.append((current_position[0]+1,current_position[1]))
        if applicant[current_position[0]+1][current_position[1]-1] == 0:
            possible_adds.append((current_position[0]+1,current_position[1]-1))
        if applicant[current_position[0]][current_position[1]-1] == 0:
            possible_adds.append((current_position[0],current_position[1]-1))
        if possible_adds:
           return possible_adds
        else:
            return 100
    #sides 
    elif  current_position[0] == 0:
        if applicant[current_position[0]+1][current_position[1]] == 0:
            possible_adds.append((current_position[0]+1,current_position[1]))
        if applicant[current_position[0]+1][current_position[1]-1] == 0:
            possible_adds.append((current_position[0]+1,current_position[1]-1))
        if applicant[current_position[0]][current_position[1]-1] == 0:
            possible_adds.append((current_position[0],current_position[1]-1))
        if applicant[current_position[0]][current_position[1]+1] == 0:
            possible_adds.append((current_position[0],current_position[1]+1))
        if applicant[current_position[0]+1][current_position[1]+1] == 0:
            possible_adds.append((current_position[0]+1,current_position[1]+1))
        if possible_adds:
            return possible_adds
        else:
            return 100
            
    elif  current_position[0] == len(applicant)-1:
        if applicant[current_position[0]-1][current_position[1]] == 0:
            possible_adds.append((current_position[0]-1,current_position[1]))
        if applicant[current_position[0]-1][current_position[1]-1] == 0:
            possible_adds.append((current_position[0]-1,current_position[1]-1))
        if applicant[current_position[0]][current_position[1]-1] == 0:
            possible_adds.append((current_position[0],current_position[1]-1))
        if applicant[current_position[0]][current_position[1]+1] == 0:
            possible_adds.append((current_position[0],current_position[1]+1))
        if applicant[current_position[0]-1][current_position[1]+1] == 0:
            possible_adds.append((current_position[0]-1,current_position[1]+1))
        if possible_adds:
            return possible_adds
        else:
            return 100
            
    elif  current_position[1] == len(applicant)-1:
        if applicant[current_position[0]][current_position[1]-1] == 0:
            possible_adds.append((current_position[0],current_position[1]-1))
        if applicant[current_position[0]-1][current_position[1]-1] == 0:
            possible_adds.append((current_position[0]-1,current_position[1]-1))
        if applicant[current_position[0]+1][current_position[1]-1] == 0:
            possible_adds.append((current_position[0]+1,current_position[1]-1))
        if applicant[current_position[0]+1][current_position[1]] == 0:
            possible_adds.append((current_position[0]+1,current_position[1]))
        if applicant[current_position[0]-1][current_position[1]] == 0:
            possible_adds.append((current_position[0]-1,current_position[1]))
        if possible_adds:
            return possible_adds
        else:
            return 100
    elif  current_position[1] == 0:
        if applicant[current_position[0]][current_position[1]+1] == 0:
            possible_adds.append((current_position[0],current_position[1]+1))
        if applicant[current_position[0]-1][current_position[1]+1] == 0:
            possible_adds.append((current_position[0]-1,current_position[1]+1))
        if applicant[current_position[0]+1][current_position[1]+1] == 0:
            possible_adds.append((current_position[0]+1,current_position[1]+1))
        if applicant[current_position[0]+1][current_position[1]] == 0:
            possible_adds.append((current_position[0]+1,current_position[1]))
        if applicant[current_position[0]-1][current_position[1]] == 0:
            possible_adds.append((current_position[0]-1,current_position[1]))
        if possible_adds:
            return possible_adds
        else:
            return 100
    #interior
    else:
        if applicant[current_position[0]+1][current_position[1]+1] == 0:
            possible_adds.append((current_position[0]+1,current_position[1]+1))
        if applicant[current_position[0]][current_position[1]+1] == 0:
            possible_adds.append((current_position[0],current_position[1]+1))
        if applicant[current_position[0]+1][current_position[1]] == 0:
            possible_adds.append((current_position[0]+1,current_position[1]))
        if applicant[current_position[0]-1][current_position[1]-1] == 0:
            possible_adds.append((current_position[0]-1,current_position[1]-1))
        if applicant[current_position[0]-1][current_position[1]] == 0:
            possible_adds.append((current_position[0]-1,current_position[1]))
        if applicant[current_position[0]][current_position[1]-1] == 0:
            possible_adds.append((current_position[0],current_position[1]-1))
        if applicant[current_position[0]-1][current_position[1]+1] == 0:
            possible_adds.append((current_position[0]-1,current_position[1]+1))
        if applicant[current_position[0]+1][current_position[1]-1] == 0:
            possible_adds.append((current_position[0]+1,current_position[1]-1))
        if possible_adds:
            return possible_adds
        else:
            return 100

def add_district(current_positions, applicant):    
    #find a position to add from
    if len(current_positions) == 1:
        if not not_surrounded(current_positions[0], applicant):
            return applicant
        else:
            current_position = current_positions[0]
    else:
        vect = [i for i in range(0,len(current_positions))]
        r.shuffle(vect)
        i = 0
        if i == len(current_positions):
            return applicant
        else:            
            while not not_surrounded(current_positions[vect[i]], applicant):
                if i == len(current_positions):
                    return applicant
                i = i+1
                if i == len(vect):
                    return applicant
            current_position = current_positions[i]
    # find a viable node to expand to
    possible_nodes = open_to_add(current_position, applicant)
    if possible_nodes == 100:
        return applicant
    else:
        if isinstance(possible_nodes, list):
            vect = [i for i in range(0,len(possible_nodes))]
            r.shuffle(vect)
            vect = vect[0]
            node_to_change = possible_nodes[vect]
            applicant[node_to_change[0]][node_to_change[1]] = applicant[current_position[0]][current_position[1]]
        else:
            node_to_change = possible_nodes
            applicant[node_to_change[0]][node_to_change[1]] = applicant[current_position[0]][current_position[1]]
        return applicant
#####
        # Hunger games helper functions
#####

def Ffitness(current,true,districts):
    Fcurrent = _Ffitness(current,true,districts)
    Ftrue = float(sum(x.count('D') for x in true))/(float(len(true))**2)
    if  Fcurrent != 0:   
        Fcurrent = float(Fcurrent.count('dragons'))/len(Fcurrent)
        return (Fcurrent-Ftrue)**2
    else:
        return Ftrue
    
def _Ffitness(current,true,districts):
    newdistricts = []
    for i in range(1,districts):
        indexstore = []
        for j in range(0,len(current)-1):
            for k in range(0,len(current)-1):
                if current[j][k] == i:
                    indexstore.append(true[j][k])
        if len(indexstore) == 0:
            return 0
        percent_D = float(indexstore.count('D'))
        percent_D = percent_D/len(indexstore)
        percent_R = float(indexstore.count('R'))
        percent_R = percent_R/len(indexstore)
        if percent_R > percent_D:
            district_percent = ('rabbits', float(percent_R))
        else:
            district_percent = ('dragons', float(percent_D))
        newdistricts.append(district_percent[0])
    return newdistricts
    
def Finalfitness(current,true,districts):
    Fcurrent = _Ffitness(current,true,districts)
    Ftrue = float(sum(x.count('D') for x in true))/(float(len(true))**2)
    if  Fcurrent != 0:   
        Fcurrent = float(Fcurrent.count('dragons'))/len(Fcurrent)
        return (Fcurrent)
    else:
        return Ftrue
        
def Finalfitnessoth(current,true,districts):
    Fcurrent = _Ffitness(current,true,districts)
    Ftrue = float(sum(x.count('D') for x in true))
    if  Fcurrent != 0:   
        Fcurrent = float(Fcurrent.count('dragons'))
        return (Fcurrent)
    else:
        return Ftrue
        
def find_best(genepool,n,true,districts):
    scores = []
    for i in range(0,len(genepool)):
        score = Ffitness(genepool[i],true,districts)
        scores.append((i,score))
    scores.sort(key=lambda x: x[1])
    scores = scores[0:n-1]
    return scores

def partydev(current, true, districts):
    check = [i for i in range(1,districts)]
    dist = []
    for i in range(0, len(check)):
        indexstore = []
        for j in range(0,len(current)-1):
            for k in range(0,len(current)-1):
                if current[j][k] == check[i]:
                    indexstore.append((j,k))
        dist.append(indexstore)
    return dist

def mutate(genepool, districts):
    mutants = []
    for i in range(0,len(genepool)):
        gene = list(genepool[i])
        mutants.append(gene)
        if still_all_districts(gene,districts):
            if connected(gene,districts):
                for j in range(0,len(gene)-1):
                    for k in range(0,len(gene)-1):
                        mutatable_positions = _mutatable_positions(gene,[j,k])
                        if isinstance(mutatable_positions, list):
                            for l in range(0,len(mutatable_positions)-1): 
                                new_gene = list(gene)
                                #printmat(gene)
                                new_gene[mutatable_positions[l][0]][mutatable_positions[l][1]] = gene[j][k]
                                if connected(new_gene, districts) and still_all_districts(new_gene,districts):
                                    mutants.append(new_gene)
    return  mutants
                                    
def connected(applicant, districts):
    #print_connections(applicant)
    still_connected = False
    check = [i for i in range(1,districts+1)]
    connections = create_connections(applicant)
    for i in range(0,len(check)-1):
        current_district_check = []
        for j in range(0,len(connections)-1):
            if connections[j] != 100:
                if connections[j][0] == check[i]:
                    current_district_check.append([connections[j][1],connections[j][2]])
        #subgraph
        G = {}
        for q in range(0,len(current_district_check)):
            temp = str(current_district_check[q][0])
            if isinstance(current_district_check[q][1], list):
                temp1 = [str(current_district_check[q][1][x]) for x in range(0,len(current_district_check[q][1])-1)]
                G[temp] = set(temp1)
            else:
                G[temp] = []
        keys = list(G.keys())
        temporary = []
        for x in range(0,len(keys)):
            for y in range(0,len(keys)):
                still_connected = True
                newlist = list(bfs_paths(G, keys[x], keys[y]))
                if not newlist:
                    still_connected = False
                temporary.append(still_connected)
        if False in temporary:
            still_connected = False
        else:
            still_connected = True
            
    return still_connected

def bfs_paths(graph, start, goal):
    queue = [(start, [start])]
    while queue:
        (vertex, path) = queue.pop(0)
        for next in graph[vertex] - set(path):
            if next == goal:
                yield path + [next]
            else:
                queue.append((next, path + [next]))
                
def all_connections(applicant,current_position):
    possible_adds = []
    type1 = applicant[current_position[0]][current_position[1]]
    #corners
    if  current_position[0] == 0 and current_position[1] == 0:
        if applicant[1][0] == type1:
            possible_adds.append((1,0))
        if applicant[1][1] == type1:
            possible_adds.append((1,1))
        if applicant[0][1] == type1:
            possible_adds.append((0,1))
        if possible_adds:
            return possible_adds
        else:
            return 100
            
    elif  current_position[0] == len(applicant)-1 and current_position[1] == 0:
        if applicant[current_position[0]-1][current_position[1]] == type1:
            possible_adds.append((current_position[0]-1,current_position[1]))
        if applicant[current_position[0]-1][current_position[1]+1] == type1:
            possible_adds.append((current_position[0]-1,current_position[1]+1))
        if applicant[current_position[0]][current_position[1]+1] == type1:
            possible_adds.append((current_position[0],current_position[1]))
        if possible_adds:
            return possible_adds
        else:
            return 100
        
    elif  current_position[0] == len(applicant)-1 and current_position[1] == len(applicant)-1:
        if applicant[current_position[0]-1][current_position[1]] == type1:
            possible_adds.append((current_position[0]-1,current_position[1]))
        if applicant[current_position[0]-1][current_position[1]-1] == type1:
            possible_adds.append((current_position[0]-1,current_position[1]-1))
        if applicant[current_position[0]][current_position[1]-1] == type1:
            possible_adds.append((current_position[0],current_position[1]-1))
        if possible_adds:
            return possible_adds
        else:
            return 100
            
    elif  current_position[0] == 0 and current_position[1] == len(applicant)-1:
        if applicant[current_position[0]+1][current_position[1]] == type1:
            possible_adds.append((current_position[0]+1,current_position[1]))
        if applicant[current_position[0]+1][current_position[1]-1] == type1:
            possible_adds.append((current_position[0]+1,current_position[1]-1))
        if applicant[current_position[0]][current_position[1]-1] == type1:
            possible_adds.append((current_position[0],current_position[1]-1))
        if possible_adds:
           return possible_adds
        else:
            return 100
    #sides 
    elif  current_position[0] == 0:
        if applicant[current_position[0]+1][current_position[1]] == type1:
            possible_adds.append((current_position[0]+1,current_position[1]))
        if applicant[current_position[0]+1][current_position[1]-1] == type1:
            possible_adds.append((current_position[0]+1,current_position[1]-1))
        if applicant[current_position[0]][current_position[1]-1] == type1:
            possible_adds.append((current_position[0],current_position[1]-1))
        if applicant[current_position[0]][current_position[1]+1] == type1:
            possible_adds.append((current_position[0],current_position[1]+1))
        if applicant[current_position[0]+1][current_position[1]+1] == type1:
            possible_adds.append((current_position[0]+1,current_position[1]+1))
        if possible_adds:
            return possible_adds
        else:
            return 100
            
    elif  current_position[0] == len(applicant)-1:
        if applicant[current_position[0]-1][current_position[1]] == type1:
            possible_adds.append((current_position[0]-1,current_position[1]))
        if applicant[current_position[0]-1][current_position[1]-1] == type1:
            possible_adds.append((current_position[0]-1,current_position[1]-1))
        if applicant[current_position[0]][current_position[1]-1] == type1:
            possible_adds.append((current_position[0],current_position[1]-1))
        if applicant[current_position[0]][current_position[1]+1] == type1:
            possible_adds.append((current_position[0],current_position[1]+1))
        if applicant[current_position[0]-1][current_position[1]+1] == type1:
            possible_adds.append((current_position[0]-1,current_position[1]+1))
        if possible_adds:
            return possible_adds
        else:
            return 100
            
    elif  current_position[1] == len(applicant)-1:
        if applicant[current_position[0]][current_position[1]-1] == type1:
            possible_adds.append((current_position[0],current_position[1]-1))
        if applicant[current_position[0]-1][current_position[1]-1] == type1:
            possible_adds.append((current_position[0]-1,current_position[1]-1))
        if applicant[current_position[0]+1][current_position[1]-1] == type1:
            possible_adds.append((current_position[0]+1,current_position[1]-1))
        if applicant[current_position[0]+1][current_position[1]] == type1:
            possible_adds.append((current_position[0]+1,current_position[1]))
        if applicant[current_position[0]-1][current_position[1]] == type1:
            possible_adds.append((current_position[0]-1,current_position[1]))
        if possible_adds:
            return possible_adds
        else:
            return 100
    elif  current_position[1] == 0:
        if applicant[current_position[0]][current_position[1]+1] == type1:
            possible_adds.append((current_position[0],current_position[1]+1))
        if applicant[current_position[0]-1][current_position[1]+1] == type1:
            possible_adds.append((current_position[0]-1,current_position[1]+1))
        if applicant[current_position[0]+1][current_position[1]+1] == type1:
            possible_adds.append((current_position[0]+1,current_position[1]+1))
        if applicant[current_position[0]+1][current_position[1]] == type1:
            possible_adds.append((current_position[0]+1,current_position[1]))
        if applicant[current_position[0]-1][current_position[1]] == type1:
            possible_adds.append((current_position[0]-1,current_position[1]))
        if possible_adds:
            return possible_adds
        else:
            return 100
    #interior
    else:
        if applicant[current_position[0]+1][current_position[1]+1] == type1:
            possible_adds.append((current_position[0]+1,current_position[1]+1))
        if applicant[current_position[0]][current_position[1]+1] == type1:
            possible_adds.append((current_position[0],current_position[1]+1))
        if applicant[current_position[0]+1][current_position[1]] == type1:
            possible_adds.append((current_position[0]+1,current_position[1]))
        if applicant[current_position[0]-1][current_position[1]-1] == type1:
            possible_adds.append((current_position[0]-1,current_position[1]-1))
        if applicant[current_position[0]-1][current_position[1]] == type1:
            possible_adds.append((current_position[0]-1,current_position[1]))
        if applicant[current_position[0]][current_position[1]-1] == type1:
            possible_adds.append((current_position[0],current_position[1]-1))
        if applicant[current_position[0]-1][current_position[1]+1] == type1:
            possible_adds.append((current_position[0]-1,current_position[1]+1))
        if applicant[current_position[0]+1][current_position[1]-1] == type1:
            possible_adds.append((current_position[0]+1,current_position[1]-1))
        if possible_adds:
            return possible_adds
        else:
            return 100
            
def create_connections(mat):
    connections = []
    for i in range(0,len(mat)):
        for j in range(0,len(mat)):
            current_connect = all_connections(mat,[i,j])
            connections.append([mat[i][j],(i,j),current_connect])
            #print(current_connect)
    return connections 

def still_all_districts(new_mutant,districts):
    check = [i for i in range(1,districts)]
    missing = []
    mis = True
    for i in range(0,len(check)-1):
        for j in range(0,len(new_mutant)-1):
            if check[i] in new_mutant[j]:
                mis = False
        missing.append(mis)
    if True in missing:
        return False
    else:
        return True
                    
def _mutatable_positions(applicant,current_position):
    possible_adds = []
    type1 = applicant[current_position[0]][current_position[1]]
    #corners
    if  current_position[0] == 0 and current_position[1] == 0:
        if applicant[1][0] != type1:
            possible_adds.append((1,0))
        if applicant[1][1] != type1:
            possible_adds.append((1,1))
        if applicant[0][1] != type1:
            possible_adds.append((0,1))
        if possible_adds:
            return possible_adds
        else:
            return 100
            
    elif  current_position[0] == len(applicant)-1 and current_position[1] == 0:
        if applicant[current_position[0]-1][current_position[1]] != type1:
            possible_adds.append((current_position[0]-1,current_position[1]))
        if applicant[current_position[0]-1][current_position[1]+1] != type1:
            possible_adds.append((current_position[0]-1,current_position[1]+1))
        if applicant[current_position[0]][current_position[1]+1] != type1:
            possible_adds.append((current_position[0],current_position[1]))
        if possible_adds:
            return possible_adds
        else:
            return 100
        
    elif  current_position[0] == len(applicant)-1 and current_position[1] == len(applicant)-1:
        if applicant[current_position[0]-1][current_position[1]] != type1:
            possible_adds.append((current_position[0]-1,current_position[1]))
        if applicant[current_position[0]-1][current_position[1]-1] != type1:
            possible_adds.append((current_position[0]-1,current_position[1]-1))
        if applicant[current_position[0]][current_position[1]-1] != type1:
            possible_adds.append((current_position[0],current_position[1]-1))
        if possible_adds:
            return possible_adds
        else:
            return 100
            
    elif  current_position[0] == 0 and current_position[1] == len(applicant)-1:
        if applicant[current_position[0]+1][current_position[1]] != type1:
            possible_adds.append((current_position[0]+1,current_position[1]))
        if applicant[current_position[0]+1][current_position[1]-1] != type1:
            possible_adds.append((current_position[0]+1,current_position[1]-1))
        if applicant[current_position[0]][current_position[1]-1] != type1:
            possible_adds.append((current_position[0],current_position[1]-1))
        if possible_adds:
           return possible_adds
        else:
            return 100
    #sides 
    elif  current_position[0] == 0:
        if applicant[current_position[0]+1][current_position[1]] != type1:
            possible_adds.append((current_position[0]+1,current_position[1]))
        if applicant[current_position[0]+1][current_position[1]-1] != type1:
            possible_adds.append((current_position[0]+1,current_position[1]-1))
        if applicant[current_position[0]][current_position[1]-1] != type1:
            possible_adds.append((current_position[0],current_position[1]-1))
        if applicant[current_position[0]][current_position[1]+1] != type1:
            possible_adds.append((current_position[0],current_position[1]+1))
        if applicant[current_position[0]+1][current_position[1]+1] != type1:
            possible_adds.append((current_position[0]+1,current_position[1]+1))
        if possible_adds:
            return possible_adds
        else:
            return 100
            
    elif  current_position[0] == len(applicant)-1:
        if applicant[current_position[0]-1][current_position[1]] != type1:
            possible_adds.append((current_position[0]-1,current_position[1]))
        if applicant[current_position[0]-1][current_position[1]-1] != type1:
            possible_adds.append((current_position[0]-1,current_position[1]-1))
        if applicant[current_position[0]][current_position[1]-1] != type1:
            possible_adds.append((current_position[0],current_position[1]-1))
        if applicant[current_position[0]][current_position[1]+1] != type1:
            possible_adds.append((current_position[0],current_position[1]+1))
        if applicant[current_position[0]-1][current_position[1]+1] != type1:
            possible_adds.append((current_position[0]-1,current_position[1]+1))
        if possible_adds:
            return possible_adds
        else:
            return 100
            
    elif  current_position[1] == len(applicant)-1:
        if applicant[current_position[0]][current_position[1]-1] != type1:
            possible_adds.append((current_position[0],current_position[1]-1))
        if applicant[current_position[0]-1][current_position[1]-1] != type1:
            possible_adds.append((current_position[0]-1,current_position[1]-1))
        if applicant[current_position[0]+1][current_position[1]-1] != type1:
            possible_adds.append((current_position[0]+1,current_position[1]-1))
        if applicant[current_position[0]+1][current_position[1]] != type1:
            possible_adds.append((current_position[0]+1,current_position[1]))
        if applicant[current_position[0]-1][current_position[1]] != type1:
            possible_adds.append((current_position[0]-1,current_position[1]))
        if possible_adds:
            return possible_adds
        else:
            return 100
    elif  current_position[1] == 0:
        if applicant[current_position[0]][current_position[1]+1] != type1:
            possible_adds.append((current_position[0],current_position[1]+1))
        if applicant[current_position[0]-1][current_position[1]+1] != type1:
            possible_adds.append((current_position[0]-1,current_position[1]+1))
        if applicant[current_position[0]+1][current_position[1]+1] != type1:
            possible_adds.append((current_position[0]+1,current_position[1]+1))
        if applicant[current_position[0]+1][current_position[1]] != type1:
            possible_adds.append((current_position[0]+1,current_position[1]))
        if applicant[current_position[0]-1][current_position[1]] != type1:
            possible_adds.append((current_position[0]-1,current_position[1]))
        if possible_adds:
            return possible_adds
        else:
            return 100
    #interior
    else:
        if applicant[current_position[0]+1][current_position[1]+1] != type1:
            possible_adds.append((current_position[0]+1,current_position[1]+1))
        if applicant[current_position[0]][current_position[1]+1] != type1:
            possible_adds.append((current_position[0],current_position[1]+1))
        if applicant[current_position[0]+1][current_position[1]] != type1:
            possible_adds.append((current_position[0]+1,current_position[1]))
        if applicant[current_position[0]-1][current_position[1]-1] != type1:
            possible_adds.append((current_position[0]-1,current_position[1]-1))
        if applicant[current_position[0]-1][current_position[1]] != type1:
            possible_adds.append((current_position[0]-1,current_position[1]))
        if applicant[current_position[0]][current_position[1]-1] != type1:
            possible_adds.append((current_position[0],current_position[1]-1))
        if applicant[current_position[0]-1][current_position[1]+1] != type1:
            possible_adds.append((current_position[0]-1,current_position[1]+1))
        if applicant[current_position[0]+1][current_position[1]-1] != type1:
            possible_adds.append((current_position[0]+1,current_position[1]-1))
        if possible_adds:
            return possible_adds
        else:
            return 100   


# read in data
filename = raw_input()
# convert text to matrix
current = []
with open(filename,"r") as f:
    text = f.readlines()
    for i in range(0,len(text)):
        text[i] = text[i].split(' ')
        s = len(text[i])-1
        text[i][s] = text[i][s].split('\n')
        text[i][s] = text[i][s][0]
#print(text)
'''
intitializations
'''
#print(text)
#number of districts
sdistricts = len(text)
#optimal percentages
sPercentage = 0 
for j in range(0,len(text)):
    for i in range(0,len(text)):
        if text[0][i] == 'D':
            sPercentage = sPercentage + 1 
sPercentage = sPercentage/(len(text)**2)
#print(sPercentage)
# size of initial pool
pool = 1200
districts = len(text)
genepool = Initialize(pool, districts, len(text))
#print_genepool(genepool)
n = 60
iterations = 2
#i = iteration
i = 0
sSize = len(text)+2
while i < iterations:
    #print("epoch", i)
    #pick best from current pool
    new_epoch = find_best(genepool,n,text,sdistricts)
    #print(new_epoch)
    new_pool = list([genepool[int(new_epoch[q][0])] for q in range(0,n-10)])
    genepool = new_pool
    genepool = Initialize(pool, sSize, len(text))
    next_pool = list(genepool)
    #print_genepool(next_pool)
    #genepool.append(next_pool)
    #print_genepool(genepool)
    #genepool.append(nextpool)
    i = i+1
best_solution = find_best(genepool,2,text,sdistricts)
best_solution = best_solution[0][0]
best_solution = genepool[best_solution]
#printmat(best_solution)
dist = partydev(best_solution, text, districts+1)
D = Finalfitness(best_solution,text,sdistricts)
R = 1-Finalfitness(best_solution,text,sdistricts)
numberD = Finalfitnessoth(best_solution,text,sdistricts)
numberR = sdistricts - Finalfitnessoth(best_solution,text,sdistricts)
#print(dist)
print('Party division in population:')
print('*************************************')
print('R: ', R, '%')
print('D: ', D, '%')
print('*************************************')
print('Number of districts with a majority for each party:')
print('*************************************')
print('R: ', numberR)
print('D: ', numberD)
print('*************************************')
print('Locations assigned to each district:')
print('*************************************')
for i in range(0,districts):
    print 'District ', i+1,': ', dist[i]
print('*************************************')
print('Applied GA:')
print('*************************************')
print('*************************************')
print('Number of search states explored: ', iterations*100+pool)
print('*************************************')
    