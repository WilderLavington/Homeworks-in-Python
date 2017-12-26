# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 16:15:30 2016

@author: wilder
"""

import string

def viterbi(obs, states, start_p, trans_p, emit_p):
    """ INITIALIZE LIST OF DICTIONARIES """
    V = [{}]
    """ V_1,X_1  = EMISSION PROB * STARTING PROB """
    for st in states:
        """ EACH STATE st IS KEY TO DICTIONARY WITH INITIAL DISTRIBUTION V_1,X_1,st """
        """ V is the list of all such dictionaries at time 1 """
        """ there is only one prob so there is no need to find a maximum for each state"""
        #print(start_p[st])
        #print(emit_p[st][obs[0]])
        V[0][st] = {"prob": start_p[st] * emit_p[st][obs[0]], "prev": None}
        """ for example this gives the prob of fever given normal and prob healthy given normal"""
    # Run Viterbi when t > 0
    """ V_t,X_t  = MAX(EMISSION PROB * CURRENT PROB *  V_t-1,X_t-1) """
    for t in range(1, len(obs)):
        """ APPEND NEXT TIMESTEP, DICTIONARY V_t,X_t """
        V.append({})
        for st in states:
            """ EACH STATE st IS KEY TO DICTIONARY WITH INITIAL DISTRIBUTION V_t,X_t,st """
            """ V is the list of all such dictionaries at time up to time t """
            """ finds all state probabilities """
            """ find state with max probabilty in final time step """
            max_tr_prob = max(V[t-1][prev_st]["prob"]*trans_p[prev_st][st] for prev_st in states)
            """ go back through all previous states that maximize the likelyhood for each timestep """
            for prev_st in states:
                if V[t-1][prev_st]["prob"] * trans_p[prev_st][st] == max_tr_prob:
                    """Uses conditional info from timestep in front of it to calculate maximum probability before it."""
                    max_prob = max_tr_prob * emit_p[st][obs[t]]
                    """ calculate emission probability at stat t, then store it """
                    V[t][st] = {"prob": max_prob, "prev": prev_st}
                    break
    #for line in dptable(V):
        #print line
    
    opt = []
    # The highest probability
    """ find the maximum  """
    max_prob = max(value["prob"] for value in V[-1].values())
    previous = None
    # Get most probable state and its backtrack
    for st, data in V[-1].items():
        #print st, data
        if data["prob"] == max_prob:
            opt.append(st)
            previous = st
            break
    # Follow the backtrack till the first observation
    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]

    print 'The steps of states are ' + ' '.join(opt) + ' with highest probability of %s' % max_prob

def dptable(V):
     # Print a table of steps from dictionary
     yield " ".join(("%12d" % i) for i in range(len(V)))
     for state in V[0]:
         yield "%.7s: " % state + " ".join("%.7s" % ("%f" % v[state]["prob"]) for v in V)
    
if __name__ == "__main__":
    """ 
    example:
    states = ('Bull', 'Bear')
    observations = ('Low', 'Med', 'High')
    start_probability = {'Bull': 0.6, 'Bear': 0.4}
    transition_probability = {
        'Bull' : {'Bull': 0.7, 'Bear': 0.3},
        'Bear' : {'Bull': 0.4, 'Bear': 0.6}
    }

    emission_probability = {
        'Bull' : {'Low': 0.5, 'Med': 0.4, 'High': 0.1},
        'Bear' : {'Low': 0.1, 'Med': 0.3, 'High': 0.6}
    }
    viterbi(observations, states, start_probability, transition_probability, emission_probability)
    """
    
    """
    PARSE DATA AND CREATE SENTANCES
    """
    lines = []
    sentences = []
    nextline = 0
    with open("penntree.tag", "r") as sentences:
        lines.append(("START START"))
        for line in sentences:
            if(line=="\n"):
                nextline = 1
            elif nextline == 1:
                lines.append(("START START"))
                nextline = 0
            else:
                line = string.replace(line,"\t", " ")
                #print(line)
                lines.append(line.rstrip())
    """
    FIND TRANSITION PROBABILITIES of TAGS, STORE IN DICTIONARY (of dictionaries)
    """
    #make a list of tuples
    new_list = []
    for ii in range(0,len(lines)):
        new_list.append(tuple(lines[ii].split(" ")))
    # make a dictionary for the total number of instance of each tag
    total_tags = {}
    # poplulate dictionary
    for ii in range(0,len(new_list)):
        if new_list[ii][1] not in total_tags:
            total_tags[new_list[ii][1]] = 0
            counter = 0
            for jj in range(0,len(new_list)):
                if new_list[ii][1] == new_list[jj][1]:
                    counter = counter+1         
            total_tags[new_list[ii][1]] = counter
    # find the transtion probability with dict of dict
    transition_prob = {}
    #create dictionary 
    for start in total_tags:
        transition_prob[start] = {}
        for end in total_tags:
            transition_prob[start][end] = 0
    for ii in range(0,len(new_list)-1):
        transition_prob[new_list[ii][1]][new_list[ii+1][1]] = transition_prob[new_list[ii][1]][new_list[ii+1][1]]+1
    for start in total_tags:
        for end in total_tags:
            transition_prob[start][end] = float(transition_prob[start][end]) #/total_tags[start]
            transition_prob[start][end] = float(transition_prob[start][end])/float(total_tags[start])
    """
    FIND EMISSION PROBABILITIES of TAGS, STORE IN DICTIONARY (of dictionaries)
    """                
    #find the number of time you have a word and a tag / number of times you see a tag
    emission_prob = {}
    #create dictionary of words
    for match in total_tags:
        if  match not in emission_prob:
            emission_prob[match] = {}
            for ii in range(0,len(new_list)):
                if  new_list[ii][0] not in emission_prob[match]:
                    emission_prob[match][new_list[ii][0]] = 0
    #calculate emmision prob
    for ii in range(0,len(new_list)):  
        emission_prob[new_list[ii][1]][new_list[ii][0]] = emission_prob[new_list[ii][1]][new_list[ii][0]]+1
        #print(emission_prob[new_list[ii][1]][new_list[ii][0]])
    for word in emission_prob:
        for tag in emission_prob[word]:
            emission_prob[word][tag] = float(emission_prob[word][tag])/float(total_tags[word])
    """
    TAKE USER INPUT (SENTANCE) PREDICT TAG FOR EACH WORD IN SENTENCE
    """    
    # the states are the possible tags 
    states = ()
    for key in total_tags:
        states = states+(key,)
    #the observations are the words in the sentance
    from sys import argv
    #sentence = str(input('input sentence:'))
    sentence = ' '.join(argv[1:])
    #sentence = 'Can you walk the walk and talk the talk?'
    #parse into tuple 
    sentence = sentence.split(' ')
    """
    Split up the final word if there is a ? or . or ,
    """  
    last_word = list(sentence[-1])
    if len(last_word) > 1:
        if '.' in last_word:
            new_word = last_word[0:len(last_word)-1]
            ending  = last_word[-1]
            new_word = last_word[0:len(last_word)-1]
            ending  = last_word[-1]
            sentence = sentence[0:len(sentence)-1]
            sentence.append(''.join(new_word))
            sentence.append(ending)
        elif '?' in last_word:
            new_word = last_word[0:len(last_word)-1]
            ending  = last_word[-1]
            sentence = sentence[0:len(sentence)-1]
            sentence.append(''.join(new_word))
            sentence.append(ending)
        elif ',' in last_word:
            new_word = last_word[0:len(last_word)-1]
            ending  = last_word[-1]
            sentence = sentence[0:len(sentence)-1]
            sentence.append(''.join(new_word))
            sentence.append(ending)
        else:
            sentence = sentence
    """
    Run viterbi algorithm
    """
    observations = sentence
    # give initial probabilities for viterbi
    start_probability = transition_prob["START"]
    transition_probability = transition_prob
    emission_probability = emission_prob
    print('\n')
    print "Sentence: ", ' '.join(argv[1:])
    print("States Transitions:")
    viterbi(observations, states, start_probability, transition_probability, emission_probability)
