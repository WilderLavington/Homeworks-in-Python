#Artificial Intelligence: A Modern Approach

# Search AIMA
#AIMA Python file: mdp.py

'''
QUESTIONS:
What action is	assigned in the	terminal states?

Within the MDP class, the action function checks whether or not the state
argument is one of the terminal states that was initialized. If the state
is not a terminal state, it returns the list of possible actions for that state.
If the state is terminal, it returns NONE.

Where are the transition probabilities defined in the	program,
and what are those	probabilities?

The transition probabilities are defined in the GridMDP class,
under the T function. The probabilities are as follows: the probability of
taking the chosen action is .8, the probability of not taking the chosen
action and instead going right is .1, and the probability of not taking
the chosen action and going right is .1.

What function needs to	be called to run value iteration?

To run value iteration, first you must run the GridMDP function to create a
usable grid to run the program on. the function also requires value iteration. 

When you run value_iteration, on the MDP provided, the results are stored 
in a variable called myMDP, What is the utility of (0,1), (3, 1), and (2, 2)?

The utility values for each of the cooridinates is as follows: U((0,1)) = 62.9714499917616,
U((3,1)) = 92.71881434487076, and U((2,2)) is not within the dictionary because the space
that is associated to that coordinate is not accessible. Because it is not accessible,
it is never assigned a utility because it does not have one.

How are actions represented, and what are the possible	actions for 
each state in the program?

The actions for each state are represented as a tuple with values that are either 0, 1, or
-1. Where 0 means to stay on that axis, and 1 and -1 means that you will move left, right,
up or down depending on which element of the tuple it is contained in. 
The possible actions for each state as defined by the prompt are movement left, 
right, up, or down. That is up to the constraints of the current state; if 
the state is surrounded by a wall, the possible movement is restricted.

'''

"""Markov Decision Processes (Chapter 17)

First we define an MDP, and the special case of a GridMDP, in which
states are laid out in a 2-dimensional grid.  We also represent a policy
as a dictionary of {state:action} pairs, and a Utility function as a
dictionary of {state:number} pairs.  We then define the value_iteration
and policy_iteration algorithms."""

from utils import *
import random as r
import matplotlib.pyplot as plt

class MDP:
    """A Markov Decision Process, defined by an initial state, transition model,
    and reward function. We also keep track of a gamma value, for use by
    algorithms. The transition model is represented somewhat differently from
    the text.  Instead of T(s, a, s') being  probability number for each
    state/action/state triplet, we instead have T(s, a) return a list of (p, s')
    pairs.  We also keep track of the possible states, terminal states, and
    actions for each state. [page 615]"""

    def __init__(self, init, actlist, terminals, gamma=.9):
        update(self, init=init, actlist=actlist, terminals=terminals,
               gamma=gamma, states=set(), reward={})

    def R(self, state):
        "Return a numeric reward for this state."
        return self.reward[state]

    def T(state, action):
        """Transition model.  From a state and an action, return a list
        of (result-state, probability) pairs."""
        abstract

    def actions(self, state):
        """Set of actions that can be performed in this state.  By default, a
        fixed list of actions, except for terminal states. Override this
        method if you need to specialize by state."""
        if state in self.terminals:
            return [None]
        else:
            return self.actlist

class GridMDP(MDP):
    """A two-dimensional grid MDP, as in [Figure 17.1].  All you have to do is
    specify the grid as a list of lists of rewards; use None for an obstacle
    (unreachable state).  Also, you should specify the terminal states.
    An action is an (x, y) unit vector; e.g. (1, 0) means move east."""
    def __init__(self, grid, terminals, gamma, jump , init=(0, 0)):
        grid.reverse() ## because we want row 0 on bottom, not on top
        if jump == 1:
            current = jump_orientations
        else:
            current = orientations
        MDP.__init__(self, init, actlist=current,
                     terminals=terminals, gamma=gamma)
        update(self, grid=grid, rows=len(grid), cols=len(grid[0]))
        for x in range(self.cols):
            for y in range(self.rows):
                self.reward[x, y] = grid[y][x]
                if grid[y][x] is not None:
                    self.states.add((x, y))

    def T(self, state, action):
        if action in sublist:
            return [(0.5, self.go(state, action)),
                    (0.5, self.go(state, jump_fail(action)))]
        if action == None:
            return [(0.0, state)]
        else:
            return [(0.8, self.go(state, action)),
                    (0.1, self.go(state, turn_right(action))),
                    (0.1, self.go(state, turn_left(action)))]

    def go(self, state, direction):
        "Return the state that results from going in this direction."
        state1 = vector_add(state, direction)
        return if_(state1 in self.states, state1, state)

    def to_grid(self, mapping):
        """Convert a mapping from (x, y) to v into a [[..., v, ...]] grid."""
        return list(reversed([[mapping.get((x,y), None)
                               for x in range(self.cols)]
                              for y in range(self.rows)]))

    def to_arrows(self, policy):
        chars = {(1, 0):'>', (0, 1):'^', (-1, 0):'<', (0, -1):'v', None: '.',
                 (2, 0):'>>', (0, 2):'^^', (-2, 0):'<<', (0, -2):'vv',}
        return self.to_grid(dict([(s, chars[a]) for (s, a) in policy.items()]))


def value_iteration(mdp, epsilon=0.001):
    "Solving an MDP by value iteration. [Fig. 17.4]"
    U1 = dict([(s, 0) for s in mdp.states])
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    while True:
        U = U1.copy()
        delta = 0
        for s in mdp.states:
            U1[s] = R(s) + gamma * max([sum([p * U[s1] for (p, s1) in T(s, a)])
                                        for a in mdp.actions(s)])
            delta = max(delta, abs(U1[s] - U[s]))
        if delta < epsilon * (1 - gamma) / gamma:
             return U

def best_policy(mdp, U):
    """Given an MDP and a utility function U, determine the best policy,
    as a mapping from state to action. (Equation 17.4)"""
    pi = {}
    for s in mdp.states:
        pi[s] = argmax(mdp.actions(s), lambda a:expected_utility(a, s, U, mdp))
    return pi

def expected_utility(a, s, U, mdp):
    "The expected utility of doing a in state s, according to the MDP and U."
    return sum([p * U[s1] for (p, s1) in mdp.T(s, a)])


def policy_iteration(mdp):
    "Solve an MDP by policy iteration [Fig. 17.7]"
    U = dict([(s, 0) for s in mdp.states])
    pi = dict([(s, random.choice(mdp.actions(s))) for s in mdp.states])
    while True:
        U = policy_evaluation(pi, U, mdp)
        unchanged = True
        for s in mdp.states:
            a = argmax(mdp.actions(s), lambda a: expected_utility(a,s,U,mdp))
            if a != pi[s]:
                pi[s] = a
                unchanged = False
        if unchanged:
            return pi

def policy_evaluation(pi, U, mdp, k=20):
    """Return an updated utility mapping U from each state in the MDP to its
    utility, using an approximation (modified policy iteration)."""
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    for i in range(k):
        for s in mdp.states:
            U[s] = R(s) + gamma * sum([p * U[s1] for (p, s1) in T(s, pi[s])])
    return U
    
def print_mat(mat):
    for i in range(0,len(mat)):
        print(mat[i])
        
def Traverse_grid(grid, policy_arrows, moves):
    """Traverse grid based on each policy using transtion probabilities"""
    current = (len(grid)-1,0)
    #print_mat(policy_arrows)
    #print_mat(grid)
    score  = 0
    for ii in range(0,moves):
        #simulate movement error
        simulate = r.uniform(0, 1)
        sublist = ['vv', '^^', '<<', '>>']
        if policy_arrows[current[0]][current[1]] in sublist:
            if simulate <= .5:
                action = 0
            if simulate > .5:
                action = 1
        else:
            if simulate <= .1:
                action = -1
            elif simulate <= .2:
                action = 1
            else:
                action = 0
        #pick move
        '''
        SINGLE MOVE TRANSITIONS 
        '''
        score = score + grid[current[0]][current[1]]
        if policy_arrows[current[0]][current[1]] == '>':
            #turns right
            if action == 1:
                if current[0] != len(grid)-1:
                    if grid[current[0]+1][current[1]] != None:
                        temp = current[0]+1
                        current = (temp,current[1])
            #turns left
            if action == -1:
                if current[0] != 0:
                    if grid[current[0]-1][current[1]] != None:
                        temp = current[0]-1
                        current = (temp,current[1])
            #goes straight
            if action == 0:
                if current[1] != len(grid[1])-1:
                    if grid[current[0]][current[1]+1] != None:
                        temp = current[1]+1
                        current = (current[0],temp)
        elif policy_arrows[current[0]][current[1]] == '<':
            #turns left
            if action == -1:
                if current[0] != len(grid)-1:
                    if grid[current[0]+1][current[1]] != None:
                        temp = current[0]+1
                        current = (temp,current[1])
            #turns right
            if action == 1:
                if current[0] != 0:
                    if grid[current[0]-1][current[1]] != None:
                        temp = current[0]-1
                        current = (temp,current[1])
            #moves forward
            if action == 0:
                if current[1] != 0:
                    if grid[current[0]][current[1]-1] != None:
                        temp = current[1]-1
                        current = (current[0],temp)
        elif policy_arrows[current[0]][current[1]] == 'v':
            #goes down
            if action == 0:
                if current[0] != len(grid)-1:
                    if grid[current[0]+1][current[1]] != None:
                        temp = current[0]+1
                        current = (temp,current[1])
            #goes right (my left)
            if action == 1:
                if current[1] != 0:
                    if grid[current[0]][current[1]-1] != None:
                        temp = current[1]-1
                        current = (current[0],temp)
            #goes left (my right)
            if action == -1:
                if current[1] != len(grid[1])-1:
                    if grid[current[0]][current[1]+1] != None:
                        temp = current[1]+1
                        current = (current[0],temp)
        elif policy_arrows[current[0]][current[1]] == '^':
            #goes up
            if action == 0:
                if current[0] != 0:
                    if grid[current[0]-1][current[1]] != None:
                        temp = current[0]-1
                        current = (temp,current[1])
            #goes right
            if action == 1:
                if current[1] != len(grid[1])-1:
                    if grid[current[0]][current[1]+1] != None:
                        temp = current[1]+1
                        current = (current[0],temp)
            #goes left
            if action == -1:
                if current[1] != 0:
                    if grid[current[0]][current[1]-1] != None:
                        temp = current[1]-1
                        current = (current[0],temp)
        elif policy_arrows[current[0]][current[1]] == 'vv':
            #jumps
            if action == 0:
                if current[0] != len(grid)-2:
                    if grid[current[0]+2][current[1]] != None:
                        temp = current[0]+2
                        current = (temp,current[1])
            #fails
            if action == 1:
                pass
            
        elif policy_arrows[current[0]][current[1]] == '^^':
            #jumps
            if action == 0:
                if current[0] != 1:
                    if grid[current[0]-2][current[1]] != None:
                        temp = current[0]-2
                        current = (temp,current[1])
            #fails
            if action == 1:
                pass
                        
        elif policy_arrows[current[0]][current[1]] == '>>':
            #jumps
            if action == 0:
                if current[1] != len(grid)-2:
                    if grid[current[0]][current[1]+2] != None:
                        temp = current[1]+2
                        current = (current[0],temp)
                        
            #fails
            if action == 1:
                pass
            
        elif policy_arrows[current[0]][current[1]] == '<<':
            #jump
            if action == 0:
                if current[1] != 1:
                    if grid[current[0]-2][current[1]] != None:
                        temp = current[1]-2
                        current = (current[0],temp)
            #fail
            if action == 1:
                pass
            
        elif policy_arrows[current[0]][current[1]] == '.':
            score = score + grid[current[0]][current[1]]
            return score, current, ii
            
    return score, current, ii
#example 
myMDP = GridMDP([[-0.04, -0.04, -0.04, +1],
                     [-0.04, None,  -0.04, -1],
                     [-0.04, -0.04, -0.04, -0.04]],
                    terminals=[(3,1),(3,2)], gamma = .9, jump = 0)
U = value_iteration(myMDP)
'''
P1:
Living reward
'''
def Sim_living_reward(jump1):
    scores_positions = []
    import copy
    matrix = [[0, 0, 0, 0, -1, 0, -1, -1, 0, 50],
              [None, None, -1, -1, 0, -.5, None, 0, None, 0],
              [0, 0, 0, 0, 0, -.5, None, 0, 0, 0],
              [None, 2, None, None, None, -.5, 0, 2, None, 0],
              [None, 0, 0, 0, 0, None, -1, -.5, -1, 0],
              [0, -.5, None, 0, 0, None, 0, 0, None, 0],
              [0, -.5, None, 0, -1, None, 0, -1, None, None],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    temporary = copy.deepcopy(matrix)  
    # shift living reward (negative) 
    for k in range(0,100):
        mat_neg = copy.deepcopy(matrix)
        for i in range(0,len(matrix)):
            for j in range(0,len(matrix)):
                if matrix[i][j] == 0:
                    mat_neg[i][j] = -1+k*0.01
        temp = copy.deepcopy(mat_neg)
        myMDP_new = GridMDP(mat_neg,terminals=[(9,7)], gamma = .9, jump = jump1)
        U2 = policy_iteration(myMDP_new)
        U2 = myMDP_new.to_arrows(U2)
        s = Traverse_grid(temp, U2, 1000)
        del mat_neg
        del temp
        scores_positions.append([round(-1+k*0.01,2),s[2]])
    #original
    myMDP_new = GridMDP(matrix,terminals=[(9,7)], gamma = .9, jump = jump1)
    U2 = policy_iteration(myMDP_new)
    U2 = myMDP_new.to_arrows(U2)
    del matrix
    matrix = copy.deepcopy(temporary) 
    s = Traverse_grid(matrix, U2, 1000)
    scores_positions.append([0,s[2]])
    # shift living reward (posative)
    for k in range(0,100):
        mat_pos = copy.deepcopy(matrix)
        for i in range(0,len(matrix)):
            for j in range(0,len(matrix)):
                if matrix[i][j] == 0:
                    mat_pos[i][j] = k*0.01
        temp = copy.deepcopy(mat_pos)
        myMDP_new = GridMDP(mat_pos,terminals=[(9,7)], gamma = .9, jump = jump1)
        U2 = policy_iteration(myMDP_new)
        U2 = myMDP_new.to_arrows(U2)
        s = Traverse_grid(temp, U2, 1000)
        del mat_pos
        del temp
        scores_positions.append([round(k*0.01,2),s[2]])
    # plot the simulation
    #print('Current arrow matrix')
    #print_mat(U2)
    x = [scores_positions[i][0] for i in range(0,len(scores_positions))]
    y = [scores_positions[i][1] for i in range(0,len(scores_positions))]
    plt.plot(x,y)
    plt.title('Horse 2 Apple Simulation')
    plt.ylabel('Number of Moves to get Apple')
    plt.xlabel('living reward')
    plt.show()
#run it
print('')
print('')
print('Simulations without Jumps: Living Reward') 
Sim_living_reward(0)
'''
P2:
Changing the Gamma
'''
def Gamma_sim(jump1):
    gamma_values = []
    matrix = [[0, 0, 0, 0, -1, 0, -1, -1, 0, 50],
              [None, None, -1, -1, 0, -.5, None, 0, None, 0],
              [0, 0, 0, 0, 0, -.5, None, 0, 0, 0],
              [None, 2, None, None, None, -.5, 0, 2, None, 0],
              [None, 0, 0, 0, 0, None, -1, -.5, -1, 0],
              [0, -.5, None, 0, 0, None, 0, 0, None, 0],
              [0, -.5, None, 0, -1, None, 0, -1, None, None],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    temporary = copy.deepcopy(matrix)
    temporary1 = copy.deepcopy(matrix)
    myMDP_new = GridMDP(temporary1,terminals=[(9,7)], gamma = .9, jump = jump1)
    U = policy_iteration(myMDP_new)
    P = myMDP_new.to_arrows(U)
    print_mat(P)
    for i in range(0,90):
        gm = round(.01*i+.1,2)
        #print_mat(matrix)
        myMDP_new = GridMDP(matrix,terminals=[(9,7)], gamma = gm, jump = jump1)
        U = policy_iteration(myMDP_new)
        P = myMDP_new.to_arrows(U)
        del matrix
        matrix = copy.deepcopy(temporary)
        s = Traverse_grid(matrix, P, 1000)
        gamma_values.append([gm,s[2]])
    #plot everything
    #print('Current arrow matrix')
    x = [gamma_values[i][0] for i in range(0,len(gamma_values))]
    y = [gamma_values[i][1] for i in range(0,len(gamma_values))]
    plt.plot(x,y)
    plt.title('Horse 2 Apple Simulation')
    plt.ylabel('Number of Moves to get Apple')
    plt.xlabel('Gamma Value')
    plt.show()
#run it
print('')
print('')
print('Simulations without Jumps: Gamma') 
Gamma_sim(0)
'''
P3:
Code Modification
'''
gamma = .9
matrix = [[0, 0, 0, 0, -1, 0, -1, -1, 0, 50],
          [None, None, -1, -1, 0, -.5, None, 0, None, 0],
          [0, 0, 0, 0, 0, -.5, None, 0, 0, 0],
          [None, 2, None, None, None, -.5, 0, 2, None, 0],
          [None, 0, 0, 0, 0, None, -1, -.5, -1, 0],
          [0, -.5, None, 0, 0, None, 0, 0, None, 0],
          [0, -.5, None, 0, -1, None, 0, -1, None, None],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]       
temporary = copy.deepcopy(matrix) 
#print(grid)
print_mat(matrix)
#print  new movement set
print('')
print('')
print('Maps')
myMDP_new = GridMDP(matrix,terminals=[(9,7)], gamma = .9, jump = 1)
U2 = policy_iteration(myMDP_new)
U2 = myMDP_new.to_arrows(U2)
print('')
print('Arrow mapping with jumps')
print_mat(U2)
del U2
#print od movement set
myMDP_new = GridMDP(temporary,terminals=[(9,7)], gamma = .9, jump = 0)
U2 = policy_iteration(myMDP_new)
U2 = myMDP_new.to_arrows(U2)
print('')
print('')
print('Arrow mapping without jumps')
print_mat(U2)
#now simulate everything with the jump transitions 
print('')
print('')
print('Simulations with Jumps: Living Reward')
Sim_living_reward(1)
print('')
print('')
print('Simulations with Jumps: Gamma')
Gamma_sim(1)






