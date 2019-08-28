# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 15:22:51 2018

@author: kunwe
"""
import numpy as np
import matplotlib.pyplot as plt
     


def find_neighbor_grids (row_num,col_num,grid):
    '''
    input: row_num and col_num: the grid location (0-based matrix)
            gird: the np matrix
    output: list of 4 two-elemnt list outputing the neighboring grids in terms of matrix indexes in a list
    
    if there is a non-movable area, the output will be "wall"
    example: find_neighbor_grids (row_num = 1,col_num = 1,
    grid = np.zeros(shape = (3,4)))
    
    output wlll be [[0, 1], [1, 0], [2, 1], [1, 2]]
    
        
    '''
    nrow = grid.shape[0]
    ncol = grid.shape[1]
    
    result = []
    #move up
    if (row_num-1 >= 0) and (grid[row_num-1,col_num] != "wall"):
        result.append([row_num-1,col_num])
    else: result.append("wall")
    
    #move left
    if (col_num-1 >= 0) and (col_num-1 <= ncol-1) and (grid[row_num,col_num-1] != "wall"):
        result.append([row_num, col_num - 1])
    else: result.append("wall")    
    
    #move down    
    if (row_num+1 >= 0) and (row_num+1 <= nrow-1) and (grid[row_num+1,col_num] != "wall"):
        result.append([row_num+1,col_num])
    else: result.append("wall")
    
    #move right
    if (col_num+1 >= 0) and (col_num+1 <= ncol-1) and (grid[row_num, col_num + 1] != "wall"):
        result.append([row_num, col_num + 1])
    else: result.append("wall")  
    
    return (result)


def to_abs_loc (location, game):
    '''
    convert the matrix location to absolute location, 0 means top left,
    1 means moving 1 grid right of the top left, etc...
    
    example: 
    convert_to_absolute_location ([0,1],np.zeros(shape = (3,4))) returns 1
    convert_to_absolute_location ([1,1],np.zeros(shape = (3,4))) returns 5
    '''
    if (location != "wall"):
        row_num = location[0]
        col_num = location[1]
        return (row_num*game.shape[1]+col_num)
    else:
        return ("wall")


def find_Tmatrix(game,terminal,P1=0.8,P2=0.1):
    '''
    This function calculate the transitional matrix based on the input reward matrix
    it will take account into terminal condition and any obstacle on the way
    
    input: game: numpy matrix with "wall" indiating obstacles
           terminal: a list of two element list containing the conditions
           that will end the game
           P1: the probablity that the object move as intended
           P2: the probablity that the object will malfunction and move side way

    output: a three dimensional np matrix with indexes as [current location, next location, action]
            the index of action can be 0,1,2,3, corresponding to up, left, down right
    '''
    orientations = [(1,0), (0, 1), (-1, 0), (0, -1)]
    nrow = game.shape[0]
    ncol = game.shape[1]
    T_matrix = np.zeros(shape = (nrow*ncol, nrow*ncol, len(orientations)))
    
    # loop through rows and columns

    for row in range(0,game.shape[0]):
        for col in range(0,game.shape[1]):
            
            # the absolute location of the corresponding row and column
            current_grid = to_abs_loc([row,col],game)
            
            if [row,col] in terminal:
                T_matrix [current_grid,:,:] = 0.0
            elif game[row,col] == "wall":
                T_matrix [current_grid,:,:] = 0.0
            else:
                neighbor_grid = find_neighbor_grids (row,col,game)
                
                up =  to_abs_loc(neighbor_grid[0],game)
                left = to_abs_loc(neighbor_grid[1],game)
                down = to_abs_loc(neighbor_grid[2],game)
                right = to_abs_loc(neighbor_grid[3],game)
                
                #print (up,left,down,right)
                
        # stupid method to calculate probability and considering obstacles
        # sometimes stupid method is the smartest method -- Albert Einstein or somebody 
                
                for action in range(0,4,1):
                    if action == 0: #moving up
                        
                        # if moving up, but upper and left neighbor are wall
                        if (up == "wall") and (left== "wall") and (right != "wall"):
                            T_matrix [current_grid,current_grid,action] = P1+P2
                            
                            # calculate probablity of landing on the right
                            T_matrix [current_grid,right,action] = P2                               
                        
                        # if moving up, but upper and left and right neighbor are wall,stay in original place
                        elif  (up  == "wall") and (left == "wall") and (right== "wall") :
                            T_matrix [current_grid,current_grid,action] = 1
                            
                        # if moving up, but upper and right neighbor are wall 
                        elif  (up == "wall") and (right == "wall") and (left!= "wall"):
                            T_matrix [current_grid,current_grid,action] = P1+P2
                            # calculate probablity of landing on the left
                            T_matrix [current_grid,left,action] = P2
                        
                        elif (left == "wall") and (right == "wall") and (up!= "wall"):
                            
                            T_matrix [current_grid,current_grid,action] = P2+P2
                            T_matrix [current_grid,up,action] = P1
                            
                        # if moving up, right neighbor are wall 
                        elif (right == "wall") and (left != "wall") and (up != "wall"):
                            # probablity of staying in the same place
                            T_matrix [current_grid,current_grid,action] = P2
                            # calculate probablity of landing on the left
                            T_matrix [current_grid,left,action] = P2    
                            #probability of moving up
                            T_matrix [current_grid,up,action] = P1
                            
                        # if moving up, left neighbor are wall 
                        elif (left == "wall") and (up != "wall") and (right != "wall"):
                            # probablity of staying in the same place
                            T_matrix [current_grid,current_grid,action] = P2
                            # calculate probablity of landing on the right
                            T_matrix [current_grid,right,action] = P2    
                            #probability of moving up
                            T_matrix [current_grid,up,action] = P1
                            
                        elif (up== "wall") and (left != "wall") and (right != "wall"):                
                            T_matrix [current_grid,left,action] = P2                     
                            T_matrix [current_grid,right,action] = P2                      
                            T_matrix [current_grid,current_grid,action] = P1
                            
                        else: # if there is no obstacle surround the grid:
                            T_matrix [current_grid,up,action] = P1
                            T_matrix [current_grid,left,action] = P2
                            T_matrix [current_grid,right,action] = P2
                                                          
                    if action == 1: #moving left
                        if current_grid ==6:
                            print (to_abs_loc(neighbor_grid[1],game))
                            #print (up,left,down,right)
                        if (left == "wall") and (up != "wall") and (down != "wall"):
                            T_matrix [current_grid,current_grid,action] = P1                           
                            T_matrix [current_grid,up,action] = P2   
                            T_matrix [current_grid,down,action] = P2  
                        elif (left != "wall") and (up == "wall") and (down != "wall"):
                            T_matrix [current_grid,current_grid,action] = P2
                            T_matrix [current_grid,down,action] = P2
                            T_matrix [current_grid,left,action] = P1
                        elif (left != "wall") and (up != "wall") and (down == "wall"):
                            T_matrix [current_grid,current_grid,action] = P2
                            T_matrix [current_grid,up,action] = P2
                            T_matrix [current_grid,left,action] = P1
                            
                        elif (left == "wall") and (down == "wall") and (up != "wall"):
                            T_matrix [current_grid,current_grid,action] = P1+P2
                            T_matrix [current_grid,up,action] = P2
                            
                        elif (left == "wall") and (up == "wall") and (down != "wall") :
                            T_matrix [current_grid,current_grid,action] = P1+P2
                            T_matrix [current_grid,down,action] = P2
                            
                        elif (down == "wall") and (up == "wall") and (left != "wall"):
                            T_matrix [current_grid,current_grid,action] = P2+P2
                            T_matrix [current_grid,left,action] = P1
                            
                        elif (left == "wall") and (up == "wall") and (down == "wall"):
                            T_matrix [current_grid,current_grid,action] = 1
                        
                        else:
                            T_matrix [current_grid,down,action] = P2
                            T_matrix [current_grid,up,action] = P2
                            T_matrix [current_grid,left,action] = P1
                        
                    if action == 2: #moving down
                        
                        if (left== "wall") and (down != "wall") and (right != "wall"):
                            T_matrix [current_grid,current_grid,action] = P2 
                            T_matrix [current_grid,right,action] = P2   
                            T_matrix [current_grid,down,action] = P1   
                        elif (right == "wall") and (left!= "wall") and (down != "wall"):
                            T_matrix [current_grid,current_grid,action] = P2
                            T_matrix [current_grid,left,action] = P2
                            T_matrix [current_grid,down,action] = P1
                        elif (down == "wall") and (right != "wall") and (left != "wall"):
                            T_matrix [current_grid,current_grid,action] = P1
                            T_matrix [current_grid,right,action] = P2
                            T_matrix [current_grid,left,action] = P2
                            
                        elif (left == "wall") and (down == "wall") and (right != "wall"):
                            T_matrix [current_grid,current_grid,action] = P1+P2
                            T_matrix [current_grid,right,action] = P2
                            
                        elif (down == "wall") and (right == "wall") and (left != "wall"):
                            T_matrix [current_grid,current_grid,action] = P1+P2
                            T_matrix [current_grid,left,action] = P2
                            
                        elif (left == "wall") and (right == "wall") and (down != "wall"):
                            T_matrix [current_grid,current_grid,action] = P2+P2
                            T_matrix [current_grid,down,action] = P1
                            
                        elif (left == "wall") and (right == "wall") and (down == "wall"):
                            T_matrix [current_grid,current_grid,action] = 1
                        
                        else:
                            T_matrix [current_grid,left,action] = P2
                            T_matrix [current_grid,right,action] = P2
                            T_matrix [current_grid,down,action] = P1   
                            
                            
                    if action == 3: #moving right
                        
                        if (up== "wall") and (down != "wall") and (right != "wall"):
                            T_matrix [current_grid,current_grid,action] = P2 
                            T_matrix [current_grid,down,action] = P2   
                            T_matrix [current_grid,right,action] = P1   
                        elif (right == "wall") and (up!= "wall") and (down != "wall"):
                            T_matrix [current_grid,current_grid,action] = P1
                            T_matrix [current_grid,up,action] = P2
                            T_matrix [current_grid,down,action] = P2
                        elif (down == "wall") and (up!= "wall") and (right != "wall"):
                            T_matrix [current_grid,current_grid,action] = P2
                            T_matrix [current_grid,right,action] = P1
                            T_matrix [current_grid,up,action] = P2
                            
                        elif (up == "wall") and (down == "wall") and (right != "wall"):
                            T_matrix [current_grid,current_grid,action] = P2+P2
                            T_matrix [current_grid,right,action] = P1
                            
                        elif (down == "wall") and (right == "wall") and (up != "wall") :
                            T_matrix [current_grid,current_grid,action] = P1+P2
                            T_matrix [current_grid,up,action] = P2
                            
                        elif (up == "wall") and (right == "wall") and (down != "wall"):
                            T_matrix [current_grid,current_grid,action] = P1+P2
                            T_matrix [current_grid,down,action] = P2
                            
                        elif (up == "wall") and (right == "wall") and (down == "wall"):
                            T_matrix [current_grid,current_grid,action] = 1
                        
                        else:
                            T_matrix [current_grid,up,action] = P2
                            T_matrix [current_grid,down,action] = P2
                            T_matrix [current_grid,right,action] = P1  
    return (T_matrix)



                    
def return_state_utility(v, T, u, reward, gamma):
    """Return the state utility.

    @param v the value vector
    @param T transition matrix
    @param u utility vector
    @param reward for that state
    @param gamma discount factor
    @return the utility of the state
    """
    action_array = np.zeros(4)
    for action in range(0, 4): 
        # 0 means left, 1 means up, 2 means right, 3 means down
        action_array[action] = np.sum(np.multiply(u, np.dot(v, T[:,:,action])))
        # multiply the utility vector by (the dot product of value vector and transition matrix)
    return (reward + gamma * np.max(action_array))


def print_utlity_matrix(nparray,game):
    '''
    print the utility matrix according to the game rule
    nparray is a flatten array with index indicating absolute locatino of the grid
    '''
    tot_grid= game.shape[0]*game.shape[1]
    for i in range(0,tot_grid,game.shape[1]):
        print (np.round(nparray[i:i+game.shape[1]],6))
        
def print_policy_from_utility(u,game,terminal):
    # loop thru each row and column in the grid
    # initialize a empty policy matrix
    p = np.zeros(shape = game.shape).flatten().astype(np.float) 
    policy_string = ""
    
    for row in range(0,game.shape[0]):
        for col in range(0,game.shape[1]):
            current_grid = to_abs_loc([row,col],game)
            
            neighbors = find_neighbor_grids (row,col,game)
            
            if (game[row,col] != "wall") and ([row,col] not in terminal):
                max_u = np.amin(u)
                max_dir = 0
                for i in range(0,4): # for four different direction
                    if (neighbors[i] != "wall"):
                        temp_utility = u[to_abs_loc(neighbors[i],game)]
                        
                        if temp_utility >max_u:
                            max_u = temp_utility
                            max_dir = i
                        
                p[current_grid] = max_dir
            elif (game[row,col]== "wall"):
                p[current_grid] = np.NaN               
            else: 
                p[current_grid] = -1    

    policy_string = ''
    
    for row in range(game.shape[0]):
        for col in range(game.shape[1]):
            loc = to_abs_loc ([row,col], game)
            #print (p[loc])
            if(p[loc] == -1): policy_string += " *  "            
            elif(p[loc] == 0): policy_string += " ^  "
            elif(p[loc] == 1): policy_string += " <  "
            elif(p[loc] == 2): policy_string += " v  "           
            elif(p[loc] == 3): policy_string += " >  "
            elif(np.isnan(p[loc])): policy_string += " W  "
        policy_string += '\n'                
        
    print(policy_string)    
    
def generate_graph(utility_list):
    """Given a list of utility arrays (one for each iteration)
       it generates a matplotlib graph and save it as 'output.jpg'
    """
    name_list = ('(1,3)', '(2,3)', '(3,3)', '+1', '(1,2)', '#', '(3,2)', '-1', '(1,1)', '(2,1)', '(3,1)', '(4,1)')
    color_list = ('cyan', 'teal', 'blue', 'green', 'magenta', 'black', 'yellow', 'red', 'brown', 'pink', 'gray', 'sienna')
    counter = 0
    index_vector = np.arange(len(utility_list))
    for state in range(12):
        state_list = list()
        for utility_array in utility_list:
             state_list.append(utility_array[state])
        plt.plot(index_vector, state_list, color=color_list[state], label=name_list[state])  
        counter += 1
    #Adjust the legend and the axis
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.4), ncol=3, fancybox=True, shadow=True)
    plt.ylim((-1.1, +1.1))
    plt.xlim((1, len(utility_list)-1))
    plt.ylabel('Utility', fontsize=15)
    plt.xlabel('Iterations', fontsize=15)
    plt.savefig("./output.jpg", dpi=500)
    
def main():
    # This section below defines the game
    
    # game is basically the reward vector but with obstacles identified as "wall"
    game = np.array([[-0.04, -0.04, -0.04,  +1.0],
                  [-0.04,   "wall", -0.04,  -1.0],
                  [-0.04, -0.04, -0.04, -0.04]])    
                
    terminal = [[0,3],[1,3]]
    
    P1 = 0.8 #probablity of robot moves as intended
    P2 = 0.1 #probability of robot that will move side way
    ############################################
    
    tot_states = game.shape[0]*game.shape[1] #total # of grids
    
    #parameter that can be tuned########
    
    for epsilon in [0.1,0.01,0.001,0.0001]: #input a list of epsilon value to loop through
        gamma_space = np.linspace(0.05,0.99,30) # check out different gamma value
        iteration_list = []
        for gamma in gamma_space:
            iteration = 0 #Iteration counter
            
            #List containing the utility data for each iteation
            graph_list = list()
                  
            # calculate the transition matrix based on the reward matrix
            T = find_Tmatrix(game,terminal,P1=P1,P2=P2)
            
   
            #Reward vector, convert the "wall" into zero reward
            r = np.where(game == "wall",0.0,game).flatten().astype(np.float) 
            #use flatten so that we can reuse the reference's code
            # use astype.(np.float) to convert the data type to float
            
            
            # initialize the utility vectors
            #Utility vectors
            u = np.zeros(shape = game.shape).flatten().astype(np.float)
            u1 = np.zeros(shape = game.shape).flatten().astype(np.float)
            
            
            while True:
                delta = 0
                u = u1.copy()
                iteration += 1
                graph_list.append(u)
                for s in range(tot_states):
                    reward = r[s]
                    v = np.zeros((1,tot_states))
                    v[0,s] = 1.0
                    u1[s] = return_state_utility(v, T, u, reward, gamma)
                    delta = max(delta, np.abs(u1[s] - u[s])) #Stopping criteria       
                if delta < epsilon * (1 - gamma) / gamma:
                        iteration_list.append(iteration)
                        print("=================== FINAL RESULT ==================")
                        print("Iterations: " + str(iteration))
                        print("Delta: " + str(delta))
                        print("Gamma: " + str(gamma))
                        print("Epsilon: " + str(epsilon))
                        
                        print("The utility matrix converge to:")
                        print("===================================================")
                        # invoke the print utility functions
                        print_utlity_matrix(u,game)
                        print("===================================================")
                        
                        print("The optimal policy is the following:")
                        print("===================================================")
                        print_policy_from_utility(u,game,terminal)
                        print("===================================================")
                        break
        
        ##########plotting##############
        
        plt.figure(1)
        plt.plot(gamma_space,iteration_list,label = epsilon)
    
    plt.title("Iteration needed for various Gamma and Epsilon values",fontsize = 15)
    plt.xlabel("Discount Factor",fontsize=15), plt.ylabel("Iteration to Converge",fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=14) # change axis tick size
    plt.legend(loc="best",prop={'size': 12})
    plt.tight_layout()    
    plt.show()
    
    plt.figure(2)
    # generate the utitlity vector for each state. 
    generate_graph(graph_list)        
            
if __name__ == "__main__":
    main()
                        
                    

                
            
                    
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
    
    