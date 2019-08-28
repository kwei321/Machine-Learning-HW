# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 11:35:19 2018

@author: kunwe
"""
# reference
import numpy as np
import matplotlib.pyplot as plt
def find_neighbor_grids (row_num,col_num,grid):
    '''
    input: row_num and col_num: the grid location (0-based matrix)
            gird: the np matrix
    output: output the neighboring grids in terms of matrix indexes in a list
    
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
    if location != "wall":
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

def return_policy_evaluation(p, u, r, T, gamma):
  """Return the policy utility.

  @param p policy vector
  @param u utility vector
  @param r reward vector
  @param T transition matrix
  @param gamma discount factor
  @return the utility vector u
  """
  for s in range(T.shape[0]):
    if not np.isnan(p[s]):
      v = np.zeros((1,T.shape[0]))
      v[0,s] = 1.0
      action = int(p[s])
      u[s] = r[s] + gamma * np.sum(np.multiply(u, np.dot(v, T[:,:,action])))
  return u

def return_expected_action(u, T, v):
    """Return the expected action.

    It returns an action based on the
    expected utility of doing a in state s, 
    according to T and u. This action is
    the one that maximize the expected
    utility.
    @param u utility vector
    @param T transition matrix
    @param v starting vector
    @return expected action (int)
    """
    actions_array = np.zeros(4)
    for action in range(4):
       #Expected utility of doing a in state s, according to T and u.
       actions_array[action] = np.sum(np.multiply(u, np.dot(v, T[:,:,action])))
    return np.argmax(actions_array)


def print_policy(p, shape):
    """Printing utility.

    Print the policy actions using symbols:
    ^, v, <, > up, down, left, right
    * terminal states
    W obstacles
    """
    
    # map numbers to symbol
    # 0 means up, 1 means left, 2 means down, 3 means right
    # * means terminal, W means obstacles
    counter = 0
    policy_string = ""
    for row in range(shape[0]):
        for col in range(shape[1]):
            if(p[counter] == -1): policy_string += " *  "            
            elif(p[counter] == 0): policy_string += " ^  "
            elif(p[counter] == 1): policy_string += " <  "
            elif(p[counter] == 2): policy_string += " v  "           
            elif(p[counter] == 3): policy_string += " >  "
            elif(np.isnan(p[counter])): policy_string += " W  "
            counter += 1
        policy_string += '\n'
    print(policy_string)

def print_utlity_matrix(nparray,game):
    '''
    print the utility matrix according to the game rule
    nparray is a flatten array with index indicating absolute locatino of the grid
    '''
    tot_grid= game.shape[0]*game.shape[1]
    for i in range(0,tot_grid,game.shape[1]):
        print (np.round(nparray[i:i+game.shape[1]],4))


def main():
    
    ###############################################
    # This section below defines the game rules and conditions
    
    # game is basically the reward matrix but with obstacles identified as "wall"
    game = np.array([
                  [-1.00, 'wall', -0.04, -0.04, -1.00, -1.00, -0.04, 1.00],
                  [-0.04, -0.04, 'wall', -0.04, -0.04, -0.04, -0.04, -0.04],
                  [-1.00, 'wall', -0.04, -1.00, -0.04, -1.00, -0.04, -0.04],
                  [-0.04, -0.04, 'wall', -0.04, -0.04, -0.04, -1.00, -0.04],
                  ['wall', -0.04, 'wall',-0.04, -1.00, -1.00, -0.04,'wall'],
                  [-1.00, -0.04, -0.04, -0.04, -0.04, -0.04, -1.00, -0.04],
                  [-0.04, -0.04, -0.04, -0.04, -0.04, 'wall', -0.04, -0.04],
                  [-0.04, -1.00, -1.00, -0.04, -1.00, -0.04, -0.04, -0.04],
                  ])    
                
    trap = np.argwhere((game == "-1.0")).tolist()
    charging_station = np.argwhere((game == "1.0")).tolist()
    terminal = trap + charging_station
    
    
    P1 = 0.8 #probablity of robot moves as intended
    P2 = 0.1 #probability of robot that will move side way
    
    ############################################
    
    tot_states = game.shape[0]*game.shape[1] #total # of grids
    
    # calculate the transition matrix based on reward matrix
    T = find_Tmatrix(game,terminal,P1,P2)
    
    for epsilon in [0.1,0.01,0.001,0.0001]:
        gamma_space = np.linspace(0.05,0.99,30) # check out different gamma value
        iteration_list = []
        for gamma in gamma_space:
            iteration = 0
            ######Generate the first policy randomly##############
            # NaN=Nothing, -1=Terminal, 0=Up, 1=Left, 2=Down, 3=Right
            # p matrix is a 1 dimensional numpy array
            p = np.random.randint(0, 4, size=(tot_states)).astype(np.float32)
            
            # convert the "wall" location into nan
            p = np.where(game.flatten() == "wall",np.NaN,p).astype(np.float) 
            
            # convert the terminal location into -1
            for terminal_loc in terminal:
                p[to_abs_loc(terminal_loc,game)] = -1
            ##################################################
           
            
            
            # initialize the utility vectors
            
            u = np.zeros(shape = game.shape).flatten().astype(np.float)
            
            
            #Reward vector, convert the "wall" into zero reward
            r = np.where(game == "wall",0.0,game).flatten().astype(np.float) 
            print (r)
            #use flatten so that we can reuse the reference's code
            # use astype.(np.float) to convert the data type to float
        
            while True:
                iteration += 1
                #1- Policy evaluation
                u_0 = u.copy()
                u = return_policy_evaluation(p, u, r, T, gamma) # return the utility of the current policy
                
                p_copy = p.copy()
            
                for s in range(tot_states): # loop through every possible state
                    # if the policy array is not empty or it is not terminal
                    if not np.isnan(p[s]) and not p[s]==-1:
                        v = np.zeros((1,tot_states))
                        v[0,s] = 1.0
                        
                        #2- Policy improvement
                        a = return_expected_action(u, T, v)  
                        # if the action is not equal to the current policy then
                        # update the policy to action
                        if a != p[s]: 
                            p[s] = a
                            
                #Stopping criteria
                delta = np.absolute(u - u_0).max()
                if (delta < epsilon * (1 - gamma) / gamma) and (np.array_equal(p.astype(int),p_copy.astype(int)) == True):
                
                    iteration_list.append(iteration)
                       
                    break
                
                
    
                
            print("=================== FINAL RESULT ==================")
            print("Iterations: " + str(iteration))
            print("Delta: " + str(delta))
            print("Gamma: " + str(gamma))
            print("Epsilon: " + str(epsilon))
            print("===================================================")
            # invoke the print utility functions
            print_utlity_matrix(u,game)
            print("===================================================")
            print_policy(p, shape=(game.shape))
            print("===================================================")
            
        plt.figure(1) 
        plt.plot(gamma_space,iteration_list,label = epsilon)
        
            
    
    plt.title("Iteration needed for various Gamma and Epsilon values",fontsize = 15)
    plt.xlabel("Discount Factor",fontsize=15), plt.ylabel("Iteration to Converge",fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=14) # change axis tick size
    plt.legend(loc="best",prop={'size': 12})
    plt.tight_layout()    
    plt.show() 
     
    
if __name__ == "__main__":
    main()