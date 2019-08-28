import numpy as np
from gridworld import *
import matplotlib.pyplot as plt
import time


def update_state_action(state_action_matrix, visit_counter_matrix, observation, new_observation, 
                        action, reward, alpha, gamma):
    '''Return the updated utility matrix
    @param state_action_matrix the matrix before the update
    @param observation the state obsrved at t
    @param new_observation the state observed at t+1
    @param action the action at t
    @param new_action the action at t+1
    @param reward the reward observed after the action
    @param alpha the ste size (learning rate)
    @param gamma the discount factor
    @return the updated state action matrix
    '''
    #Getting the values of Q at t and at t+1
    col = observation[1] + (observation[0]*4)
    q = state_action_matrix[action ,col]
    col_t1 = new_observation[1] + (new_observation[0]*4)
    q_t1 = np.max(state_action_matrix[: ,col_t1])
    #Calculate alpha based on how many time it
    #has been visited
    alpha_counted = 1.0 / (1.0 + visit_counter_matrix[action, col])
    #Applying the update rule
    #Here you can change "alpha" with "alpha_counted" if you want
    #to take into account how many times that particular state-action
    #pair has been visited until now.
    state_action_matrix[action ,col] = state_action_matrix[action ,col] + alpha * (reward + gamma * q_t1 - q)
    return state_action_matrix

def update_visit_counter(visit_counter_matrix, observation, action):
    '''Update the visit counter
   
    Counting how many times a state-action pair has been 
    visited. This information can be used during the update.
    @param visit_counter_matrix a matrix initialised with zeros
    @param observation the state observed
    @param action the action taken
    '''
    col = observation[1] + (observation[0]*4)
    visit_counter_matrix[action ,col] += 1.0
    return visit_counter_matrix

def update_policy(policy_matrix, state_action_matrix, observation):
    '''Return the updated policy matrix (q-learning)
    @param policy_matrix the matrix before the update
    @param state_action_matrix the state-action matrix
    @param observation the state obsrved at t
    @return the updated state action matrix
    '''
    col = observation[1] + (observation[0]*4)
    #Getting the index of the action with the highest utility
    best_action = np.argmax(state_action_matrix[:, col])
    #Updating the policy
    policy_matrix[observation[0], observation[1]] = best_action
    return policy_matrix

def return_epsilon_greedy_action(policy_matrix, observation, epsilon=0.1):
    tot_actions = int(np.nanmax(policy_matrix) + 1)
    action = int(policy_matrix[observation[0], observation[1]])
    non_greedy_prob = epsilon / tot_actions
    greedy_prob = 1 - epsilon + non_greedy_prob
    weight_array = np.full((tot_actions), non_greedy_prob)
    weight_array[action] = greedy_prob
    return np.random.choice(tot_actions, 1, p=weight_array)

def print_policy(policy_matrix):
    '''Print the policy using specific symbol.
    * terminal state
    ^ > v < up, right, down, left
    # obstacle
    '''
    counter = 0
    shape = policy_matrix.shape
    policy_string = ""
    for row in range(shape[0]):
        for col in range(shape[1]):
            if(policy_matrix[row,col] == -1): policy_string += " *  "            
            elif(policy_matrix[row,col] == 0): policy_string += " ^  "
            elif(policy_matrix[row,col] == 1): policy_string += " >  "
            elif(policy_matrix[row,col] == 2): policy_string += " v  "           
            elif(policy_matrix[row,col] == 3): policy_string += " <  "
            elif(np.isnan(policy_matrix[row,col])): policy_string += " #  "
            counter += 1
        policy_string += '\n'
    print(policy_string)

def return_decayed_value(starting_value, global_step, decay_step):
        """Returns the decayed value.
        decayed_value = starting_value * decay_rate ^ (global_step / decay_steps)
        @param starting_value the value before decaying
        @param global_step the global step to use for decay (positive integer)
        @param decay_step the step at which the value is decayed
        """
        decayed_value = starting_value * np.power(0.1, (global_step/decay_step))
        return decayed_value


def main():

    env = GridWorld(3, 4)
    
    ###############################################
    # This section below defines the game rules and conditions
    
    # game is basically the reward matrix but with obstacles identified as "wall"
    game = np.array([[-0.04, -0.04, -0.04,  +1.0],
                  [-0.04,   "wall", -0.04,  -100],
                  [-0.04, -0.04, -0.04, -0.04]])  
    terminal = [[0,3],[1,3]]
    P1 = 0.8 #probablity of robot moves as intended
    P2 = 0.1 #probability of robot that will move side way
    #####################################
    
    tot_states = game.shape[0]*game.shape[1] #total # of grids
    
    
    
    
    
    #######Define the state matrix#################
    # make regular condition to zero
    # make terminal condition to 1
    # make obstacle to - 1
    
    state_matrix = np.zeros(shape = (game.shape))
    state_matrix = np.where(game == "wall",-1,0).astype(np.float)
    for terminal_loc in terminal:
        state_matrix[terminal_loc[0],terminal_loc[1]] = 1

    print("State Matrix:")
    print(state_matrix)
    #################################################
    
    
    ###Reward vector, convert the "wall" into zero reward#############
    reward_matrix = np.where(game == "wall",0.0,game).astype(np.float) 
    #use flatten so that we can reuse the reference's code
    # use astype.(np.float) to convert the data type to float
    print("Reward Matrix:")
    print(reward_matrix)
    ######################################################
    
      #Define the transition matrix
    # up left down right
    # first row means moving moving up, 2nd means left, 3rd means down, 4 means right
    # first column means moving moving up, 2nd means left, 3rd means down, 4 means right
    transition_matrix = np.array([[P1, P2, 0.0, P2],
                                  [P2, P1, P2, 0.0],                                  
                                  [0.0, P2, P1, P2],
                                  [P2, 0.0, P2, P1]])

    
    



                                          
    #exploratory_policy_matrix = np.random.randint(low=0, high=4, size=(3, 4)).astype(np.float32)
    exploratory_policy_matrix = np.array([[1,      1, 1, -1],
                                          [0, np.NaN, 0, -1],
                                          [0,      3, 3,  3]])

    print("Exploratory Policy Matrix:")
    print(exploratory_policy_matrix)
    print_policy(exploratory_policy_matrix)

    env.setStateMatrix(state_matrix)
    env.setRewardMatrix(reward_matrix)
    env.setTransitionMatrix(transition_matrix)

    
    gamma = 0.9
    alpha = 0.1 #constant step size
    tot_epoch = 400000
    print_epoch = 1000
    
    for alpha in [0.1]:
        state_action_matrix = np.zeros((4,tot_states))
        visit_counter_matrix = np.zeros((4,tot_states))
            #Random policy
        policy_matrix = np.random.randint(low=0, high=4, size=game.shape).astype(np.float32)
        
        # convert the "wall" location into nan
        policy_matrix = np.where(game == "wall",np.NaN,policy_matrix).astype(np.float) 
        
        
        # convert the terminal location into -1
        for terminal_loc in terminal:
            policy_matrix[terminal_loc[0],terminal_loc[1]] = -1
        ##################################################
        
        policy_list = []
        for epoch in range(tot_epoch):
            #Reset and return the first observation
            observation = env.reset(exploring_starts=True)
            
            epsilon = return_decayed_value(0.1, epoch, decay_step=50000)
            is_starting = True
            policy_matrix_copy = policy_matrix.copy()
            
            for step in range(1000):
                #Take the action from the action matrix
                #action = policy_matrix[observation[0], observation[1]]
                #Take the action using epsilon-greedy
                
                action = return_epsilon_greedy_action(exploratory_policy_matrix, observation, epsilon=epsilon)
                
                
                if(is_starting): 
                    action = np.random.randint(0, 4)
                    is_starting = False  
                #Move one step in the environment and get obs and reward
                new_observation, reward, done = env.step(action)
                #Updating the state-action matrix
                state_action_matrix = update_state_action(state_action_matrix, visit_counter_matrix, observation, new_observation, 
                                                          action, reward, alpha, gamma)
                #Updating the policy
                policy_matrix = update_policy(policy_matrix, state_action_matrix, observation)
                #Increment the visit counter
                visit_counter_matrix = update_visit_counter(visit_counter_matrix, observation, action)
                observation = new_observation
                if done: 
                    break
                
    
            
            # specify whether to print all the iteration
            print_detail = True
            
            if(epoch % print_epoch == 0) and (print_detail == True):
                print("")
                print("Epsilon: " + str(epsilon))
                print("State-Action matrix after " + str(epoch+1) + " iterations:") 
                print(state_action_matrix)
                print("Policy matrix after " + str(epoch+1) + " iterations:") 
                print_policy(policy_matrix)
                
                policy_list.append(policy_matrix.tolist())
                
                
                
        #Time to check the utility matrix obtained
        print("State-Action matrix after " + str(tot_epoch) + " iterations:")
        print(state_action_matrix)
        print("Policy matrix after " + str(tot_epoch) + " iterations:")
        #print (policy_matrix)
        #print_policy(policy_matrix)
        
        # save the policy to csv to view later
        with open("QlearningPolicy"+"alpha"+ str(alpha)+".csv",'w') as file_handler:
            for item in policy_list:
                file_handler.write("{}\n".format(item))
            
    #return (policy_matrix,state_action_matrix,visit_counter_matrix )


if __name__ == "__main__":
    main()