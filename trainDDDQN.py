#!/usr/bin/env python
# coding=utf-8

import os
import argparse
import json
import numpy as np


import retro
from collections import deque
import random
from utils import print_var # User-defined lib. Look at utils.py

def parse_arg(argv=None):
    parser = argparse.ArgumentParser(description='Emoticon Text Classification')

    parser.add_argument('--run_mode_container', type=str, help='Whether to run local or docker ')    
    parser.add_argument('--model_export_path', type=str, help='Model export path')
    parser.add_argument('--max_steps', type=int, default=10, help='Training max steps per episode')
    parser.add_argument('--total_num_records', type=int, default=100, help='Number of total records')    
    args = parser.parse_args()
    
    return args

def get_hyperparameters_local(args):
    
    hyperparams = {}
    hyperparams["max_steps"] = 1
    hyperparams["total_episodes"] = 1
    hyperparams["batch_size"] = 2   
    hyperparams["pretrain_length"] = 4  
    hyperparams["memory_size"] = 4
    hyperparams["max_tau"] = 2        

    
    
    
    return hyperparams
    
    

def get_hyperparameters_sagemaker():

    prefix = '/opt/ml/'

#     input_path = prefix + 'input/data'
#     output_path = os.path.join(prefix, 'output')
#     model_path = os.path.join(prefix, 'model')
    param_path = os.path.join(prefix, 'input/config/hyperparameters.json')

    
    # Here we only support a single hyperparameter. Note that hyperparameters are always passed in as
    # strings, so we need to do any necessary conversions.
    
    with open(param_path, 'r') as tc:
        trainingParams = json.load(tc)
    
    total_episodes = trainingParams.get('total_episodes', None)  
    total_episodes = int(total_episodes)

    max_steps = trainingParams.get('max_steps', None)    
    max_steps = int(max_steps)
    
    batch_size = trainingParams.get('batch_size', None)    
    batch_size = int(batch_size)
    
    pretrain_length = trainingParams.get('pretrain_length', None)    
    pretrain_length = int(pretrain_length)

    memory_size = trainingParams.get('memory_size', None)    
    memory_size = int(memory_size)
    
    max_tau = trainingParams.get('max_tau', None)    
    max_tau = int(max_tau)

    hyperparams = {}
    hyperparams["total_episodes"] = total_episodes        
    hyperparams["max_steps"] = max_steps

    hyperparams["batch_size"] = batch_size    
    hyperparams["pretrain_length"] = pretrain_length    
    hyperparams["memory_size"] = memory_size    
    hyperparams["max_tau"] = max_tau        
    
    
    return hyperparams

    
if __name__ == "__main__":
    
    args = parse_arg()
    
    # Whether the SageMaker job is run
    run_mode_container = args.run_mode_container     
    print("run_mode_container: ", run_mode_container)
    
    if run_mode_container:
        print("The mode is NO container")
        hyperparams = get_hyperparameters_local(args)
        
    else:
        print("The mode is container")   
        hyperparams = get_hyperparameters_sagemaker()        
    
    
    # location to be saved for a model
    export_model_dir = "/opt/ml/model/model.ckpt"   
    model_path = export_model_dir
    print("model_path: ", model_path)

    
    
    

    ##########################
    # Build Game Environment       
    ##########################    
    env = retro.make(game='SpaceInvaders-Atari2600')

    print("The size of our frame is: ", env.observation_space)
    print("The action size is: ", env.action_space.n)
    # Here we create an hot encoded version of our actions
    # possible_actions = [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0]...]
    possible_actions = np.array(np.identity(env.action_space.n, dtype=int).tolist())

    ##########################
    # Hyperparameter
    ##########################    
    
    ### MODEL HYPERPARAMETERS

    ### PREPROCESSING HYPERPARAMETERS
    stack_size = 4 # Number of frames stacked

    state_size = [110, 84, 4] # Our input is a stack of 4 frames 110 * 84 * 4 (width, heifht, channel)
    action_size = env.action_space.n # 8 possible actions
    learning_rate = 0.00025

    ### TRAINING HYPERPARAMETERS
    total_episodes = hyperparams["total_episodes"] # Total episodes for training
    max_steps = hyperparams["max_steps"] # Max possible steps in an episode

    # batch_size = 64
    batch_size = hyperparams["batch_size"]

    ### MEMORY HYPERPARAMETERS
    ## If you have GPU, change to 1 million
    # pretrain_length = batch_size # Number of experience stored in the Memory when initialized for the first time
    pretrain_length = hyperparams["pretrain_length"]
    memory_size = hyperparams["memory_size"] # Number of experiences the Memory can keep

    # Fixed Q targets hyperparameters
    max_tau = hyperparams["max_tau"] #Tau is the C step where we update our target network

    # Exploration parameters for epsilon greedy strategy
    explore_start = 1.0 # exploration prob. at start
    explore_stop = 0.01 # minimum exploration prob
    decay_rate = 0.00001 # exponential decay rate for exploration prob

    # Q learning hyperparameters
    gamma = 0.9 # Diccounting rate

    ### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
    training = True
    ## TRUN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
    episode_render = False    


    #######################
    # Load Image Processing lib
    #######################

    # Initialize deque with zero-images one array for each image
    stacked_frames = deque([np.zeros((110,84), dtype=np.int) for i in range(stack_size)], maxlen=4)
    from utils import preprocess_frame
    from utils import stack_frames 



    #######################
    # Initialize Memory
    #######################
    from utils import Memory


    # Instantiate memory
    # SumTree = utils.SumTree(memory_size)
    # memory = utils.Memory(memory_size)
    memory = Memory(memory_size)

    # Render the envrironment
    # game.new_episodes()

    for i in range(pretrain_length):
        # If it's the first step
        if i == 0:
            # First we need a state
            state = env.reset()
            state, stacked_frames = stack_frames(stacked_frames, state, True)

        # Random action
        # action = random.choice(possible_actions)
        # Get the next_state, the rewards, done by taking a random action
        choice = random.randint(1, len(possible_actions)) -1
        # print_var("choice", choice)
        action = possible_actions[choice]

        # Get the rewards
        next_state, reward, done, _ = env.step(action)

        # Stack the frames
        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

        # If the episode is finished (We're dead 3x)
        if done:
            # We finished the episode
            next_state = np.zeros(state.shape)

            # Add experience to memory
            experience = state, action, reward, next_state, done
            memory.store(experience)

            # Start a new episode
            state = env.reset()
            # Stack the frames
            state, stacked_frames = stack_frames(stacked_frames, state, True)
        else:
            # Add experience to memory
            experience = state, action, reward, next_state, done
            memory.store(experience)

            # Our state is now the next_state
            state = next_state
        

    

    #######################
    # Define DQNetwork
    #######################

    import DDDQNNet 

    # Instantiate the DQNetwork
    DQNetwork = DDDQNNet.DDDQNNetwork(state_size, action_size, learning_rate, name="DQNetwork")
    tf = DQNetwork.build()

    # Instantiate the target network
    TargetNetwork = DDDQNNet.DDDQNNetwork(state_size, action_size, learning_rate, name="TargetNetwork")
    tf2 = TargetNetwork.build()

    #######################
    # Load Train lib
    #######################

    """
    This function will do the part
    With ϵϵ select a random action atat, otherwise select at=argmaxaQ(st,a)
    """
    def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
        ## EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        ## First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

        if (explore_probability > exp_exp_tradeoff):
            # Make a random action (exploration)
            choice = random.randint(1,len(possible_actions))-1
            action = possible_actions[choice]

        else:
            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1, *state.shape))})

            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            action = possible_actions[choice]


        return action, explore_probability

    # This function helps us to copy one set of variables to another
    # In our case wt_e use it when we want to copy the parameters of DQN to Target_network

    def update_target_graph():
        # Get the parameters of our DQNNetwork
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "DQNetwork")

        # Get the parameters of our Target_network
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "TargetNetwork")
        op_holder = []

        # Update our target_network parameters with DQNNetwork parameters
        for from_var, to_var in zip(from_vars, to_vars):
            op_holder.append(to_var.assign(from_var))
        return op_holder    
    
    #######################
    # Train
    #######################

    # Saver will help us to save our model
    saver = tf.train.Saver()

    rewards_list= list()

    if training == True:
        with tf.Session() as sess:
            # Initialize the variables
            sess.run(tf.global_variables_initializer())

            # Initialize the decay rate (that will use to reduce epsilon) 
            decay_step = 0

            # Set tau = 0
            tau = 0

            # Update the parameters of our TargetNetwork with DQN_weights
            update_target = update_target_graph()
            sess.run(update_target)

            for episode in range(total_episodes):
                # Set step to 0
                step = 0

                # Initialize the rewards of the episode
                episode_rewards = []

                # Make a new episode and observe the first state
                state = env.reset()

                # Remember that stack frame function also call our preprocess function.
                state, stacked_frames = stack_frames(stacked_frames, state, True)

                while step < max_steps:
                    step += 1

                    # Increase decay_step
                    decay_step += 1

                    #Increase decay_step
                    decay_step +=1

                    # Predict the action to take and take it
                    action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state, possible_actions)

                    #Perform the action and get the next_state, reward, and done information
                    next_state, reward, done, _ = env.step(action)

                    if episode_render:
                        env.render()

                    # Add the reward to total reward
                    episode_rewards.append(reward)

                    # If the game is finished
                    if done:
                        # The episode ends so no next state
                        next_state = np.zeros((110,84), dtype=np.int)

                        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                        # Set step = max_steps to end the episode
                        step = max_steps

                        # Get the total reward of the episode
                        total_reward = np.sum(episode_rewards)

                        print('Episode: {}'.format(episode),
                                      'Total reward: {}'.format(total_reward),
                                      'Explore P: {:.4f}'.format(explore_probability),
                                    'Training Loss {:.4f}'.format(loss))

                        rewards_list.append((episode, total_reward))

                        # Add experience to memory
                        experience = state, action, reward, next_state, done
                        memory.store(experience)


                    else:
                        # Stack the frame of the next_state
                        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                        # Add experience to memory
                        experience = state, action, reward, next_state, done
                        memory.store(experience)


                        # st+1 is now our current state
                        state = next_state


                    ### LEARNING PART            
                    # Obtain random mini-batch from memory
                    tree_idx, batch, ISWeights_mb = memory.sample(batch_size)
                    # batch = memory.sample(batch_size)


                    states_mb = np.array([each[0][0] for each in batch], ndmin=3)
                    # print_var("states_mb", states_mb.shape)                
                    actions_mb = np.array([each[0][1] for each in batch])
                    # print_var("actions_mb", actions_mb.shape)    
                    # print_var("actions_mb", actions_mb)                    


                    rewards_mb = np.array([each[0][2] for each in batch]) 
                    # print_var("rewards_mb", rewards_mb.shape)   
                    # print_var("rewards_mb", rewards_mb)                                                                
                    next_states_mb = np.array([each[0][3] for each in batch], ndmin=3)
                    # print_var("next_states_mb", next_states_mb.shape)                                                                
                    dones_mb = np.array([each[0][4] for each in batch])
                    # print_var("dones_mb", dones_mb.shape)  


                    target_Qs_batch = []

                    ### Double DQN Logic
                    # Use DQNNetwork to select the action to take at next_state (a') 
                    # (action with the highest Q-value)
                    # Use TargetNetwork to calculate the Q_val of Q(s',a')

                    # Get Q values for next_state                
                    q_next_state = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: next_states_mb})


                    # Calculate Qtarget for all actions that state
                    q_target_next_state = sess.run(TargetNetwork.output, feed_dict = {TargetNetwork.inputs_: next_states_mb})

                    # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma * Qtarget(s',a')
                    for i in range(0, len(batch)):
                        terminal = dones_mb[i]

                        # We got a'
                        action = np.argmax(q_next_state[i])

                        # If we are in a terminal state, only equals reward
                        if terminal:
                            target_Qs_batch.append(rewards_mb[i])
                        else:
                            # Take the Qtarget for action a'
                            target = rewards_mb[i] + gamma * q_target_next_state[i][action]
                            target_Qs_batch.append(target)
                    targets_mb = np.array([each for each in target_Qs_batch])
                    _, loss, absolute_errors = sess.run([DQNetwork.optimizer, DQNetwork.loss, DQNetwork.absolute_errors],
                                                        feed_dict = {DQNetwork.inputs_: states_mb,
                                                                     DQNetwork.target_Q: targets_mb,
                                                                     DQNetwork.actions_: actions_mb,
                                                                     DQNetwork.ISWeights_: ISWeights_mb
                                                                    }
                                                       )

                    # Update priorituy
                    memory.batch_update(tree_idx, absolute_errors)

                    if tau > max_tau:
                        # Update the parameters of our TargetNetwork with DQN_weights
                        update_target = update_target_graph()
                        sess.run(update_target)
                        tau = 0
                        print("Model updated")

                # Save model every 5 episodes
                if episode % 5 == 0:
                    save_path = saver.save(sess, model_path)
                    print("Model Saved")