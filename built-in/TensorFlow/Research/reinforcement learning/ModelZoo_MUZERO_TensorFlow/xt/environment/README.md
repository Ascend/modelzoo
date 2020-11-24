##Env Mddule Introduction
Env module supplies 3 interfaces for other modules including reset, step, and get_init_state.
The following lists the details of these interfaces.
1. reset
Reset interface is used to reset the whole environment,when you start a new game or a new episode.
Reset interface doesn't have any input parameter and it returns the state of environment.
2. step
Step interface is used to control a agent to interact with the environment.
It's iuput is the control action and agent's index, and it returns the state of environment and
the done flag of each epsiode.
3. get_init_state
The interface is used to get init state of an agent.

##Finished Environment Introduction
### torcs
Torcs is an open sourced race game, in which an agent car can be controlled by
users to we can control finish competition.
Under torcs directory, there are 3 files in which  gym_torcs_multi.py and snakeoil3_gym.py
are open sourced codes,which are used to communicate with trocs game.
torcs_basic.py provides standard interface to Env module

### kyber_sim
kyber_sim is a simulator developed by our team. kyber_sim.py is used to communicate
with the simulator and supply standard interface to Env module.
