import numpy as np
import gym
import itertools as it
from dqn.dqn_agent import DQNAgent
from tensorboard_evaluation import *
from dqn.networks import NeuralNetwork, TargetNetwork
from utils import EpisodeStats


def run_episode(env, agent, deterministic, do_training=True, rendering=False, max_timesteps=1000):
    """
    This methods runs one episode for a gym environment. 
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """
    
    stats = EpisodeStats()        # save statistics like episode reward or action usage
    state = env.reset()

    step = 0
    while True:
        
        action_id = agent.act(state=state, deterministic=deterministic)
        next_state, reward, terminal, info = env.step(action_id)

        if do_training:  
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state
        
        if rendering:
            env.render()

        if terminal or step > max_timesteps: 
            break

        step += 1

    return stats

def train_online(env, agent, num_episodes, model_dir="./models_cartpole", tensorboard_dir="./tensorboard"):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train agent")

    tensorboard = Evaluation(os.path.join(tensorboard_dir, "train"), ["episode_reward", "a_0", "a_1"])

    # training
    for i in range(num_episodes):
        print("episode: ",i)
        stats = run_episode(env, agent, deterministic=False, do_training=True)
        tensorboard.write_episode_data(i, eval_dict={  "episode_reward" : stats.episode_reward, 
                                                                "a_0" : stats.get_action_usage(0),
                                                                "a_1" : stats.get_action_usage(1)})

        # TODO: evaluate your agent once in a while for some episodes using run_episode(env, agent, deterministic=True, do_training=False) to 
        # check its performance with greedy actions only. You can also use tensorboard to plot the mean episode reward.
        # ...
       
        # store model every 100 episodes and in the end.
        if i % 100 == 0 or i >= (num_episodes - 1):
            agent.saver.save(agent.sess, os.path.join(model_dir, "dqn_agent.ckpt"))
   
    tensorboard.close_session()


if __name__ == "__main__":

    # You find information about cartpole in 
    # https://github.com/openai/gym/wiki/CartPole-v0
    # Hint: CartPole is considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.

    env = gym.make("CartPole-v0").unwrapped

    # TODO: 
    # 1. init Q network and target network (see dqn/networks.py)
    # 2. init DQNAgent (see dqn/dqn_agent.py)
    # 3. train DQN agent with train_online(...)
 
