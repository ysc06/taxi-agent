import random, numpy as np, matplotlib.pyplot as plt
import gymnasium as gym # our interface to environments, in particular Taxi-v3

class QAgent:
    def __init__(self, state_space : gym.Space, action_space : gym.Space, init_val=0.0):
        self._state_space = state_space
        self._action_space = action_space
        # TODO(III): initialize table for Q-values as self.q_table
        # .n : number of states state_space is an object 
        n_states = state_space.n
        n_actions = action_space.n
        self.q_table = np.full((n_states, n_actions), init_val, dtype=float)
    def save_weights(self, filename="QAgent_weights.npy"):
        np.save(filename, self.q_table)

    # TODO(III): implement epsilon-greedy action selection
    # return the action index (integer) chosen by the agent for the given state
    def act(self, state, epsilon=0.0):
        # 不能永遠選最好，必須要能探索新路。有epsilon機率亂走。
        if np.random.rand() < epsilon: # 決定是否要隨機探索
            # explore 
            action = self._action_space.sample()
        else:
            # exploit
            action = np.argmax(self.q_table[state])
        return action
    
    # TODO(III): implement the Q-learning update rule to update self.q_table
    # terminal is a boolean indicating whether next_state is a terminal state, if so then the target should be just the reward
    # alpha is the learning rate and gamma is the discount factor
    def update(self, state, action, reward, next_state, terminal, alpha, gamma):
        current_q = self.q_table[state][action]
        if terminal:
            target = reward
        else:
            # update
            target = reward + gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] += alpha * (target-self.q_table[state][action])

# TODO(III): implement the training loop for QAgent using an OpenAI Gym environment
# epochs: number of episodes to train for, an episode ends when the environment step returns either terminated or truncated as True
# alpha: learning rate for Q-learning update
# gamma: discount factor for Q-learning update
# epsilon: exploration rate for epsilon-greedy action selection
# NOTE: you may want to returns a list of episode rewards obtained during training for debugging purposes
def train_agent(env : gym.Env, agent : QAgent,
                epochs=10000, alpha=0.1, 
                gamma=0.9, epsilon=0.1):
    
    episode_rewards = []

    for _ in range(epochs):

        state, _ = env.reset()
        
        done = False
        total_reward = 0

        while not done:

            # choose action (epsilon-greedy)
            action = agent.act(state, epsilon)

            # take action
            next_state, reward, terminated, truncated, _ = env.step(action)
       
            # update Q-table
            agent.update(state, action, reward, next_state, terminated or truncated, alpha, gamma)

            # move to next state
            state = next_state

            # accumulate reward
            total_reward += reward

            # check episode end
            done = terminated or truncated

        episode_rewards.append(total_reward)

    return episode_rewards
 

# you may want to consider implementing a separate "evaluation" function
# which checks the performance of the trained agent, e.g., using epsilon=0.0
# you will be graded on whether the greedy extracted policy from you self.q_table can successfully solve the task
def eval_agent(env : gym.Env, agent : QAgent, episodes=100):
    epsilon_rewards = []
    for _ in range(episodes): 
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state, epsilon=0.0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            total_reward += reward 
            done = terminated or truncated
        epsilon_rewards.append(total_reward)
    return epsilon_rewards


def q_learning(epochs=10000, alpha=0.1, gamma=0.9, epsilon=0.1, init_val=0, save_weights=False):
    # Create the environment, set the render_mode
    # This render mode means you can print states to debug, see Taxi-v3 documentation for more details
    env = gym.make("Taxi-v3", render_mode="ansi")
    # Initialize our agent
    agent = QAgent(env.observation_space, env.action_space, init_val=init_val)
    # Train
    episode_rewards = train_agent(env, agent, epochs=epochs, alpha=alpha, gamma=gamma, epsilon=epsilon)
    # Evaluation
    eval_agent(env, agent)
    # Save weights
    if save_weights:
        # submit your final saved weights file "QAgent_weights.npy" to Gradescope for grading
        agent.save_weights("QAgent_weights.npy")
    return episode_rewards

if __name__ == "__main__":
    experiments = {
        "pessimistic_init": {
            "epochs": 5000, # is this enough?
            "alpha": 0.1, # is this too big? too small?
            # in lecture we assumed gamma was part of the problem description, here you can tune it...
            "gamma": 0.99, # is this too big? too small?
            "epsilon": 0.1, # what is a good exploration rate for this problem? 
            "init_val": 0, # very pessimistic initialization...
            "save_weights": True 
        }
    }
    # example code for running experiments and plotting the resulting returns
    # NOTE: this code will only work if you return a list of episode rewards from your train_agent function...
    moving_average_window = 10
    moving_average_episode_rewards = {}
    for exp in experiments:
        print(f"Running experiment {exp} with parameters: {experiments[exp]}")
        episode_rewards = q_learning(**experiments[exp])
        # create moving average plot
        if len(episode_rewards) >= moving_average_window:
            moving_average_episode_rewards[exp] = np.convolve(np.array(episode_rewards), np.ones(moving_average_window) / moving_average_window, mode='valid')
        
    for exp, mov_avg_ep_rewards in moving_average_episode_rewards.items():
        plt.plot(mov_avg_ep_rewards, label=exp)

    plt.xlabel("Epochs")
    plt.ylabel("Smoothed Episode Returns")
    plt.legend()
    plt.show()