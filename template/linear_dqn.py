import argparse, json
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from gym_robot import GYM_DUBINS_CAR_ENV_ID

class LinearDQNAgent:
    def __init__(self, num_features, num_actions, target_weights_update_freq=100, max_buffer_size=10000):
        self.target_weights_update_freq = target_weights_update_freq
        self.num_updates = 0
        self.replay_buffer = []
        self.max_buffer_size = max_buffer_size
        # TODO(IV): initialize the linear Q-function weights
        self.weights = np.zeros((num_features, num_actions)) # learn # 改
        self.target_weights = np.copy(self.weights) # calculate target
        
    def save_weights(self, filename):
        np.save(filename, self.weights)

    def load_weights(self, filename):
        self.weights = np.load(filename) 
        self.target_weights = np.copy(self.weights)

    def act(self, features, epsilon=0.1):
        # TODO(IV): implement epsilon-greedy action selection
        # compute Q values
        q_values = features @ self.weights # dot features # 改
        #q_values = [Q(action0), Q(action1), Q(action2)]

        # epsilon-greedy 
        if np.random.rand() < epsilon:
            action = np.random.randint(len(q_values)) # action is an integer
        else: 
            action = np.argmax(q_values)
        return action
#改
    def update(self, features, action, reward, next_features, terminated, alpha=0.1, gamma=0.9, batch_size=64):
        # TODO(IV): implement the replay-buffer DQN update.
        # A standard pattern is:
        # 1. push the new transition into self.replay_buffer
        # 2. once the buffer is large enough, sample a minibatch
        # 3. compute current Q-values and target Q-values
        # 4. apply a linear gradient step to the weights for the chosen action
        # 5. periodically copy self.weights into self.target_weights
        # 根據rewards 更新 weights 
       
        # if len(self.replay_buffer) > self.max_buffer_size:
        #     self.replay_buffer.pop(0)
        # # 2. sample a minibatch 
        # if len(self.replay_buffer) < batch_size:
        #     return 
        # import random
        # batch = random.sample(self.replay_buffer, batch_size)
        # # 3. compute Q values

        # current_q = q_values[action] # 改 本來features@self.weight[action] 
        # # target
        # if terminated:
        #     target = reward
        # else:
        #     target = reward + gamma * np.max(self.target_weights @ next_features)
        # # 4. 
        # error = target - current_q 
        # self.weights[action] += alpha * error * features

        # # 5. 
        # self.num_updates += 1 
        # if self.num_updates % self.target_weights_update_freq == 0:
        #     self.target_weights = np.copy(self.weights)
        self.replay_buffer.append((features, action, reward, next_features, terminated))
        if len(self.replay_buffer) > self.max_buffer_size:
            self.replay_buffer.pop(0)

        if len(self.replay_buffer) < batch_size:
            return

        import random
        batch = random.sample(self.replay_buffer, batch_size)

        for f, a, r, nf, done in batch:

            q_values = f @ self.weights
            current_q = q_values[a]

            if done:
                target = r
            else:
                target = r + gamma * np.max(nf @ self.target_weights)

            error = target - current_q

            self.weights[:, a] += alpha * error * f

        self.num_updates += 1
        if self.num_updates % self.target_weights_update_freq == 0:
            self.target_weights = np.copy(self.weights)
                
def train(
    agent: LinearDQNAgent, envs: list[gym.Env],
    num_episodes, epsilon_start, epsilon_end, epsilon_decay_steps,
    learning_rate, discount_factor, batch_size, output_path: Path | None = None):
# TODO(IV): implement the training loop.
# Recommended structure:
# - iterate over episodes
# - for each environment, reset and roll out until terminated or truncated
# - call agent.act(..., epsilon=...)
# - call env.step(action)
# - call agent.update(...)
# - track episode rewards for plotting/debugging

    episode_rewards = []
    
    for episode in range(num_episodes):
        # 讓agent從多探索到少探索
        epsilon = max(epsilon_end, epsilon_start - episode/epsilon_decay_steps) 
        # 學不同環境，學會泛化
        for env in envs:

            state, _ = env.reset() # 隨機生成起點、狀態、清空狀態
            done = False
            total_reward = 0

            while not done:

                action = agent.act(state, epsilon)

                next_state, reward, terminated, truncated, _ = env.step(action)

                agent.update(
                    state,
                    action,
                    reward,
                    next_state,
                    terminated,
                    alpha=learning_rate,
                    gamma=discount_factor,
                    batch_size=batch_size
                )

                state = next_state
                total_reward += reward
                done = terminated or truncated

            episode_rewards.append(total_reward)

    return episode_rewards
        



def test(agent: LinearDQNAgent, env: gym.Env, output_path: Path | None = None):
    state, _ = env.reset(seed=0)
    total_reward = 0.0
    path = [env.config.copy()]
    done = False
    terminated = False
    truncated = False

    while not done:
        action = agent.act(state, epsilon=0.0)
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        state = next_state
        path.append(env.config.copy())
        done = terminated or truncated

    path = np.array(path)
    fig, _ = env.render(
        path=path,
        title=f"{env.env_name}: Total Reward {total_reward:.3f}",
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return {
        "env_name": env.env_name,
        "total_reward": float(total_reward),
        "num_steps": int(len(path) - 1),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "reached_goal": bool(np.linalg.norm(env.config[:2] - env.goal_config[:2]) <= env.car_params.goal_tolerance),
        "image_path": str(output_path) if output_path is not None else None,
    }

# helpers for loading and creating environments from the provided JSON files
def numeric_env_sort_key(path: Path):
    return int(path.stem.split("_")[-1])
def make_env(params_file: Path):
    return gym.make(
        GYM_DUBINS_CAR_ENV_ID,
        params_file=str(params_file),
        disable_env_checker=True,
    ).unwrapped


def next_results_dir(base_dir: Path):
    results_root = base_dir / "results"
    run_idx = 1
    while (results_root / f"run_{run_idx}").exists():
        run_idx += 1
    return results_root / f"run_{run_idx}"


def print_test_summary(test_results):
    if not test_results:
        print("Test summary: no test results to report.")
        return

    num_successes = sum(result["reached_goal"] for result in test_results)
    mean_reward = float(np.mean([result["total_reward"] for result in test_results]))
    mean_steps = float(np.mean([result["num_steps"] for result in test_results]))
    num_truncated = sum(result["truncated"] for result in test_results)
    failed_envs = [result["env_name"] for result in test_results if not result["reached_goal"]]

    print(f"Test summary: {num_successes}/{len(test_results)} successful episodes")
    print(f"Mean reward: {mean_reward:.3f}, Mean steps: {mean_steps:.1f}, Truncated episodes: {num_truncated}")
    if failed_envs:
        print("Failed environments:", ", ".join(failed_envs))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS440 MP5 Reinforcement Learning")
    parser.add_argument("--num_episodes", type=int, default=20000, help="Number of training episodes")
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="Initial epsilon value for epsilon-greedy exploration")
    parser.add_argument("--epsilon_end", type=float, default=0.05, help="Final epsilon value for epsilon-greedy exploration")
    parser.add_argument("--epsilon_decay_steps", type=int, default=None, help="Number of episodes over which to linearly decay epsilon; defaults to num_episodes")
    parser.add_argument("--target_weights_update_freq", type=int, default=100, help="Number of updates between target network weight updates")
    parser.add_argument("--max_buffer_size", type=int, default=50000, help="Maximum size of the replay buffer")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the DQN")
    parser.add_argument("--discount_factor", type=float, default=0.95, help="Discount factor for the DQN")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for the DQN")
    parser.add_argument("--test_only", action="store_true", help="Skip training and only run testing with loaded weights")
    parser.add_argument("--weights_file", type=str, default="linear_dqn_weights.npy", help="Weights file to load for --test_only or --resume_training; you can also point this at a saved run such as results/run_3/linear_dqn_weights.npy")
    parser.add_argument("--resume_training", action="store_true", help="Load weights before training and continue optimizing from them")
    parser.add_argument("--results_dir", type=str, default=None, help="Optional directory for saved plots and test outputs")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    train_param_files = sorted(data_dir.glob("train_env_*.json"), key=numeric_env_sort_key)
    test_param_files = sorted(data_dir.glob("test_env_*.json"), key=numeric_env_sort_key)

    if not test_param_files:
        raise FileNotFoundError(f"No test environments found in {data_dir}")
    if not args.test_only and not train_param_files:
        raise FileNotFoundError(
            f"No training environments found in {data_dir}. Create files named train_env_*.json before training."
        )

    print(f"Loading {len(train_param_files)} training environments and {len(test_param_files)} test environments")
    train_envs = [make_env(params_file) for params_file in train_param_files]
    test_envs = [make_env(params_file) for params_file in test_param_files]

    if args.results_dir is not None:
        results_dir = Path(args.results_dir)
        if not results_dir.is_absolute():
            results_dir = base_dir / results_dir
    else:
        results_dir = next_results_dir(base_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    example_env = train_envs[0] if train_envs else test_envs[0]
    if len(example_env.observation_space.shape) != 1:
        raise ValueError(f"Expected a flat observation vector, got shape {example_env.observation_space.shape}")
    num_features = int(example_env.observation_space.shape[0])
    agent = LinearDQNAgent(
        num_features=num_features,
        num_actions=int(example_env.action_space.n),
        target_weights_update_freq=args.target_weights_update_freq,
        max_buffer_size=args.max_buffer_size,
    )

    weights_path = Path(args.weights_file)
    if not weights_path.is_absolute():
        weights_path = base_dir / weights_path
    if args.test_only or args.resume_training:
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Weights file not found: {weights_path}. "
                "Copy a saved run's linear_dqn_weights.npy into the template directory or pass --weights_file with the run-specific path."
            )
        agent.load_weights(weights_path)

    if not args.test_only:
        num_episodes = args.num_episodes
        epsilon_decay_steps = args.epsilon_decay_steps if args.epsilon_decay_steps is not None else num_episodes
        train(
            agent,
            train_envs,
            num_episodes=num_episodes,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay_steps=epsilon_decay_steps,
            learning_rate=args.learning_rate,
            discount_factor=args.discount_factor,
            batch_size=args.batch_size,
            output_path=results_dir / "training_returns.png",
        )

    test_results = []
    for env in test_envs:
        test_results.append(test(agent, env, output_path=results_dir / f"{env.env_name}_path.png"))

    print_test_summary(test_results)

    with open(results_dir / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    if not args.test_only:
        saved_weights_path = results_dir / "linear_dqn_weights.npy"
        agent.save_weights(saved_weights_path)
        print(f"Saved weights to {saved_weights_path}")
        print(
            "To reuse these weights later, either copy this file to "
            f"{base_dir / 'linear_dqn_weights.npy'} or pass --weights_file {saved_weights_path}"
        )
