import numpy as np
import gym
from gym import spaces
import random
import matplotlib.pyplot as plt
from prettytable import PrettyTable

class PoliticalDebateEnv(gym.Env):
    def __init__(self):
        super(PoliticalDebateEnv, self).__init__()
        self.num_agents_per_party = 5
        self.action_space = spaces.Discrete(2)  # 0 = Cooperate, 1 = Defect (Misinform)
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        # Initialize states and rewards
        self.state = np.random.rand(1)
        self.num_steps = 0
        self.agents = ["party_yellow", "party_green"]
        
    def reset(self):
        self.state = np.random.rand(1)
        self.num_steps = 0
        self.defection_counts = {agent: [0] * self.num_agents_per_party for agent in self.agents}
        return {f"{agent}_{i}": self.state for agent in self.agents for i in range(self.num_agents_per_party)}
    
    def step(self, actions):
        self.num_steps += 1
        rewards = {agent: 0 for agent in self.agents}
        defection_counts = {agent: [0] * self.num_agents_per_party for agent in self.agents}

        # Random pairing of agents
        yellow_indices = list(range(self.num_agents_per_party))
        green_indices = list(range(self.num_agents_per_party))
        random.shuffle(green_indices)  # Shuffle to pair agents randomly

        # Evaluate payoffs for each pair based on Prisoner's Dilemma
        for i in range(self.num_agents_per_party):
            yellow_action = actions[f"party_yellow_{yellow_indices[i]}"]
            green_action = actions[f"party_green_{green_indices[i]}"]

            # Prisoner's Dilemma payoff logic
            if yellow_action == 1 and green_action == 1:  # Both defect
                rewards["party_yellow"] -= 2
                rewards["party_green"] -= 2
                defection_counts["party_yellow"][yellow_indices[i]] += 1
                defection_counts["party_green"][green_indices[i]] += 1
            elif yellow_action == 1 and green_action == 0:  # Yellow defects, Green cooperates
                rewards["party_yellow"] += 3
                rewards["party_green"] -= 1
                defection_counts["party_yellow"][yellow_indices[i]] += 1
            elif yellow_action == 0 and green_action == 1:  # Yellow cooperates, Green defects
                rewards["party_yellow"] -= 1
                rewards["party_green"] += 3
                defection_counts["party_green"][green_indices[i]] += 1
            else:  # Both cooperate
                rewards["party_yellow"] += 1
                rewards["party_green"] += 1

        done = self.num_steps >= 50
        return (
            {f"{agent}_{i}": self.state for agent in self.agents for i in range(self.num_agents_per_party)},
            rewards,
            done,
            defection_counts
        )
    
    def render(self):
        print(f"Step: {self.num_steps}, State: {self.state}")
    

# Q-learning setup
class QLearning:
    def __init__(self):
        self.env = PoliticalDebateEnv()
        self.q_tables = {f"{agent}_{i}": np.zeros((1, 2)) for agent in self.env.agents for i in range(self.env.num_agents_per_party)}
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.exploration_rate = 1.0
        self.exploration_decay = 0.995
        self.min_exploration_rate = 0.01
        self.episode_rewards = []
        self.defections_over_time = []
        self.exploration_rate_over_time = []

    def choose_action(self, agent):
        if random.uniform(0, 1) < self.exploration_rate:
            return self.env.action_space.sample()  # Explore
        else:
            return np.argmax(self.q_tables[agent][0])  # Exploit

    def update_q_values(self, agent, old_state, action, reward, next_state):
        old_q_value = self.q_tables[agent][0, action]
        next_max_q = np.max(self.q_tables[agent][0])
        self.q_tables[agent][0, action] = old_q_value + self.learning_rate * (reward + self.discount_factor * next_max_q - old_q_value)

    def decay_exploration(self):
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)

    def train(self, episodes=1000):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            episode_rewards = {agent: 0 for agent in self.env.agents}
            defection_counts = {agent: 0 for agent in self.env.agents}  # Track defections per episode

            while not done:
                actions = {f"{agent}_{i}": self.choose_action(f"{agent}_{i}") for agent in self.env.agents for i in range(self.env.num_agents_per_party)}
                next_state, rewards, done, defections = self.env.step(actions)

                # Track episode rewards and defections
                for agent in self.env.agents:
                    episode_rewards[agent] += rewards[agent]
                    defection_counts[agent] += sum(defections[agent])

                for agent in self.env.agents:
                    for i in range(self.env.num_agents_per_party):
                        self.update_q_values(f"{agent}_{i}", state[f"{agent}_{i}"], actions[f"{agent}_{i}"], rewards[agent], next_state[f"{agent}_{i}"])

                state = next_state

            self.episode_rewards.append(episode_rewards)
            self.defections_over_time.append(defection_counts)
            self.exploration_rate_over_time.append(self.exploration_rate)
            self.decay_exploration()

            # Print a summary every 100 episodes
            if (episode + 1) % 100 == 0:
                self.display_summary(episode + 1, episode_rewards, defection_counts)

        self.plot_results()

    def display_summary(self, episode, rewards, defections):
        table = PrettyTable()
        table.field_names = ["Episode", "Party", "Rewards", "Defections", "Exploration Rate"]
        table.add_row([episode, "Yellow", rewards["party_yellow"], defections["party_yellow"], self.exploration_rate])
        table.add_row([episode, "Green", rewards["party_green"], defections["party_green"], self.exploration_rate])
        print(table)

    def plot_results(self):
        episodes = np.arange(1, len(self.episode_rewards) + 1)
        yellow_rewards = [r["party_yellow"] for r in self.episode_rewards]
        green_rewards = [r["party_green"] for r in self.episode_rewards]
        yellow_defections = [d["party_yellow"] for d in self.defections_over_time]
        green_defections = [d["party_green"] for d in self.defections_over_time]

        fig, ax = plt.subplots(3, 1, figsize=(12, 8))

        # Plot rewards
        ax[0].plot(episodes, yellow_rewards, label="Party Yellow Rewards", color="gold")
        ax[0].plot(episodes, green_rewards, label="Party Green Rewards", color="green")
        ax[0].set_title("Rewards per Episode")
        ax[0].set_xlabel("Episodes")
        ax[0].set_ylabel("Rewards")
        ax[0].legend()

        # Plot defections
        ax[1].plot(episodes, yellow_defections, label="Party Yellow Defections", color="gold")
        ax[1].plot(episodes, green_defections, label="Party Green Defections", color="green")
        ax[1].set_title("Defections per Episode")
        ax[1].set_xlabel("Episodes")
        ax[1].set_ylabel("Defections")
        ax[1].legend()
        
        ax[2].plot(episodes, self.exploration_rate_over_time, label="Exploration Rate", color="purple")
        ax[2].set_title("Exploration Rate over time")
        ax[2].set_xlabel("Episodes")
        ax[2].set_ylabel("Exploration Rate")
        ax[2].legend()        
        

        plt.tight_layout()
        plt.show()

# Instantiate and train the agent
ql = QLearning()
ql.train(episodes=1000)
