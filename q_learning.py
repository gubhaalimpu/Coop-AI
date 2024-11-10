import numpy as np
import gym
from gym import spaces
import random
import matplotlib.pyplot as plt

class PoliticalDebateEnv(gym.Env):
    def __init__(self):
        super(PoliticalDebateEnv, self).__init__()
        self.num_agents_per_party = 5
        self.action_space = spaces.Discrete(2)  # 0 = Do not misinform, 1 = Misinform
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)  # Simple observation
        
        # Initialize states and rewards
        self.state = np.random.rand(1)  # Random initial state
        self.num_steps = 0
        self.agents = ["party_yellow", "party_green"]
    
    def reset(self):
        self.state = np.random.rand(1)  # Reset state for a new episode
        self.num_steps = 0
        self.defection_counts = {agent: [0] * self.num_agents_per_party for agent in self.agents}
        return {f"{agent}_{i}": self.state for agent in self.agents for i in range(self.num_agents_per_party)}
    
    def step(self, actions):
        self.num_steps += 1
        rewards = {agent: 0 for agent in self.agents}
        defection_counts = {agent: [0] * self.num_agents_per_party for agent in self.agents}

        # Calculate payoffs and update defection counts
        for agent in self.agents:
            for i in range(self.num_agents_per_party):
                if actions[f"{agent}_{i}"] == 1:  # If the agent misinforms
                    defection_counts[agent][i] += 1
                    rewards[agent] += 5 if agent == "party_yellow" else -5  # Example payoff for misinforming
                else:
                    rewards[agent] -= 1  # Example payoff for not misinforming

        done = self.num_steps >= 50  # End the episode after 50 steps
        return {f"{agent}_{i}": self.state for agent in self.agents for i in range(self.num_agents_per_party)}, rewards, {f"{agent}_{i}": done for agent in self.agents for i in range(self.num_agents_per_party)}, defection_counts
    
    def render(self):
        print(f"Step: {self.num_steps}, State: {self.state}")

    def close(self):
        pass

# Q-learning setup
class QLearning:
    def __init__(self):
        self.env = PoliticalDebateEnv()
        self.q_tables = {f"{agent}_{i}": np.zeros((1, 2)) for agent in self.env.agents for i in range(self.env.num_agents_per_party)}  # Q-table for each agent
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.exploration_rate = 1.0
        self.exploration_decay = 0.995
        self.min_exploration_rate = 0.01

        # Track defections every 100 episodes
        self.defection_history = {"party_yellow": [], "party_green": []}

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
            done = {f"{agent}_{i}": False for agent in self.env.agents for i in range(self.env.num_agents_per_party)}
            episode_rewards = {agent: 0 for agent in self.env.agents}
            defection_counts = {agent: [0] * self.env.num_agents_per_party for agent in self.env.agents}  # Reset defection counts

            while not all(done.values()):
                actions = {f"{agent}_{i}": self.choose_action(f"{agent}_{i}") for agent in self.env.agents for i in range(self.env.num_agents_per_party)}
                next_state, rewards, done, defection = self.env.step(actions)

                # Track defection counts
                for agent in self.env.agents:
                    for i in range(self.env.num_agents_per_party):
                        defection_counts[agent][i] += defection[agent][i]
                    episode_rewards[agent] += rewards[agent]

                for agent in self.env.agents:
                    for i in range(self.env.num_agents_per_party):
                        self.update_q_values(f"{agent}_{i}", state[f"{agent}_{i}"], actions[f"{agent}_{i}"], rewards[agent], next_state[f"{agent}_{i}"])

                state = next_state

            # Decay exploration rate
            self.decay_exploration()

            # Track defections every 100 episodes (non-cumulative)
            if episode % 100 == 0:
                yellow_defections = sum(defection_counts["party_yellow"])
                green_defections = sum(defection_counts["party_green"])
                self.defection_history["party_yellow"].append(yellow_defections)
                self.defection_history["party_green"].append(green_defections)

            if episode % 100 == 0:
                print(f"Episode {episode} - Rewards: {episode_rewards}")
                print(f"Defections (misinformations) - Party Yellow: {sum(defection_counts['party_yellow'])}, Party Green: {sum(defection_counts['party_green'])}")

        self.env.close()
        self.plot_defections()

    def plot_defections(self):
        # Plotting defections for both parties every 100 episodes
        episodes = [i * 100 for i in range(len(self.defection_history["party_yellow"]))]

        plt.figure(figsize=(10, 6))
        plt.plot(episodes, self.defection_history["party_yellow"], label="Party Yellow Defections", color='yellow', marker='o')
        plt.plot(episodes, self.defection_history["party_green"], label="Party Green Defections", color='green', marker='x')

        plt.title("Defections by Each Party Every 100 Episodes")
        plt.xlabel("Episodes (in multiples of 100)")
        plt.ylabel("Defections (not cumulative)")
        plt.legend()
        plt.grid(True)
        plt.show()

# Instantiate and train the agent
ql = QLearning()
ql.train(episodes=1000)
