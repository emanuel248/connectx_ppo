
import gym
import random
from kaggle_environments import evaluate, make

class ConnectX(gym.Env):
    def __init__(self, switch_prob=0.5, random_agent=False, test_mode=False):
        self.env = make('connectx', debug=True)
        if random.uniform(0, 1) < 0.5:
            self.pair = [None, 'negamax']
        else:
            self.pair = [None, 'random']

        #test setup
        if random_agent and test_mode:
            self.pair = [None, 'random']
        elif test_mode:
            self.pair = [None, 'negamax']

        self.trainer = self.env.train(self.pair)
        self.switch_prob = switch_prob
        
        # Define required gym fields (examples):
        config = self.env.configuration
        self.action_space = gym.spaces.Discrete(config.columns)
        self.observation_space = gym.spaces.Discrete(config.columns * config.rows)

    def switch_trainer(self):
        self.pair = self.pair[::-1]
        self.trainer = self.env.train(self.pair)

    def step(self, action):
        return self.trainer.step(int(action))
    
    def reset(self):
        if random.uniform(0, 1) < self.switch_prob:
            self.switch_trainer()
        return self.trainer.reset()
    
    def render(self, **kwargs):
        return self.env.render(**kwargs)