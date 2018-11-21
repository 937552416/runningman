from pommerman import agents
from pommerman.runner import DockerAgentRunner

import numpy as np
from pommerman.constants import BOARD_SIZE

def sample_dist(dists):
    actions = []
    distss = dists[0]
    for dist in distss:
        act = np.random.choice(dist, p=dist)
        act = np.argmax(dist == act)
        actions.append(act)
    return actions

class MyAgent(DockerAgentRunner):
    '''An example Docker agent class'''

    def __init__(self):
        # model_path = 'saved_models/a3c_multi_agent'
        # network = AC_Network('global', None)
        # saver = tf.train.Saver()
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # with tf.Session(config=config) as sess:
        #     ckpt = tf.train.get_checkpoint_state(model_path)
        #     saver.restore(sess, ckpt.model_checkpoint_path)
        # self.step_count = 1
        # self.ratio = self.step_count / 800.
        # self._agent = network
        self._agent = agents.SimpleAgent()

    def init_agent(self, id, game_type):
        return self._agent.init_agent(id, game_type)

    def act(self, observation, action_space):
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # with tf.Session(config=config) as sess:
        #     agent_obs = np.concatenate([featurize(observation, self.ratio), featurize(observation, self.ratio)],-1)
        #     actions = sample_dist(action_dist)
        list = np.array([0.20, 0.17, 0.21, 0.23, 0.17, 0.02])
        # list = np.array([2.0131826e-01, 1.5195006e-01, 1.9724846e-01, 3.0026057e-01, 1.4897382e-01, 2.4885626e-04])
        action = np.random.choice(list, p=list)
        action = np.argmax(list == action)
        return int(action)
        #     return actions[0]

    def episode_end(self, reward):
        self.step_count = 1
        self.ratio = self.step_count / 800.
        return self._agent.episode_end(reward)

    def shutdown(self):
        return self._agent.shutdown()


def main():
    '''Inits and runs a Docker Agent'''
    agent = MyAgent()
    agent.run()


if __name__ == "__main__":
    main()
