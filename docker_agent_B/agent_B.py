from collections import defaultdict
import queue
import random

import numpy as np
from pommerman.agents import BaseAgent
from pommerman.runner import DockerAgentRunner
from pommerman import agents
import constants
import utility
from available_check import *
import available_check
our_testing_prob = np.array([0.20, 0.17, 0.21, 0.23, 0.17, 0.02])

class MyAgent(DockerAgentRunner):
    '''An example Docker agent class'''

    def __init__(self):
        # model_path = 'saved_models/a3c_multi_agent'
        # master_network = AC_Network('global', None)
        # saver = tf.train.Saver()
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # with tf.Session(config=config) as sess:
        #     ckpt = tf.train.get_checkpoint_state(model_path)
        #     saver.restore(sess, ckpt.model_checkpoint_path)
        # self.step_count = 1
        # self.ratio = self.step_count / 800.
        # self._agent = master_network
        self._agent = agents.SimpleAgent()
        self._recently_visited_positions = []
        self._recently_visited_length = 6
        self._prev_direction = None
        self.action_pro = our_testing_prob
    def init_agent(self, id, game_type):
        return self._agent.init_agent(id, game_type)

    def act(self, obs, action_space):
        def convert_bombs(bomb_map):
            ret = []
            locations = np.where(bomb_map > 0)
            for r, c in zip(locations[0], locations[1]):
                ret.append({
                    'position': (r, c),
                    'blast_strength': int(bomb_map[(r, c)])
                })
            return ret

        self.action_pro = our_testing_prob
        always_list = [0, 1, 2, 3, 4, 5]
        action_pro = np.array([0.20, 0.17, 0.21, 0.23, 0.17, 0.02])
        my_position = tuple(obs['position'])
        board = np.array(obs['board'])
        bombs = convert_bombs(np.array(obs['bomb_blast_strength']))
        enemies = []
        for e in obs['enemies']:
            enemies.append(e)
        ammo = int(obs['ammo'])
        blast_strength = int(obs['blast_strength'])

        items, dist, prev = available_check._djikstra(
            board, my_position, bombs, enemies, depth=10)
        unsafe_directions = available_check._directions_in_range_of_bomb(
            board, my_position, bombs, dist)
        if unsafe_directions:
            directions = available_check._find_safe_directions(
                board, my_position, unsafe_directions, bombs, enemies)
            # action_list = [direct.value for direct in directions]
            # for i in range(6):
            #     action_pro[i] = action_pro[i] if i in action_list else 0
            # action_pro /= np.sum(action_pro)
            # return np.random.choice(always_list, p=action_pro)
            return int(random.choice(directions).value)

        if available_check._is_adjacent_enemy(items, dist, enemies) and available_check._maybe_bomb(
                ammo, blast_strength, items, dist, my_position):
            return int(constants.Action.Bomb.value)

        direction = available_check._near_enemy(my_position, items, dist, prev, enemies, 3)
        if direction is not None and (self._prev_direction != direction or
                                      random.random() < .5):
            self._prev_direction = direction
            return int(direction.value)

        direction = available_check._near_good_powerup(my_position, items, dist, prev, 2)
        if direction is not None:
            return int(direction.value)

        if available_check._near_wood(my_position, items, dist, prev, 1):
            if available_check._maybe_bomb(ammo, blast_strength, items, dist, my_position):
                return int(constants.Action.Bomb.value)
            else:
                return int(constants.Action.Stop.value)

        direction = available_check._near_wood(my_position, items, dist, prev, 2)
        if direction is not None:
            directions = available_check._filter_unsafe_directions(board, my_position,
                                                   [direction], bombs)
            if directions:
                action_list = [direct.value for direct in directions]
                for i in range(6):
                    action_pro[i] = action_pro[i] if i in action_list else 0
                action_pro /= np.sum(action_pro)
                return int(np.random.choice(always_list, p=action_pro))

        directions = [
            constants.Action.Stop, constants.Action.Left,
            constants.Action.Right, constants.Action.Up, constants.Action.Down
        ]
        valid_directions = available_check._filter_invalid_directions(
            board, my_position, directions, enemies)
        directions = available_check._filter_unsafe_directions(board, my_position,
                                               valid_directions, bombs)
        directions = available_check._filter_recently_visited(
            directions, my_position, self._recently_visited_positions)
        if len(directions) > 1:
            directions = [k for k in directions if k != constants.Action.Stop]
        if not len(directions):
            directions = [constants.Action.Stop]

        self._recently_visited_positions.append(my_position)
        self._recently_visited_positions = self._recently_visited_positions[
            -self._recently_visited_length:]



        if directions:
            action_list = [direct.value for direct in directions]
            for i in range(6):
                action_pro[i] = action_pro[i] if i in action_list else 0
            action_pro /= np.sum(action_pro)
            return int(np.random.choice(always_list, p=action_pro))
        else:
            return int(random.choice(directions).value)


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
