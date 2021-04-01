from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import shutil
import types
import numbers

def str_numbers(nums, prefix="", sufix=""):
    if isinstance(nums,numbers.Number):
        return prefix + '{: 7.3f}'.format(nums) + sufix
    out_str = prefix + '['
    for n in nums[:-1]:
        out_str += '{: 7.3f} '.format(n)
    return out_str + '{: 7.3f}]'.format(nums[-1]) + sufix

def print_sarsa(sarsa):
    for ind_sar in range(len(sarsa)//3):
        s = sarsa[ind_sar*3]
        a = sarsa[ind_sar*3+1]
        r = sarsa[ind_sar*3+2]
        for i_agent in range(2):
            agent_state = s[i_agnet,:].reshape((3,-1))
            print(str_numbers(agent_state[0,:],prefix='['))
            print(str_numbers(agent_state[1,:],prefix=' '))
            print(str_numbers(agent_state[2,:],prefix=' ',sufix=']'),str_numbers(a[i_agent]),str_numbers(r[i_agent]))
            pass
        pass
        print("")
    pass

class unity_env(UnityEnvironment):

    def __init__(self, file_name, ind_brain=0, no_graphics=False, verbose_level=1, max_steps_in_single_episode=2000, score_goal=30,\
                 score_window_size=100, seed=1, set_modify_state=None, set_modify_reward=None):
        super().__init__(file_name=file_name, no_graphics=no_graphics, seed=seed)
        if set_modify_state is not None:
            self.modify_state = types.MethodType(set_modify_state,self)
        if set_modify_reward is not None:
            self.modify_reward = types.MethodType(set_modify_reward,self)
        self.ind_brain     = ind_brain        
        self.brain_name    = super().brain_names[ind_brain]
        self.brain         = super().brains[self.brain_name]
        self.action_size   = self.brain.vector_action_space_size
        self.max_steps_in_single_episode = max_steps_in_single_episode
        self.score_goal    = score_goal
        self.goal          = False
        self.verbose_level = verbose_level
        states             = self.reset(train_mode=True)
        self.num_agents    = states.shape[0]
        self.state_shape   = states.shape[1:]
        if self.verbose_level > 0:
            print("Exploring Unity environment...")
            print("Selected brain name: ", self.brain_name)
            print("Selected brain:      ", self.brain)
            print("Number of actions:   ", self.action_size)
            print("Number of agents:    ", self.num_agents)
            print("States have shape:   ", self.state_shape)
            for i in range(self.num_agents):
                print("State of agent",i,"look like:\n",str_numbers(states[i]))
        self.reset_all_scores(score_window_size)
    
    
    def train(self, agent, output_filename, max_num_episodes = 10000, num_episode_search = 500, score_window_size=None,\
              noise_minimal=0.01, noise_decay=0.99):
        """DDPG learning.

        Params
        ======
            agent (ddpg_agent):       the agent
            output_filename (string): file name for check point
            max_num_episodes (int):   maximum number of training episodes
            num_episode_search (int): maximum number of episode to search for better score
            noise_minimal (float):    minimum value of noise multiplier
            noise_decay (float):      multiplicative factor (per episode) for decreasing noise
        """
        self.reset_all_scores(score_window_size)
        if num_episode_search is None or (num_episode_search < self.scores_window.maxlen):
            num_episode_search = self.scores_window.maxlen
        if self.verbose_level>0:
            print("Training DDPG agent:")
            print("output_filename    =",output_filename)
            print("noise_minimal      =",noise_minimal)
            print("noise_decay        =",noise_decay)
            print("num_episode_search =",num_episode_search)
            print("max_num_episodes   =",max_num_episodes)
            print("score_window_size  =",self.scores_window.maxlen)

        agent.reset_noise_level()                    # initialize noise
        len_save_scores = 0
        while self.improvement >= 0:
            #curr_theta = agent.noise.theta
            #curr_sigma = agent.noise.sigma
            #curr_noise_level = agent.get_noise_level()
            self.train_agent_on_single_episode(agent)
            real_episode_score = [s for s in np.asarray(self.episode_score) if abs(s) > 1e-8]
            real_episode_score = 0 if len(real_episode_score) < 1 else np.max(real_episode_score)
            # decay noise only if new episode score is smaller than previous episode score
            if np.abs(real_episode_score) > 0 and len(self.all_scores) > 0 and real_episode_score < self.all_scores[-1]:
                agent.scale_noise(noise_decay)
            optimistic_episode_score = np.max(self.episode_score)
            current_score_is_ok = optimistic_episode_score > self.curr_window_average # compare value before updating window
            self.scores_window.append(optimistic_episode_score)
            self.all_scores.append(real_episode_score)
            self.__recalc_curr_window_score()
            self.improvement  = 0 # train may continue
            num_episodes  = len(self.all_scores)
            if num_episodes >= self.scores_window.maxlen:
                if self.curr_window_score > self.best_window_score and current_score_is_ok:
                    # real score improvement :-)
                    self.improvement = 1 # train should continue
                    self.__set_curr_score_window_as_best()
                elif self.curr_window_average > self.temp_window_average:
                    # no real score improvement but still try a bit more...
                    self.temp_window_average = self.curr_window_average
                    self.best_window_episode = num_episodes
                elif num_episodes >= self.best_window_episode + num_episode_search:
                    # no score improvement anymore :-(
                    self.improvement = -1 # train should stop
            if self.verbose_level > 1:
                print('Episode {:3} --> score={:5.2f} curr window: {}'.format(num_episodes, real_episode_score,\
                                                                              self.str_curr_window_score()))
            elif self.verbose_level > 0:
                end         = "\n" if self.improvement > 0 else ""
                print('\rEpisode {} curr={: 5.2f} \t{} \t{} '.format(num_episodes, real_episode_score,\
                                                                     self.str_curr_window_score(),agent.memory.statistics()), end=end)
            if self.__goal_reached_first_time():
                print('\nEnvironment goal reached in {:d} episodes!'.format(num_episodes))
            if num_episodes >= max_num_episodes:
                # rare case: maximum number of episodes  -->  end training loop in any case
                break
            if self.improvement > 0:
                print('Saving checkpoint...')
                agent.save(output_filename)
                len_save_scores = len(self.all_scores)
            
        print('\nNo more improvements. End of training.')
        return self.all_scores[:len_save_scores], self.best_test_score, self.best_test_average, self.best_test_stdev

    def reset_all_scores(self, score_window_size=None):
        if score_window_size is None:
            score_window_size = self.scores_window.maxlen
        self.all_scores        = []                              # list containing scores from each episode
        self.scores_window     = deque(maxlen=score_window_size) # last recent scores
        self.__recalc_curr_window_score()
        self.__set_curr_score_window_as_best()
        self.__set_curr_score_window_as_best_test()
        self.improvement       = 0
        
    def train_agent_on_single_episode(self, agent, max_steps_in_single_episode=None):
        if max_steps_in_single_episode is None:
            max_steps_in_single_episode = self.max_steps_in_single_episode
        agent.prepare_for_new_episode()
        state = self.reset(train_mode=True) # reset the environment and current episode score
        for i_step in range(max_steps_in_single_episode):
            action = agent.act(state, add_noise=True) # select an action with noise exploration
            next_state, reward, done, _ = self.step(action)
            agent.step(state, reward, next_state, done)
            if np.any(done):
                break
            state  = next_state
        
    
    def test_agent_on_single_episode(self, agent, max_steps_in_single_episode=None, verbose=True):
        if verbose:
            print("Agent: Test on single episode starts...")
        if max_steps_in_single_episode is None:
            max_steps_in_single_episode = self.max_steps_in_single_episode
        state = self.reset(train_mode=False) # reset the environment
        for i_step in range(max_steps_in_single_episode):
            action            = agent.act(state)  # select an action
            state, _, done, _ = self.step(action)
            if np.any(done):
                break
        if verbose:
            print("Agent: Test on single episode is over. Number of steps: {}  score: {:5.2f}"\
                  .format(self.episode_length, np.max(self.episode_score)))
        return self.episode_score
    
    def test_random_agent_on_single_episode(self, agent, max_steps_in_single_episode=None, verbose=True):
        if verbose:
            print("Agent: Test on single episode starts...")
        if max_steps_in_single_episode is None:
            max_steps_in_single_episode = self.max_steps_in_single_episode
        state = self.reset(train_mode=True) # reset the environment
        for i_step in range(max_steps_in_single_episode):
            action            = agent.random_action()    # select an action
            state, _, done, _ = self.step(action)
            if np.any(done):
                break
        episode_score = np.max(self.episode_score)
        if verbose:
            print("Agent: Test on single episode is over. Number of steps: {}  score: {:5.2f}"\
                  .format(self.episode_length, episode_score))
        return self.episode_score
    
    def test_agent_on_many_episodes(self, agent, max_num_episodes=None, max_steps_in_single_episode=None, verbose=True):
        if max_num_episodes is None:
            max_num_episodes = self.scores_window.maxlen
        if verbose:
            print("Begin test agent on {} episodes...".format(max_num_episodes))
        if max_steps_in_single_episode is None:
            max_steps_in_single_episode = self.max_steps_in_single_episode
        scores = []
        for i_episode in range(max_num_episodes):
            state = self.reset(train_mode=True) # reset the environment
            for i_step in range(max_steps_in_single_episode):
                action            = agent.act(state)     # select an action with eploitation only
                state, _, done, _ = self.step(action)
                if np.any(done):
                    break
            episode_score = np.max(self.episode_score)
            scores.append(episode_score)
            if verbose:
                print("\rEpisode: {} out of: {} ended with score: {:5.2f}".format(i_episode+1,\
                                                                                  max_num_episodes,episode_score), end="")
        scores_mean      = np.mean(scores)
        scores_stdev     = np.std(scores,ddof=1) if len(scores) > 1 else np.inf
        scores_composite = scores_mean - scores_stdev
        if verbose:
            print("\nTest results on {} episodes: Composite={:5.2f} Average={:5.2f} Stdev={:5.2f}".format(max_num_episodes,\
                                                                                                          scores_composite,\
                                                                                                          scores_mean, scores_stdev))
        return scores_composite, scores_mean, scores_stdev, scores

    def reset(self, train_mode):
        env_info            = super().reset(train_mode=train_mode)[self.brain_name] # reset the environment
        self.save_states    = np.asarray([self.modify_state(s) for s in env_info.vector_observations])  # get the current states
        self.episode_length = 0
        self.sarsa          = [self.save_states]
        num_agents          = self.save_states.shape[0]
        if num_agents > 1:
            self.episode_score = np.zeros(num_agents, dtype=np.float32)
            return self.save_states
        else:
            self.episode_score = 0
            return self.save_states[0]
    
    def step(self, action):
        self.sarsa.append(np.asarray(action))
        env_info             = super().step(action)[self.brain_name]          # send the action to the environment
        next_state           = np.asarray([self.modify_state(env_info.vector_observations[s],self.save_states[s]) for s in range(self.num_agents)])
        reward               = np.asarray([self.modify_reward(r) for r in env_info.rewards], dtype=np.float32) # get the rewards
        self.sarsa.append(reward)
        self.sarsa.append(next_state)
        if self.num_agents < 2:
            next_state = next_state[0]
            reward     = reward[0]
            done       = env_info.local_done[0]          # see if episode has finished
        else:
            done       = env_info.local_done          # see if episode has finished
        self.episode_length += 1
        self.episode_score  += reward
        return next_state, reward, done, env_info

    def modify_state(self,state,prev_state=None):
        return state
    
    def modify_reward(self,reward):
        return reward
    
    
    def str_curr_window_score(self):
        if len(self.all_scores) < 1:
            return "No data yet..."
        return "Average={:5.3f} Stdev={:5.3f} Composite={:5.3f} Max={:5.3f}".format(self.curr_window_average,\
                                                                                          self.curr_window_stdev,\
                                                                                          self.curr_window_score,\
                                                                                          self.curr_window_max)

    def __test_improvement(self, agent, output_filename):
        print("\nCompare last checkpoint against best test score so far...");
        score_window_size = self.scores_window.maxlen
        for i_episode in range(score_window_size):
            state = self.reset(train_mode=True) # reset the environment
            done  = False
            while not np.any(done):
                action            = agent.act(state)     # select an action
                state, _, done, _ = self.step(action)
            self.scores_window.append(np.max(self.episode_score))
            print("\rEpisode: {} out of: {}".format(i_episode+1,score_window_size),end="")
        print("\n")
        self.__recalc_curr_window_score()
        if self.curr_window_score <= self.best_test_score:
            print("Current test score {:5.2f} is worse than previous test score {:5.2f}...".format(self.curr_window_score,\
                                                                                                   self.best_test_score))
            return False
        print("Current test score {:5.2f} is better than previous test score {:5.2f}...".format(self.curr_window_score,\
                                                                                                self.best_test_score))        
        # saving test score
        self.__set_curr_score_window_as_best_test()
        self.__set_curr_score_window_as_best()
        self.improvement = 0 # train may continue
        agent.save(output_filename)
        return True
        
    def __recalc_curr_window_score(self):
        curr_len = len(self.scores_window)
        self.curr_window_average = np.mean(self.scores_window) if curr_len > 0 else -np.inf
        # estimate standard deviation of entire population
        self.curr_window_stdev   = np.std(self.scores_window,ddof=1) if curr_len > 1 else np.inf
        self.curr_window_score   = self.curr_window_average - self.curr_window_stdev
        self.curr_window_max     = np.max(self.scores_window) if curr_len > 0 else -np.inf
        
    def __set_curr_score_window_as_best(self):
        self.best_window_score   = self.curr_window_score
        self.best_window_average = self.curr_window_average
        self.best_window_stdev   = self.curr_window_stdev
        self.temp_window_average = self.curr_window_average
        self.best_window_episode = len(self.all_scores) # used for check number of episodes since best episode
 
    def __set_curr_score_window_as_best_test(self):
        self.best_test_score   = self.curr_window_score
        self.best_test_average = self.curr_window_average
        self.best_test_stdev   = self.curr_window_stdev

    def __goal_reached_first_time(self):
        if self.goal:
            return False
        self.goal = self.score_goal is not None and self.curr_window_average is not None\
        and self.curr_window_average >= self.score_goal
        return self.goal
    
