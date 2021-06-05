import numpy as np
import types
import numbers
# some users might not install torch at the first place
#import torch # optional import

class ReplayBufferBatch:

    def __init__(self, state_shape, action_size, action_type, num_agents=1, force_reward1=False, batch_size=128,\
                 duplicate_augmented_times=1, set_modify_batch=None, pytorch_device = None):
        self.state_shape     = state_shape
        self.action_size     = action_size
        self.num_agents      = num_agents
        self.action_type     = action_type
        self.batch_size      = batch_size
        self.duplicate_augmented_times = duplicate_augmented_times
        self.pytorch_device  = pytorch_device
        self.force_reward1   = force_reward1
        self.single_reward   = force_reward1 or num_agents==1
        self.create_buffers(rows=batch_size*duplicate_augmented_times,buffers=self)
        if set_modify_batch is not None:
            self.modify_batch = types.MethodType(set_modify_batch,self)
        
    def create_buffers(self, rows, buffers):
        state_shape = (self.state_shape,) if isinstance(self.state_shape,int) else self.state_shape
        num_agents  = (self.num_agents,) if self.num_agents > 1 else ()
        action_size = (self.action_size,) if isinstance(self.action_size,int) else self.action_size
        reward_size = (rows,1) if self.single_reward else (rows,self.num_agents)
        buffers.states      = np.empty((rows,)+num_agents+state_shape, dtype=np.float32)
        buffers.actions     = np.empty((rows,)+num_agents+action_size, dtype=self.action_type)
        buffers.rewards     = np.empty(reward_size,                    dtype=np.float32)
        buffers.next_states = np.empty((rows,)+num_agents+state_shape, dtype=np.float32)
        buffers.dones       = np.empty(reward_size,                    dtype=np.float32)

    def return_batch(self):
        for ind_dup in range(1,self.duplicate_augmented_times):
            ind_start, ind_end = ind_dup*self.batch_size, (ind_dup+1)*self.batch_size
            self.states[ind_start:ind_end,:]      = self.states[0:self.batch_size,:]
            self.actions[ind_start:ind_end,:]     = self.actions[0:self.batch_size,:]
            self.rewards[ind_start:ind_end,:]     = self.rewards[0:self.batch_size,:]
            self.next_states[ind_start:ind_end,:] = self.next_states[0:self.batch_size,:]
            self.dones[ind_start:ind_end,:]       = self.dones[0:self.batch_size,:]
        
        self.modify_batch()
        self.assert_NaN_batch()
        if self.pytorch_device is None:
            # not using pytorch
            return (self.states, self.actions, self.rewards, self.next_states, self.dones)
        
        # using pytorch
        import torch # import for this scope only
        states      = torch.from_numpy(self.states).float().to(self.pytorch_device)
        if self.action_type == np.float32:
            actions = torch.from_numpy(self.actions).float().to(self.pytorch_device)
        else:
            actions = torch.from_numpy(self.actions).long().to(self.pytorch_device)
        rewards     = torch.from_numpy(self.rewards).float().to(self.pytorch_device)
        next_states = torch.from_numpy(self.next_states).float().to(self.pytorch_device)
        dones       = torch.from_numpy(self.dones).float().to(self.pytorch_device)
        return (states, actions, rewards, next_states, dones)
    
    def modify_batch(self):
        pass
        
    def set_NaN_batch(self):
        self.states.fill(np.nan)
        if self.action_type == np.float32:
            self.actions.fill(np.nan)
        self.rewards.fill(np.nan)
        self.next_states.fill(np.nan)
        self.dones.fill(np.nan)

    def assert_NaN_batch(self):
        assert not np.any(np.isnan(self.states))
        if self.action_type == np.float32:
            assert not np.any(np.isnan(self.actions))
        assert not np.any(np.isnan(self.rewards))
        assert not np.any(np.isnan(self.next_states))
        assert not np.any(np.isnan(self.dones))

    def npRewardSize(self, fillValue):
        return fillValue if self.single_reward else np.full(shape=(self.num_agents,), fill_value=fillValue, dtype=np.float32)

    def printSizes(self, name=None):
        if name is not None:
            print(name)
        print("state_shape=              ",self.state_shape)
        print("action_size=              ",self.action_size)
        print("num_agents=               ",self.num_agents)
        print("action_type=              ",self.action_type)
        print("batch_size=               ",self.batch_size)
        print("duplicate_augmented_batch=",self.duplicate_augmented_batch)
        print("force_reward1=            ",self.force_reward1)
        print("single_reward=            ",self.single_reward)
        printBuffersShapes(self)
        
    def printBuffersShapes(buffers):
        print("states.shape=     ",buffers.states.shape)
        print("actions.shape=    ",buffers.actions.shape)
        print("rewards.shape=    ",buffers.rewards.shape)
        print("next_states.shape=",buffers.next_states.shape)
        print("dones.shape=      ",buffers.dones.shape)

####################################################################################        
class ReplayBufferFull:
    def __init__(self, replayBufferBatch, buffer_size=1e5, no_reward_value = None, seed=0):
        self.buffer_size     = int(buffer_size)
        self.no_reward_value = no_reward_value
        self.seed            = seed
        self.rand            = np.random.default_rng(seed)
        replayBufferBatch.create_buffers(self.buffer_size,self)
        self.replayBufferBatch = replayBufferBatch
        self.reset()
        
    def reset(self):
        self.current_len = 0
        self.total_score = self.replayBufferBatch.npRewardSize(0.f)

    def __iadd__(self, other):
        for i in range(min(other.current_len,other.buffer_size)):
            self.add(other.states[i,:],other.actions[i,:],other.rewards[i,:],other.next_states[i,:],\
                     other.dones[i,:])
        return self # must return this object or else it will be destoryed

    def __len__(self):
        return self.current_len

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        if self.replayBufferBatch.single_reward:
            reward = np.sum(reward)
            done = np.any(done)
            self.total_score += reward
            if self.no_reward_value is not None and abs(reward) < 1e-8:
                reward = self.no_reward_value
        else:
            self.total_score += reward
            if self.no_reward_value is not None:
                reward[np.abs(reward) < 1e-8] = self.no_reward_value
        ind_pos                     = self.current_len % self.buffer_size
        self.current_len           += 1
        self.states[ind_pos,:]      = state
        self.actions[ind_pos,:]     = action
        self.rewards[ind_pos,:]     = reward
        self.next_states[ind_pos,:] = next_state
        self.dones[ind_pos,:]       = done
            
    def sample(self, ind_target, batch_size, replace=True):
        """Randomly sample a batch of experiences from memory."""
        if batch_size < 1:
            return True
        minimal_len = 1 if replace else batch_size
        current_len = min(self.current_len, self.buffer_size)
        if current_len < minimal_len:
            return False
        ind_stop = ind_target + batch_size
        indexes  =  self.rand.choice(range(current_len), size=batch_size,replace=replace)
        self.replayBufferBatch.states[ind_target:ind_stop,:]      = self.states[indexes,:]
        self.replayBufferBatch.actions[ind_target:ind_stop,:]     = self.actions[indexes,:]
        self.replayBufferBatch.rewards[ind_target:ind_stop,:]     = self.rewards[indexes,:]
        self.replayBufferBatch.next_states[ind_target:ind_stop,:] = self.next_states[indexes,:]
        self.replayBufferBatch.dones[ind_target:ind_stop,:]       = self.dones[indexes,:]
        return True
        
    def printSizes(self):
        print("buffer_size=    ",self.buffer_size)
        print("no_reward_value=",self.no_reward_value)
        print("seed=           ",self.seed)
        print("current_len=    ",self.current_len)
        print("total_score=    ",self.total_score)
        printBuffersShapes(self)

####################################################################################            
class SimpleReplayBuffer(ReplayBufferBatch):
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, state_shape, action_size, action_type, num_agents=1,\
                 buffer_size=int(1e5), batch_size=128, duplicate_augmented_batch=1, seed=0,\
                 no_reward_value    = 0.0, set_modify_batch=None, force_reward1=False, pytorch_device     = None):
        """Initialize a ReplayBuffer object.

        Params
        ======
            state_shape (tuple:int): dimension of state vector
            action_size (int): dimension of action vector ( 1 for simple DQN)
            action_type (int): np.float32 for DDPG   np.int8 or np.int32 for DQN
            buffer_size (int): maximum size of buffer
            batch_size  (int): size of each training batch
            seed (int): random seed
            pytorch_device: torch.device("cuda:0" if use_cuda else "cpu")
        """
        ReplayBufferBatch.__init__(self, state_shape=state_shape, action_size=action_size,\
                                  action_type=action_type, num_agents=num_agents, batch_size=batch_size, force_reward1=force_reward1,\
                                  duplicate_augmented_batch=duplicate_augmented_batch, set_modify_batch=set_modify_batch,\
                                  pytorch_device=pytorch_device)
        self.replayBufferFull  = ReplayBufferFull(replayBufferBatch=self,  buffer_size=buffer_size, seed=seed,\
                                                  no_reward_value=no_reward_value)
        self.reset()

    
    def reset(self):
        self.replayBufferFull.reset()

    def total_score(self):
        return self.replayBufferFull.total_score
    
    def clone(self):
        return SimpleReplayBuffer(state_shape=self.state_shape, action_size=self.action_size,\
                                  num_agents=self.num_agents, action_type=self.action_type,\
                                  buffer_size=self.replayBufferFull.buffer_size,\
                                  batch_size=self.batch_size,\
                                  duplicate_augmented_batch=self.duplicate_augmented_batch,\
                                  seed=self.replayBufferFull.seed,\
                                  no_reward_value=self.replayBufferFull.no_reward_value,\
                                  set_modify_batch=self.modify_batch, force_reward1=self.force_reward1,\
                                  pytorch_device=self.pytorch_device)
            
    def sample(self, replace=True):
        if not self.replayBufferFull.sample(ind_target=0, batch_size=self.batch_size, replace=replace):
            return None
        return self.return_batch()

    def add(self, state, action, reward, next_state, done):
        self.replayBufferFull.add(state, action, reward, next_state, done)

    def __len__(self):
        return len(self.replayBufferFull)
    
    def __iadd__(self, other):
        self.replayBufferFull += other.replayBufferFull
        return self # must return this object or else it will be destoryed

        
####################################################################################        
class ZeroPosNegReplayBufferInterface:
    def __init__(self, replayBufferBatch, buffer_size=int(1e5), seed=0, no_reward_value    = None):
        """Initialize a ReplayBuffer object.

        Params
        ======
            state_shape (tuple:int): dimension of state vector
            action_size (int): dimension of action vector ( 1 for simple DQN)
            action_type (int): np.float32 for DDPG   np.int8 or np.int32 for DQN
            buffer_size (int): maximum size of buffer
            batch_size  (int): size of each training batch
            seed (int): random seed
            pytorch_device: torch.device("cuda:0" if use_cuda else "cpu")
        """
        self.save_seed            = seed
        self.save_buffer_size     = buffer_size
        self.save_no_reward_value = no_reward_value
        self.replayBufferBatch    = replayBufferBatch
        self.positives   = ReplayBufferFull(replayBufferBatch=replayBufferBatch,  buffer_size=buffer_size, seed=seed,\
                                            no_reward_value=no_reward_value)
        self.negatives   = ReplayBufferFull(replayBufferBatch=replayBufferBatch,  buffer_size=buffer_size, seed=seed,\
                                            no_reward_value=no_reward_value)
        self.zeros       = ReplayBufferFull(replayBufferBatch=replayBufferBatch,  buffer_size=buffer_size, seed=seed,\
                                            no_reward_value=no_reward_value)
        self.reset()

    
    
    def clone(self):
        return ZeroPosNegReplayBufferInterface(replayBufferBatch=self.replayBufferBatch, buffer_size=self.save_buffer_size,\
                                               seed=self.save_seed, no_reward_value=self.save_no_reward_value)

    def reset(self):
        self.positives.reset()
        self.negatives.reset()
        self.zeros.reset()

    def total_score(self):
        return self.positives.total_score + self.negatives.total_score + self.zeros.total_score
        
    def __iadd__(self, other):
        self.positives += other.positives
        self.negatives += other.negatives
        self.zeros     += other.zeros
        return self # must return this object or else it will be destoryed

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        if not np.any(done) and np.all(np.abs(reward) < 1e-8):
            self.zeros.add(state,action,reward,next_state,done)
        elif np.any(reward > 0):
            self.positives.add(state,action,reward,next_state,done)
        else:
            self.negatives.add(state,action,reward,next_state,done)

    def statistics(self):
        return "Total length={} Positives={} Negatives={} Zeros={}".format(len(self), len(self.positives),\
                                                                    len(self.negatives), len(self.zeros))

    def sample(self, ind_target, batch_size, replace=True):
        """Randomly sample a batch of experiences from memory."""
        if batch_size < 1:
            return True
        has_positives = self.positives.current_len > 0
        has_negatives = self.negatives.current_len > 0
        has_zeros     = self.zeros.current_len > 0
        if not has_positives and not has_negatives:
            return self.zeros.sample(ind_target=ind_target, batch_size = batch_size, replace=replace)
        z_batch_size = max(1,batch_size//5) if has_zeros else 0
        batch_size_no_zeros = batch_size - z_batch_size
        if not has_negatives:
            n_batch_size = 0
        else:
            n_batch_size = max(1,batch_size_no_zeros//2) if has_positives else batch_size_no_zeros
        p_batch_size = batch_size_no_zeros - n_batch_size
        if not self.zeros.sample(ind_target=ind_target, batch_size = z_batch_size, replace=replace):
            return False
        if not self.negatives.sample(ind_target=ind_target+z_batch_size, batch_size = n_batch_size, replace=replace):
            return False
        if not self.positives.sample(ind_target=ind_target+z_batch_size+n_batch_size, batch_size = p_batch_size, replace=replace):
            return False
        return True

    def __len__(self):
        return len(self.positives) + len(self.negatives) + len(self.zeros)
    
    def len_non_zero(self):
        return len(self.positives) + len(self.negatives)

    def printSizes(self, useMyBatchBuffer=True, name=""):
        print("save_seed=",self.save_seed)
        print("save_buffer_size=",self.save_buffer_size)
        print("save_no_reward_value=",self.save_no_reward_value)
        print("total_score=",self.total_score())
        self.positives.printSizes(name + ".positives")
        self.negatives.printSizes(name + ".positives")
        self.zeros.printSizes(name + ".zeros")

####################################################################################        
class ZeroPosNegReplayBuffer(ReplayBufferBatch):
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, state_shape, action_size, action_type, num_agents=1, buffer_size=int(1e5),\
                 batch_size=128, duplicate_augmented_times=1, seed=0, force_reward1=False,\
                 no_reward_value    = None, set_modify_batch=None, pytorch_device     = None):
        """Initialize a ReplayBuffer object.

        Params
        ======
            state_shape (tuple:int): dimension of state vector
            action_size (int): dimension of action vector ( 1 for simple DQN)
            action_type (int): np.float32 for DDPG   np.int8 or np.int32 for DQN
            buffer_size (int): maximum size of buffer
            batch_size  (int): size of each training batch
            seed (int): random seed
            pytorch_device: torch.device("cuda:0" if use_cuda else "cpu")
        """
        ReplayBufferBatch.__init__(self, state_shape=state_shape, action_size=action_size,\
                                  action_type=action_type, num_agents=num_agents, batch_size=batch_size, force_reward1=force_reward1,\
                                  duplicate_augmented_times=duplicate_augmented_times, set_modify_batch=set_modify_batch,\
                                  pytorch_device=pytorch_device)
        self.zeroPosNegBuffer = ZeroPosNegReplayBufferInterface(replayBufferBatch=self,  buffer_size=buffer_size, seed=seed,\
                                            no_reward_value=no_reward_value)
        self.reset()

    
    
    def clone(self):
        return ZeroPosNegReplayBuffer(state_shape=self.state_shape, action_size=self.action_size, num_agents=self.num_agents,\
                                      action_type=self.action_type, buffer_size=self.save_buffer_size, batch_size=self.batch_size,\
                                      duplicate_augmented_times=self.duplicate_augmented_times, seed=self.save_seed,\
                                      force_reward1=self.force_reward1,\
                                      no_reward_value=self.save_no_reward_value, set_modify_batch=self.modify_batch,\
                                      pytorch_device=self.pytorch_device)

    def reset(self):
        self.zeroPosNegBuffer.reset()

    def total_score(self):
        return self.zeroPosNegBuffer.total_score()
        
    def __iadd__(self, other):
        self.zeroPosNegBuffer += other.zeroPosNegBuffer
        return self # must return this object or else it will be destoryed

    def add(self, state, action, reward, next_state, done):
        self.zeroPosNegBuffer.add(state,action,reward,next_state,done)

    def statistics(self):
        return self.zeroPosNegBuffer.statistics()

    def sample(self, replace=True):
        if not self.zeroPosNegBuffer.sample(ind_target=0, batch_size=self.batch_size, replace=replace):
            return None
        return replayBufferBase.return_batch()

    def __len__(self):
        return len(self.zeroPosNegBuffer)
    
    def len_non_zero(self):
        return self.zeroPosNegBuffer.len_non_zero()

    def printSizes(self, name=""):
        ReplayBufferBatach.printSizes(name + " (ZeroPosNegReplayBuffer) : ")
        self.zeroPosNegBuffer.printSizes()
        
    
####################################################################################        
class EpisodesReplayBuffer(ReplayBufferBatch):
    def __init__(self, state_shape, action_size, action_type, num_agents=1, buffer_size=int(1e5),\
                 batch_size=128, duplicate_augmented_times=1, seed=0, force_reward1=False,\
                 no_reward_value    = None, set_modify_batch=None, set_modify_episode=None, pytorch_device     = None):

        ReplayBufferBatch.__init__(self, state_shape=state_shape, action_size=action_size,\
                                   action_type=action_type, num_agents=num_agents, batch_size=batch_size,\
                                   force_reward1=force_reward1, duplicate_augmented_times=duplicate_augmented_times,\
                                   set_modify_batch=set_modify_batch, pytorch_device=pytorch_device)

        self.long_term_mem     = ZeroPosNegReplayBufferInterface(replayBufferBatch=self,  buffer_size=buffer_size, seed=seed,\
                                            no_reward_value=no_reward_value)
        self.curr_episode_mem  = self.long_term_mem.clone()
        self.best_episode_mem  = self.long_term_mem.clone()

        if set_modify_episode is not None:
            self.modify_episode = types.MethodType(set_modify_episode,self)
        self.reset()

    def reset(self):
        self.highest_score   = self.npRewardSize(-np.inf)
        self.long_term_mem.reset()
        self.curr_episode_mem.reset()
        self.best_episode_mem.reset()

    def modify_episode(self):
        pass

    def add(self, state, action, reward, next_state, done):
        self.curr_episode_mem.add(state, action, reward, next_state, done)
        if not np.any(done):
            return
        curr_episode_score = self.curr_episode_mem.total_score()
        #print("Adding completed episode with length: {} and score: {}".format(len(self.curr_episode_mem), curr_episode_score))
        # here I can refine the rewards in current episode before adding it to long term memory
        self.modify_episode()
        # end of rewards refining algorithm
        score_diff = np.max(curr_episode_score - self.highest_score)
        if score_diff < -1e-6:
            # most usual case: no new high score
            self.long_term_mem     += self.curr_episode_mem
        else:
            if score_diff > 1e-6:
                # occasional case: new high score
                self.highest_score  = curr_episode_score
                self.long_term_mem += self.best_episode_mem
                self.best_episode_mem.reset()
            else:
                # rare case: same high score
                self.highest_score   = np.maximum(self.highest_score,curr_episode_score)
            self.best_episode_mem   += self.curr_episode_mem
        self.curr_episode_mem.reset()
        

    def sample(self, replace=True):
        if self.best_episode_mem.len_non_zero() < 1:
            return None
        self.set_NaN_batch()
        best_size = max(1,self.batch_size//2) if self.long_term_mem.len_non_zero() > 0 else self.batch_size
        if not self.best_episode_mem.sample(ind_target=0, batch_size=best_size, replace=replace):
            return None
        long_size = self.batch_size - best_size
        if not self.long_term_mem.sample(ind_target=best_size, batch_size = long_size, replace=replace):
            return None
        return self.return_batch()

    def statistics(self):
        return "best len={} long_term len={} highest={:5.2f}".format(len(self.best_episode_mem), len(self.long_term_mem),\
                                                                     self.highest_score)

    def __len__(self):
        return len(self.long_term_mem) + len(self.curr_episode_mem) + len(self.best_episode_mem)

    def printSizes(self, name =""):
        ReplayBufferBatch.printSizes(name + " (EpisodesReplayBuffer) : ")
        self.long_term_mem.printSizes(name=name + ".long_term_mem")
        print("Best episode memory:")
        self.best_episode_mem.printSizes(name=name + ".best_episode_mem")
        print("Current episode memory:")
        self.curr_episode_mem.printSizes(name=name + ".curr_episode_mem")
        