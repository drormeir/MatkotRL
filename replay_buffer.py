import numpy as np
import types
import numbers
# some users might not install torch at the first place
#import torch # optional import

class ReplayBufferBase:

    def __init__(self, state_shape, action_size, action_type, num_agents=1, force_reward1=False, batch_size=128,\
                 duplicate_batch=1, set_modify_batch=None, pytorch_device = None):
        self.state_shape     = state_shape
        self.action_size     = action_size
        self.num_agents      = num_agents
        self.action_type     = action_type
        self.batch_size      = batch_size
        self.duplicate_batch = duplicate_batch
        self.pytorch_device  = pytorch_device
        self.force_reward1   = force_reward1
        self.create_buffers(rows=batch_size*duplicate_batch,buffers=self)
        if set_modify_batch is not None:
            self.modify_batch = types.MethodType(set_modify_batch,self)

        
    def create_buffers(self, rows, buffers):
        state_shape = (self.state_shape,) if isinstance(self.state_shape,int) else self.state_shape
        num_agents  = (self.num_agents,) if self.num_agents > 1 else ()
        action_size = (self.action_size,) if isinstance(self.action_size,int) else self.action_size
        reward_size = (rows,1) if self.force_reward1 else (rows,self.num_agents)
        buffers.states      = np.empty((rows,)+num_agents+state_shape, dtype=np.float32)
        buffers.actions     = np.empty((rows,)+num_agents+action_size, dtype=self.action_type)
        buffers.rewards     = np.empty(reward_size,                    dtype=np.float32)
        buffers.next_states = np.empty((rows,)+num_agents+state_shape, dtype=np.float32)
        buffers.dones       = np.empty(reward_size,                    dtype=np.float32)

    def return_batch(self):
        for ind_dup in range(1,self.duplicate_batch):
            ind_start, ind_end = ind_dup*self.batch_size, (ind_dup+1)*self.batch_size
            self.states[ind_start:ind_end,:]      = self.states[0:self.batch_size,:]
            self.actions[ind_start:ind_end,:]     = self.actions[0:self.batch_size,:]
            self.rewards[ind_start:ind_end,:]     = self.rewards[0:self.batch_size,:]
            self.next_states[ind_start:ind_end,:] = self.next_states[0:self.batch_size,:]
            self.dones[ind_start:ind_end,:]       = self.dones[0:self.batch_size,:]
        
        self.modify_batch()
        
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
        
####################################################################################        
class ReplayBufferFull:
    def __init__(self, replayBufferBase, buffer_size=1e5, no_reward_value = None, seed=0):
        self.buffer_size     = buffer_size
        self.no_reward_value = no_reward_value
        self.seed            = seed
        self.rand            = np.random.default_rng(seed)
        replayBufferBase.create_buffers(buffer_size,self)
        self.replayBufferBase = replayBufferBase
        self.reset()
        
    def reset(self):
        self.current_len     = 0

    def __iadd__(self, other):
        for i in range(min(other.current_len,other.buffer_size)):
            self.add(other.states[i,:],other.actions[i,:],other.rewards[i,:],other.next_states[i,:],\
                     other.dones[i,:])
        return self # must return this object or else it will be destoryed

    def __len__(self):
        return self.current_len

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        if self.no_reward_value is not None:
            if isinstance(reward,numbers.Number):
                if abs(reward) < 1e-8:
                    reward = self.no_reward_value
            else:
                reward[np.abs(reward) < 1e-8] = self.no_reward_value
        ind_pos                     = self.current_len % self.buffer_size
        self.current_len           += 1
        self.states[ind_pos,:]      = state
        self.actions[ind_pos,:]     = action
        self.rewards[ind_pos,:]     = reward
        self.next_states[ind_pos,:] = next_state
        self.dones[ind_pos,:]       = done
            
    def sample(self, ind_target=0, batch_size=None, replace=True):
        """Randomly sample a batch of experiences from memory."""
        if batch_size is None:
            ind_target = 0
            batch_size = self.replayBufferBase.batch_size
        if self.current_len < (1 if replace else batch_size):
            return False
        ind_stop = ind_target + batch_size
        indexes  =  self.rand.choice(range(self.current_len), size=batch_size,replace=replace)
        self.replayBufferBase.states[ind_target:ind_stop,:]      = self.states[indexes,:]
        self.replayBufferBase.actions[ind_target:ind_stop,:]     = self.actions[indexes,:]
        self.replayBufferBase.rewards[ind_target:ind_stop,:]     = self.rewards[indexes,:]
        self.replayBufferBase.next_states[ind_target:ind_stop,:] = self.next_states[indexes,:]
        self.replayBufferBase.dones[ind_target:ind_stop,:]       = self.dones[indexes,:]
        return True
        

class SimpleReplayBuffer(ReplayBufferBase):
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, state_shape, action_size, action_type, num_agents=1,\
                 buffer_size=int(1e5), batch_size=128, duplicate_batch=1, seed=0,\
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
        ReplayBufferBase.__init__(self, state_shape=state_shape, action_size=action_size,\
                                  action_type=action_type, num_agents=num_agents, batch_size=batch_size, force_reward1=force_reward1,\
                                  duplicate_batch=duplicate_batch, set_modify_batch=set_modify_batch, pytorch_device=pytorch_device)
        self.replayBufferFull  = ReplayBufferFull(replayBufferBase=self,  buffer_size=buffer_size, seed=seed,\
                                                  no_reward_value=no_reward_value)
        self.reset()

    
    
    def clone(self):
        return SimpleReplayBuffer(state_shape=self.state_shape, action_size=self.action_size,\
                                  num_agents=self.num_agents, action_type=self.action_type,\
                                  buffer_size=self.replayBufferFull.buffer_size,\
                                  batch_size=self.batch_size,\
                                  duplicate_batch=self.duplicate_batch,\
                                  seed=self.replayBufferFull.seed,\
                                  no_reward_value=self.replayBufferFull.no_reward_value,\
                                  set_modify_batch=self.modify_batch, force_reward1=force_reward1,\
                                  pytorch_device=self.pytorch_device)
            
    def sample(self, replace=True):
        if not self.replayBufferFull.sample(replace=replace):
            return None
        return self.return_batch()

    def add(self, state, action, reward, next_state, done):
        self.replayBufferFull.add(state, action, reward, next_state, done)

    def __len__(self):
        return len(self.replayBufferFull)
    
    def __iadd__(self, other):
        self.replayBufferFull += other.replayBufferFull
        return self # must return this object or else it will be destoryed
        
        
class BalancedReplayBuffer(ReplayBufferBase):
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, state_shape, action_size, action_type, num_agents=1, buffer_size=int(1e5),\
                 batch_size=128, duplicate_batch=1, seed=0, force_reward1=False,\
                 no_reward_value    = 0.0, set_modify_batch=None, pytorch_device     = None):
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
        ReplayBufferBase.__init__(self, state_shape=state_shape, action_size=action_size,\
                                  action_type=action_type, num_agents=num_agents, batch_size=batch_size, force_reward1=force_reward1,\
                                  duplicate_batch=duplicate_batch, set_modify_batch=set_modify_batch, pytorch_device=pytorch_device)
        self.seed = seed
        self.buffer_size = buffer_size
        self.no_reward_value = no_reward_value
        self.positives   = ReplayBufferFull(replayBufferBase=self,  buffer_size=buffer_size, seed=seed,\
                                            no_reward_value=no_reward_value)
        self.negatives   = ReplayBufferFull(replayBufferBase=self,  buffer_size=buffer_size, seed=seed,\
                                            no_reward_value=no_reward_value)
        self.zeros       = ReplayBufferFull(replayBufferBase=self,  buffer_size=buffer_size, seed=seed,\
                                            no_reward_value=no_reward_value)
        self.reset()

    
    
    def clone(self):
        return BalancedReplayBuffer(state_shape=self.state_shape, action_size=self.action_size, num_agents=num_agents,\
                                    action_type=self.action_type, buffer_size=self.buffer_size, batch_size=self.batch_size,\
                                    duplicate_batch=self.duplicate_batch, seed=self.seed, force_reward1=force_reward1,\
                                    no_reward_value=self.no_reward_value, set_modify_batch=self.modify_batch,\
                                    pytorch_device=self.pytorch_device)

    def reset(self):
        self.positives.reset()
        self.negatives.reset()
        self.zeros.reset()
        
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
        return "Total={} Positives={} Negatives={} Zeros={}".format(len(self), len(self.positives),\
                                                                    len(self.negatives), len(self.zeros))
    def sample(self, replace=True):
        """Randomly sample a batch of experiences from memory."""
        pnz = int(self.positives.current_len > 0)*4 + int(self.negatives.current_len > 0)*2 + int(self.zeros.current_len > 0)
        if pnz <= 1:
            # if memory contains only zeros then there is nothing to learn
            return None
        if pnz == 2:
            if not self.negatives.sample(replace=replace):
                return None
        elif pnz == 4:
            if not self.positives.sample(replace=replace):
                return None
        elif pnz == 3:
            z_batch_size = max(1,self.batch_size//2)
            if not self.zeros.sample(ind_target=0, batch_size = z_batch_size, replace=replace):
                return None
            n_batch_size = self.batch_size - z_batch_size
            if not self.negatives.sample(ind_target=z_batch_size, batch_size = n_batch_size, replace=replace):
                return None
        elif pnz == 5:
            z_batch_size = max(1,self.batch_size//2)
            if not self.zeros.sample(ind_target=0, batch_size = z_batch_size, replace=replace):
                return None
            p_batch_size = self.batch_size - z_batch_size
            if not self.positives.sample(ind_target=z_batch_size, batch_size = p_batch_size, replace=replace):
                return None
        elif pnz == 6:
            n_batch_size = max(1,self.batch_size//2)
            if not self.negatives.sample(ind_target=0, batch_size = n_batch_size, replace=replace):
                return None
            p_batch_size = self.batch_size - n_batch_size
            if not self.positives.sample(ind_target=z_batch_size, batch_size = p_batch_size, replace=replace):
                return None
        else:
            z_batch_size = max(1,self.batch_size//5) # ratio can be a parameter
            if not self.zeros.sample(ind_target=0, batch_size = z_batch_size, replace=replace):
                return None
            n_batch_size = max(1,(self.batch_size - z_batch_size)//2)
            if not self.negatives.sample(ind_target=z_batch_size, batch_size = n_batch_size, replace=replace):
                return None
            p_batch_size = self.batch_size - z_batch_size - n_batch_size
            if not self.positives.sample(ind_target=z_batch_size+n_batch_size, batch_size = p_batch_size, replace=replace):
                return None
        return self.return_batch()

    def __len__(self):
        return len(self.positives) + len(self.negatives) + len(self.zeros)
    
    def len_non_zero(self):
        return len(self.positives) + len(self.negatives)
