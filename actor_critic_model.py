import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def multiply_int_tuple(x):
    if not isinstance(x,tuple):
        return x
    res = x[0]
    for i in range(1,len(x)):
        res *= x[i]
    return res
    
def ndim_int_tuple(x):
    return len(x) if isinstance(x,tuple) or isinstance(x,list) else 1

class ModuleListUtil(nn.ModuleList):
    def __init__(self, name="", in_features = -1, layers_arch = [], out_features=0):
        super(ModuleListUtil,self).__init__()
        self.name = name
        self.save_in_features  = in_features if isinstance(in_features,tuple) else (in_features,)
        self.build(in_features=in_features, layers_arch=layers_arch, out_features=out_features)
        self.save_out_features = self.work_out_features if isinstance(self.work_out_features,tuple) else (self.work_out_features,)
        self.output_volume     = multiply_int_tuple(self.save_out_features)
        self.ndim_in_features  = len(self.save_in_features)
        self.ndim_out_features = len(self.save_out_features)
        
    def build(self, in_features, layers_arch, out_features=0):
        self.work_out_features = in_features
        self.layers_arch  = layers_arch.copy()
        last_is_linear    = False
        for layer in layers_arch:
            if isinstance(layer, str):
                last_is_linear = False
                if layer == 'b':
                    work_length = multiply_int_tuple(self.work_out_features)
                    self.append(nn.BatchNorm1d(work_length))
                elif layer == 'r':
                    self.append(nn.LeakyReLU())
            elif isinstance(layer,int):
                self.append_linear(out_features=layer)
                last_is_linear = True
        if out_features > 0 and not last_is_linear:
            self.append_linear(out_features=out_features)
        
    def append_linear(self, out_features):
        # reference for nn.init.xavier_uniform_() can be found at: https://pytorch.org/docs/stable/nn.init.html`
        work_length = multiply_int_tuple(self.work_out_features)
        linear = nn.Linear(in_features=work_length, out_features=out_features)
        nn.init.xavier_uniform_(linear.weight)
        self.append(linear)
        self.work_out_features = out_features

    def forward(self, x):
        out_shape = tuple(list(x.shape)[:-self.ndim_in_features]) + self.save_out_features
        x = x.view((-1,)+self.save_in_features)
        for layer in self:
            x = layer(x)
        return x.view(out_shape)
    
class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_shape, action_size, layers_arch = [100,100,50], pytorch_device=None, verbose_level=0):
        """Initialize parameters and build model.
        Params
        ======
            state_shape (tuple:int): Dimension of each state
            action_size (int):       Dimension of each action
        """
        super(Actor, self).__init__()
        self.state_shape    = state_shape
        self.action_size    = action_size
        self.layers_list    = ModuleListUtil(name="Actor",in_features=self.state_shape, layers_arch=layers_arch,\
                                             out_features=self.action_size)
        if verbose_level > 0:
            print("Initializing Actor to:\n",self.layers_list)
        self.pytorch_device = pytorch_device
        if self.pytorch_device is not None:
            self.to(self.pytorch_device)
    
    
    def clone(self):
        """
        Creating new Actor with same initial parameters.
        Used mainly to initialize target network from local network
        """
        return Actor(state_shape=self.state_shape, action_size=self.action_size, layers_arch=self.layers_list.layers_arch.copy(),\
                     pytorch_device=self.pytorch_device)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        return self.layers_list.forward(state)

    def eval_numpy(self, state):
        if state.ndim == 1:
            state = state.reshape((1,len(state)))
        state = torch.from_numpy(state).float()
        if self.pytorch_device:
            state = state.to(self.pytorch_device)
        self.eval() # set model to "eval" mode
        with torch.no_grad():
            action = self(state).cpu().data.numpy()
        self.train() # set model to "train" mode
        return action
    
class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_shape, action_size, num_agents=1, layers_arch = [[100],[50],[50]], pytorch_device=None, verbose_level=0):
        """Initialize parameters and build model.
        Params
        ======
            state_shape (tuple:int): Dimension of each state
            action_size (int):       Dimension of each action
        """
        super(Critic, self).__init__()
        self.state_shape    = state_shape
        self.action_size    = action_size
        self.num_agents     = num_agents
        self.pytorch_device = pytorch_device
        self.state_layers   = ModuleListUtil(name="Critic states",in_features=self.state_shape,  layers_arch=layers_arch[0])
        self.action_layers  = ModuleListUtil(name="Critic actions",in_features=self.action_size, layers_arch=layers_arch[1])
        combine_in_features = self.num_agents*(self.state_layers.output_volume + self.action_layers.output_volume)
        self.combine_layers = ModuleListUtil(name="Critic combined",in_features=combine_in_features, layers_arch=layers_arch[2],\
                                             out_features=num_agents)
        self.ndim_actions_output = self.action_layers.ndim_out_features
        if verbose_level > 0:
            print("Initializing Critic state layers to:\n",self.state_layers)
            print("Initializing Critic action layers to:\n",self.action_layers)
            print("Initializing Critic combine layers to:\n",self.combine_layers)
        assert self.state_layers.ndim_out_features == self.action_layers.ndim_out_features
        if pytorch_device is not None:
            self.to(pytorch_device)
        return

    def clone(self):
        return Critic(state_shape=self.state_shape, action_size=self.action_size, num_agents=self.num_agents,\
                      layers_arch=[self.state_layers.layers_arch.copy(), self.action_layers.layers_arch.copy(),\
                                   self.combine_layers.layers_arch.copy()],\
                      pytorch_device=self.pytorch_device)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = self.state_layers.forward(state)

        xa = self.action_layers.forward(action)

        x  = torch.cat((xs, xa), dim=-self.ndim_actions_output)
        # convert (batch_size,num_agents,partial combine_input_size) to (batch_size,full combine_input_size)
        x  = x.view((-1,)+self.combine_layers.save_in_features)
        return self.combine_layers.forward(x)
