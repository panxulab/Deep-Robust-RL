"""
We adapt the code from https://github.com/denisyarats/pytorch_sac
"""
import collections
import io
import math
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch import distributions as pyd,nn

class TanhTransform(pyd.transforms.Transform):
	domain=pyd.constraints.real
	codomain=pyd.constraints.interval(-1.0,1.0)
	bijective=True
	sign=+1

	def __init__(self,cache_size=1):
		super().__init__(cache_size=cache_size)

	@staticmethod
	def atanh(x):
		return 0.5*(x.log1p()-(-x).log1p())

	def __eq__(self,other):
		return isinstance(other,TanhTransform)

	def _call(self,x):
		return x.tanh()

	def _inverse(self,y):
		# We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
		# one should use `cache_size=1` instead
		return self.atanh(y)

	def log_abs_det_jacobian(self,x,y):
		# We use a formula that is more numerically stable, see details in the following link
		# https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
		return 2.*(math.log(2.)-x-F.softplus(-2.*x))

class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
	def __init__(self,loc,scale):
		self.loc=loc
		self.scale=scale

		self.base_dist=pyd.Normal(loc,scale)
		transforms=[TanhTransform()]
		super().__init__(self.base_dist,transforms)

	@property
	def mean(self):
		mu=self.loc
		for tr in self.transforms:
			mu=tr(mu)
		return mu

def weight_init(t):
	if isinstance(t,nn.Linear):
		nn.init.orthogonal_(t.weight.data)
		if hasattr(t.bias,"data"):
			t.bias.data.fill_(0.0)

class ReplayBuffer(object):
	def __init__(self,state_dim,action_dim,max_size=int(1e6)):
		self.max_size=max_size
		self.at=0
		self.size=0

		self.state=np.zeros((max_size,state_dim))
		self.action=np.zeros((max_size,action_dim))
		self.state_next=np.zeros((max_size,state_dim))
		self.reward=np.zeros((max_size,1))
		self.done=np.zeros((max_size,1))

		self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def add(self,state,action,state_next,reward,done):
		self.state[self.at]=state
		self.action[self.at]=action
		self.state_next[self.at]=state_next
		self.reward[self.at]=reward
		self.done[self.at]=done

		self.at=(self.at+1)%self.max_size
		self.size=min(self.size+1,self.max_size)

	def sample(self,batch_size):
		ind=np.random.randint(0,self.size,size=batch_size)

		Batch=collections.namedtuple("Batch",["state","action","state_next","reward","done"])
		return Batch(
			state=torch.FloatTensor(self.state[ind]).to(self.device),
			action=torch.FloatTensor(self.action[ind]).to(self.device),
			state_next=torch.FloatTensor(self.state_next[ind]).to(self.device),
			reward=torch.FloatTensor(self.reward[ind]).to(self.device),
			done=torch.FloatTensor(self.done[ind]).to(self.device),
		)

def unpack_batch(batch):
	return batch.state,batch.action,batch.state_next,batch.reward,batch.done

class CPU_Unpickler(pickle.Unpickler):
	def find_class(self,module,name):
		if module=="torch.storage" and name=="_load_from_bytes":
			return lambda b:torch.load(io.BytesIO(b),map_location="cpu")
		else:
			return super().find_class(module,name)
