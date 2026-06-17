import numpy as np
from scipy.linalg import lapack,solve_triangular

class LSVI():

	def __init__(self,env,state_dim,action_dim,feature_dim,action_space,K,H,beta,lamda,rho=None):
		self.env=env
		self.state_dim=state_dim
		self.action_dim=action_dim
		self.feature_dim=feature_dim
		self.action_space=action_space

		self.K=K
		self.H=H
		self.beta=beta  # bonus parameter
		self.lamda=lamda  # ridge regression parameter
		self.rho=rho  # robust parameter

		self.at=0
		self.reward_matrix=np.zeros((K,H))
		self.history_phi=np.zeros((K,H,feature_dim))
		self.history_phi_full=np.zeros((K,H,len(action_space),feature_dim))

		self.Lambda=np.zeros((H,feature_dim,feature_dim))  # Lambda matrix
		self.Lambda_dec=np.zeros((H,feature_dim,feature_dim))  # Cholesky decomposition
		self.Lambda_rob=np.zeros((H,feature_dim))
		self.w=np.zeros((H,feature_dim))

		for h in range(H):  # initialize Lambda matrix
			self.Lambda[h]=lamda*np.eye(feature_dim)
			self.Lambda_dec[h]=np.linalg.cholesky(self.Lambda[h])
			inv=lapack.dtrtri(self.Lambda_dec[h],lower=True)[0]
			self.Lambda_rob[h]=np.linalg.norm(inv,axis=0)

	def insert(self,h,reward,phi,phi_full):
		self.reward_matrix[self.at,h]=reward
		self.history_phi[self.at,h]=phi
		self.history_phi_full[self.at,h]=phi_full
		if h==self.H-1: self.at+=1

	def get_feature(self,state,feature_func):
		return np.array([feature_func(state,a) for a in self.action_space])

	def get_Q_func(self,phi_full,h):
		if self.rho is None:
			Y=solve_triangular(self.Lambda_dec[h],phi_full.T,lower=True,check_finite=False)
			Gamma=self.beta*np.linalg.norm(Y,axis=0)
		else:
			Gamma=self.beta*phi_full@self.Lambda_rob[h]
		Q_h=np.maximum(np.minimum(phi_full@self.w[h]+Gamma,self.H-h),0)
		return Q_h

	def get_action(self,phi_full,h):
		Q_h=self.get_Q_func(phi_full,h)
		action=np.argmax(Q_h)
		return self.action_space[action],phi_full[action]

	def get_action_test(self,phi_full,h):
		Q_h=phi_full@self.w[h]
		action=np.argmax(Q_h)
		return self.action_space[action]

	def calc_matrix(self,h):
		phi_matrix=self.history_phi[:self.at,h]
		r_matrix=self.reward_matrix[:self.at,h]
		V_matrix=np.zeros(self.at)
		if h!=self.H-1:
			for tau in range(self.at):
				phi_full=self.history_phi_full[tau,h+1]
				Q_next=self.get_Q_func(phi_full,h+1)
				V_matrix[tau]=np.max(Q_next)
		return phi_matrix,r_matrix,V_matrix

	def cholupdate(self,L,x,sign):
		for k in range(self.feature_dim):
			if sign:
				r=np.sqrt(L[k,k]**2+x[k]**2)
			else:
				r=np.sqrt(L[k,k]**2-x[k]**2)
			c=r/L[k,k]
			s=x[k]/L[k,k]
			L[k,k]=r
			if sign:
				L[k+1:,k]=(L[k+1:,k]+s*x[k+1:])/c
			else:
				L[k+1:,k]=(L[k+1:,k]-s*x[k+1:])/c
			x[k+1:]=c*x[k+1:]-s*L[k+1:,k]

	def estimate_w(self):
		for h in range(self.H-1,-1,-1):
			phi=self.history_phi[self.at-1,h]
			self.Lambda[h]+=np.outer(phi,phi)
			self.cholupdate(self.Lambda_dec[h],phi.copy(),True)
			if self.rho is not None:
				inv=lapack.dtrtri(self.Lambda_dec[h],lower=True)
				assert not inv[1]
				self.Lambda_rob[h]=np.linalg.norm(inv[0],axis=0)

			phi_matrix,r_matrix,V_matrix=self.calc_matrix(h)
			Y=solve_triangular(self.Lambda_dec[h],phi_matrix.T,lower=True,check_finite=False)
			xi=solve_triangular(self.Lambda_dec[h].T,Y,lower=False,check_finite=False)

			if self.rho is None or h==self.H-1 or not len(V_matrix):
				self.w[h]=xi@(r_matrix+V_matrix)
			else:
				nu=np.zeros(self.feature_dim)
				idxs=np.argsort(V_matrix)
				xi,r_matrix,V_matrix=xi[:,idxs],r_matrix[idxs],V_matrix[idxs]

				for i in range(self.feature_dim):
					xi_i=xi[i]
					res_0,res_1=np.cumsum(xi_i*V_matrix),np.cumsum(xi_i)
					for idx in range(len(idxs)):
						nu[i]=np.maximum(nu[i],res_0[idx]+(res_1[-1]-res_1[idx]-self.rho)*V_matrix[idx])

				self.w[h]=xi@r_matrix+nu

	def explore(self,feature_func):
		state=self.env.reset()
		for h in range(self.H):
			phi_step=self.get_feature(state,feature_func)
			action,phi=self.get_action(phi_step,h)
			next_state,reward,done=self.env.step(action)
			self.insert(h,reward,phi,phi_step)
			if done: break
			state=next_state

	def train(self,feature_func):
		for k in range(self.K):
			self.explore(feature_func)
			self.estimate_w()
