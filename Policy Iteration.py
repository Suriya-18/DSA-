import numpy as np
import matplotlib.pyplot as plt

p_heads = .4
GAMMA = 1
rewards = np.zeros(101) #rewards for each state including 100
rewards[100] = 1

class policy_iteration:
	def __init__(self):
		self.val_state = np.zeros(101)
		self.policy = np.zeros(100)

	def bellman(self, state, action, val_state):
		return p_heads * (rewards[state + action] + GAMMA * val_state[state + action]) + (1 - p_heads) * (rewards[state - action] + GAMMA * val_state[state - action])

	def policy_evaluation(self, epsilon = 0.00000000000001):
		while True:
			delta = 0
			#print('STARTING POLICY EVALUATION')
			for state in range(1,100):
				v = self.val_state[state]
				# print('state', state)
				# print('policy at state', self.policy[state])
				# print('val state', self.val_state)
				self.val_state[state] = self.bellman(state, int(self.policy[state]), self.val_state)
				#val_state[state] is weighted sum over all possible transitions for the policy from this state
				delta = max(delta, np.abs(self.val_state[state] - v))
			if delta < epsilon:
				break
		return self.policy_improvement()

	def policy_improvement(self):
		policy_stable = True
		#print('STARTING POLICY IMPROVEMENT')
		for state in range(1,100):
			old_action = self.policy[state]
			max_action = min(state, 100-state)
			val_action = np.zeros(max_action+1)
			for action in range(1, max_action+1):
				val_action[action] = self.bellman(state, action, self.val_state)
			#x = np.argwhere(val_action == np.max(val_action))
			#self.policy[state] = x[-1]
			self.policy[state] = np.argmax(val_action)
			# print('state', state)
			# print(np.argwhere(val_action >= np.max(val_action)).squeeze())
			# print(self.policy[state])
			# #print(np.argwhere(val_action >= np.max(val_action)*0.99).squeeze())
			# print(np.random.choice(np.argwhere(val_action == np.max(val_action)).squeeze()))
			#self.policy[state] = np.random.choice(np.argwhere(val_action == np.max(val_action)).squeeze())
			
			if old_action != self.policy[state]:
				policy_stable = False
				print('Not stable state', state)
		print('Stable situation', policy_stable)
		if policy_stable != True:
				self.policy_evaluation()
		return self.val_state, self.policy
vi = value_iteration()
v, p = vi.val_iteration()
print(v)
print(p)

# pi = policy_iteration()
# v, p = pi.policy_evaluation()
# print(v)
# print(p)

plt.plot(range(100), v[:100])
plt.xlabel('Capital')
plt.ylabel('Value estimates')
plt.show()

plt.bar(range(100), p, align = 'center', alpha = 0.5)
for i in range(100):
	plt.text(i - 0.75, p[i] + 0.01, str(round(v[i],2)), fontsize=6)
plt.xlabel('Capital')
plt.xticks(np.arange(0, 101, 5))
plt.ylabel('Final policy (stake)')
plt.show()
