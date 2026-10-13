class Batch:
    def __init__(self, states_normalized, next_states_normalized, actions, rewards, values, terminations, log_probs, advantages, returns):
        self.states_normalized = states_normalized
        self.next_states_normalized = next_states_normalized
        self.actions = actions
        self.rewards = rewards
        self.values = values
        self.terminations = terminations
        self.log_probs = log_probs
        self.advantages = advantages
        self.returns = returns
