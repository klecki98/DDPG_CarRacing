import torch as T
import torch.nn.functional as F
from neural_networks import CriticNetwork, ActorNetwork
from action_noise import OUActionNoise
from replay_buffer import ReplayBuffer

class Agent():
    def __init__(self, alpha, beta, input_dims, tau, n_actions,
                 gamma=0.99, max_buffer = int(1e6), hidden_dims = [400,300],
                 batch_size = 64):
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.gamma = gamma

        self.replay_buffer = ReplayBuffer(max_buffer, input_dims, n_actions)

        self.noise = OUActionNoise(n_actions)

        self.actor = ActorNetwork(alpha, input_dims, hidden_dims, n_actions, "actor")
        self.critic = CriticNetwork(beta, input_dims, hidden_dims, n_actions, "critic")
        self.target_actor = ActorNetwork(alpha, input_dims, hidden_dims, n_actions, "target_actor")
        self.target_critic = CriticNetwork(beta, input_dims, hidden_dims, n_actions, "target_critic")

        self.batch_size = batch_size

        self.update_network(tau=1)

    def choose_action(self, observation):
        self.actor.eval()
        observation_tensor = T.tensor([observation], dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(observation_tensor).to(self.actor.device)
        mu_noisy = mu + T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
        self.actor.train()
        return mu_noisy.cpu().detach().numpy()[0]

    def save_transition(self, observation, action, reward, observation_, done):
        self.replay_buffer.store(observation,action,reward,observation_,done)

    def save_networks(self):
        self.actor.save()
        self.critic.save()
        self.target_actor.save()
        self.target_critic.save()

    def load_networks(self):
        self.actor.load()
        self.critic.load()
        self.target_actor.load()
        self.target_critic.load()

    def learn(self):
        if self.replay_buffer.counter >= self.batch_size:
            observations, actions, rewards,\
                new_observations, done_flags = self.replay_buffer.sample_buffer(self.batch_size)

            observations = T.tensor(observations, dtype=T.float).to(self.actor.device)
            actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
            rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
            new_observations = T.tensor(new_observations, dtype=T.float).to(self.actor.device)
            done_flags = T.tensor(done_flags).to(self.actor.device)

            target_actions = self.target_actor.forward(new_observations)
            target_critic_output = self.target_critic.forward(new_observations, target_actions)
            target_critic_output[done_flags] = 0.
            target_critic_output = target_critic_output.view(-1)

            target = rewards + self.gamma*target_critic_output
            target = target.view(self.batch_size, 1)

            critic_output = self.critic.forward(observations,actions)

            self.critic.optimizer.zero_grad()
            critic_loss = F.mse_loss(target, critic_output)
            critic_loss.backward()
            self.critic.optimizer.step()

            self.actor.optimizer.zero_grad()
            actor_loss = T.mean(-self.critic.forward(observations, self.actor.forward(observations)))
            actor_loss.backward()
            self.actor.optimizer.step()

            self.update_network()

    def update_network(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = dict(self.actor.named_parameters())
        critic_params = dict(self.critic.named_parameters())
        target_actor_params = dict(self.target_actor.named_parameters())
        target_critic_params = dict(self.target_critic.named_parameters())

        for key in critic_params:
            critic_params[key] = tau*critic_params[key].clone()+(1-tau)*target_critic_params[key].clone()

        for key in actor_params:
            actor_params[key] = tau * actor_params[key].clone() + (1 - tau) * target_actor_params[key].clone()

        self.target_critic.load_state_dict(critic_params)
        self.target_actor.load_state_dict(actor_params)
