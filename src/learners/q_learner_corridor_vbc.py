import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
import numpy as np
from torch.optim import RMSprop

# learning for 6z_vs_24zerg scenario
class QLearner_corridor:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.params += list(self.mac.env_blender.parameters())
        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
    
        # Calculate estimated Q-Values
        mac_out = []
        difference_out = []
        difference_out1 = []         
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_local_outputs, hidden_states = self.mac.forward(batch, t=t)  
            dummy0 = self.mac.env_blender(hidden_states[:,0,:].view(32,-1)) 
            dummy1 = self.mac.env_blender(hidden_states[:,1,:].view(32,-1)) 
            dummy2 = self.mac.env_blender(hidden_states[:,2,:].view(32,-1)) 
            dummy3 = self.mac.env_blender(hidden_states[:,3,:].view(32,-1)) 
            dummy4 = self.mac.env_blender(hidden_states[:,4,:].view(32,-1))
            dummy5 = self.mac.env_blender(hidden_states[:,5,:].view(32,-1)) 
            
            agent0 = (dummy1 + dummy2 + dummy3 + dummy4 + dummy5)/5.0
            agent1 = (dummy0 + dummy2 + dummy3 + dummy4 + dummy5)/5.0
            agent2 = (dummy0 + dummy1 + dummy3 + dummy4 + dummy5)/5.0
            agent3 = (dummy0 + dummy1 + dummy2 + dummy4 + dummy5)/5.0
            agent4 = (dummy0 + dummy1 + dummy2 + dummy3 + dummy5)/5.0
            agent5 = (dummy0 + dummy1 + dummy2 + dummy3 + dummy4)/5.0
    
            agent_global_outputs =th.cat((agent0.view((32,1,30)),agent1.view((32,1,30)),agent2.view((32,1,30)),agent3.view((32,1,30)),agent4.view((32,1,30)),agent5.view((32,1,30))),1)            
            agent_outs = agent_local_outputs + agent_global_outputs
            difference = agent_global_outputs 
            mac_out.append(agent_outs)
            difference_out.append(difference)
            
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        difference_out = th.stack(difference_out, dim=1)  # Concat over time
        difference_out = th.std(difference_out,dim = 3).sum()
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        avg_difference = (difference_out.sum())/((agent_outs.shape[0]*agent_outs.shape[1]*agent_outs.shape[2]*batch.max_seq_length))
        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_local_outputs, target_hidden_states = self.target_mac.forward(batch, t=t)
    
            dummy0 = self.target_mac.env_blender(target_hidden_states[:,0,:].view(32,-1)) 
            dummy1 = self.target_mac.env_blender(target_hidden_states[:,1,:].view(32,-1)) 
            dummy2 = self.target_mac.env_blender(target_hidden_states[:,2,:].view(32,-1)) 
            dummy3 = self.target_mac.env_blender(target_hidden_states[:,3,:].view(32,-1)) 
            dummy4 = self.target_mac.env_blender(target_hidden_states[:,4,:].view(32,-1))
            dummy5 = self.target_mac.env_blender(target_hidden_states[:,5,:].view(32,-1)) 
            
            target_agent0 = (dummy1 + dummy2 + dummy3 + dummy4 + dummy5)/5.0
            target_agent1 = (dummy0 + dummy2 + dummy3 + dummy4 + dummy5)/5.0
            target_agent2 = (dummy0 + dummy1 + dummy3 + dummy4 + dummy5)/5.0
            target_agent3 = (dummy0 + dummy1 + dummy2 + dummy4 + dummy5)/5.0
            target_agent4 = (dummy0 + dummy1 + dummy2 + dummy3 + dummy5)/5.0
            target_agent5 = (dummy0 + dummy1 + dummy2 + dummy3 + dummy4)/5.0  
            target_agent_global_outputs = th.cat((target_agent0.view((32,1,30)),target_agent1.view((32,1,30)),target_agent2.view((32,1,30)),target_agent3.view((32,1,30)),target_agent4.view((32,1,30)),target_agent5.view((32,1,30))),1)
            target_agent_outs = target_agent_local_outputs + target_agent_global_outputs
            target_mac_out.append(target_agent_outs)
          
        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out[avail_actions == 0] = -9999999
            cur_max_actions = mac_out[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum() + self.args.normalization_const * avg_difference

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))


