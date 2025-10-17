import torch
import warnings
from torch.serialization import SourceChangeWarning

warnings.filterwarnings("ignore", category=SourceChangeWarning)

class PlumeNet:
    def __init__(self):
        model_path = "plume_20210601_VRNN_constantx5b5noisy3x5b5_stepoob_bx0.30.8_t10000004000000_q2.00.5_dmx0.80.8_dmn0.70.4_h64_wd0.0001_n4_codeVRNN_seed2760377.pt"
        self.device = torch.device('cpu')
        self.actor_critic, self.ob_rms = torch.load(model_path, map_location=self.device, weights_only=False)
        self.actor_critic.eval().to(self.device)
        self.reset()

    def reset(self):
        if self.actor_critic.is_recurrent:
            self.rnn_hxs = torch.zeros(1, self.actor_critic.recurrent_hidden_state_size).to(self.device)
        else:
            self.rnn_hxs = torch.zeros(1, 1).to(self.device)
        self.masks = torch.ones(1, 1).to(self.device)

    def act(self, obs, deterministic=True):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value, action, _, self.rnn_hxs, _ = self.actor_critic.act(obs, self.rnn_hxs, self.masks, deterministic)
        return action.cpu().numpy().squeeze()
    
if __name__ == "__main__":
    net = PlumeNet()
    obs = [0, 0, 0.7]
    action = net.act(obs)
    print("action:", action)