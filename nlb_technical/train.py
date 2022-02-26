import os, gc
import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime

from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_tensors import make_train_input_tensors, make_eval_input_tensors
from nlb_tools.evaluation import evaluate

SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

DATAPATH_DICT = {
    'mc_maze': './000128/sub-Jenkins/',
    'mc_rtt': './000129/sub-Indy/',
    'area2_bump': './000127/sub-Han/',
    'dmfc_rsg': './000130/sub-Haydn/',
    # 'mc_maze_large': './000138/sub-Jenkins/',
    'mc_maze_large': '/home/fpei2/lvm/code/dandi/000138/sub-Jenkins/',
    'mc_maze_medium': './000139/sub-Jenkins/',
    'mc_maze_small': './000140/sub-Jenkins/',
}

def get_data(dataset_name, phase='test', bin_size=5):
    """Function that extracts and formats data for training model"""
    dataset = NWBDataset(DATAPATH_DICT[dataset_name], 
        skip_fields=['cursor_pos', 'eye_pos', 'cursor_vel', 'eye_vel', 'hand_pos'])
    dataset.resample(5)
    train_split = ['train', 'val'] if phase == 'test' else 'train'
    eval_split = phase
    train_dict = make_train_input_tensors(dataset, dataset_name, train_split, save_file=False, include_forward_pred=True)
    eval_dict = make_eval_input_tensors(dataset, dataset_name, eval_split, save_file=False)
    training_input = np.concatenate([
        train_dict['train_spikes_heldin'],
        np.zeros(train_dict['train_spikes_heldin_forward'].shape),
    ], axis=1)
    training_output = np.concatenate([
        np.concatenate([
            train_dict['train_spikes_heldin'],
            train_dict['train_spikes_heldin_forward'],
        ], axis=1),
        np.concatenate([
            train_dict['train_spikes_heldout'],
            train_dict['train_spikes_heldout_forward'],
        ], axis=1),
    ], axis=2)
    eval_input = np.concatenate([
        eval_dict['eval_spikes_heldin'],
        np.zeros((
            eval_dict['eval_spikes_heldin'].shape[0],
            train_dict['train_spikes_heldin_forward'].shape[1],
            eval_dict['eval_spikes_heldin'].shape[2]
        )),
    ], axis=1)
    del dataset
    return training_input, training_output, eval_input

class NLBRNN(torch.nn.Module):
    """Simple RNN to model spiking data"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.3, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(NLBRNN, self).__init__()
        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.rnn = torch.nn.GRU(input_size=input_dim,
                                    hidden_size=hidden_dim,
                                    num_layers=num_layers,
                                    batch_first=True,
                                    dropout=(dropout if num_layers > 1 else 0.),
                                    bidirectional=False,
                                    **factory_kwargs)
        self.dropout2 = torch.nn.Dropout(p=dropout)
        self.transform = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, X):
        output, hidden = self.rnn(self.dropout1(X))
        output = self.transform(self.dropout2(output))
        return torch.exp(output)

class NLBRunner:
    """Class that handles training NLBRNN"""
    def __init__(self, model_init, model_cfg, data, train_cfg, use_gpu=False, num_gpus=1):
        self.model = model_init(**model_cfg)
        self.data = data
        if use_gpu and torch.cuda.is_available():
            device = torch.device('cuda:0')
            gpu_idxs = np.arange(min(num_gpus, torch.cuda.device_count())).tolist()
            self.model = torch.nn.DataParallel(self.model.to(device), device_ids=gpu_idxs)
            self.data = tuple([d.to(device) for d in self.data])
        self.cd_ratio = train_cfg.get('cd_ratio', 0.2)
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=train_cfg.get('lr', 1e-3), 
                                          weight_decay=train_cfg.get('alpha', 0.0))
    
    def make_cd_mask(self, train_input, train_output):
        """Creates boolean mask for coordinated dropout.

        In coordinated dropout, a random set of inputs is zeroed out,
        and only the corresponding outputs (i.e. same trial, timestep, and neuron)
        are used to compute loss and update model weights. This prevents
        exact spike times from being directly passed through the model.
        """
        cd_ratio = self.cd_ratio
        input_mask = torch.zeros((train_input.shape[0] * train_input.shape[1] * train_input.shape[2]), dtype=torch.bool)
        idxs = torch.randperm(input_mask.shape[0])[:int(round(cd_ratio * input_mask.shape[0]))]
        input_mask[idxs] = True
        input_mask = input_mask.view((train_input.shape[0], train_input.shape[1], train_input.shape[2]))
        output_mask = torch.ones(train_output.shape, dtype=torch.bool)
        output_mask[:, :, :input_mask.shape[2]] = input_mask
        return input_mask, output_mask
    
    def train_epoch(self):
        """Trains model for one epoch. 
        This simple script does not support splitting training samples into batches.
        """
        self.model.train()
        self.optimizer.zero_grad()
        # create mask for coordinated dropout
        train_input, train_output, val_input, val_output, *_ = self.data
        input_mask, output_mask = self.make_cd_mask(train_input, train_output)
        # mask inputs
        masked_train_input = train_input.clone()
        masked_train_input[input_mask] = 0.0
        train_predictions = self.model(masked_train_input)
        # learn only from masked inputs
        loss = torch.nn.functional.poisson_nll_loss(train_predictions[output_mask], train_output[output_mask], log_input=False)
        loss.backward()
        self.optimizer.step()
        # get validation score
        train_res, train_output = self.score(train_input, train_output, prefix='train')
        val_res, val_output = self.score(val_input, val_output, prefix='val')
        res = train_res.copy()
        res.update(val_res)
        return res, (train_output, val_output)
    
    def score(self, input, output, prefix='val'):
        """Evaluates model performance on given data"""
        self.model.eval()
        predictions = self.model(input)
        self.model.train()
        loss = torch.nn.functional.poisson_nll_loss(predictions, output, log_input=False)
        num_heldout = output.shape[2] - input.shape[2]
        cosmooth_loss = torch.nn.functional.poisson_nll_loss(
            predictions[:, :, -num_heldout:], output[:, :, -num_heldout:], log_input=False)
        return {f'{prefix}_nll': loss.item(), f'{prefix}_cosmooth_nll': cosmooth_loss.item()}, predictions

    def train(self, n_iter=1000, patience=200, save_path=None, verbose=False, log_frequency=50):
        """Trains model for given number of iterations with early stopping"""
        train_log = []
        best_score = 1e8
        last_improv = -1
        for i in range(n_iter):
            res, output = self.train_epoch()
            res['iter'] = i
            train_log.append(res)
            if verbose:
                if (i % log_frequency) == 0:
                    print(res)
            if res['val_nll'] < best_score:
                best_score = res['val_nll']
                last_improv = i
                data = res.copy()
                if save_path is not None:
                    self.save_checkpoint(save_path, data)
            if (i - last_improv) > patience:
                break
        return train_log
    
    def save_checkpoint(self, file_path, data):
        default_ckpt = {
            "state_dict": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
        }
        assert "state_dict" not in data
        assert "optim_state" not in data
        default_ckpt.update(data)
        torch.save(default_ckpt, file_path)
        
# Run parameters
dataset_name = 'mc_maze_large'
phase = 'val'
bin_size = 5

# Extract data
training_input, training_output, eval_input = get_data(dataset_name, phase, bin_size)

# Train/val split and convert to Torch tensors
num_train = int(round(training_input.shape[0] * 0.75))
train_input = torch.Tensor(training_input[:num_train])
train_output = torch.Tensor(training_output[:num_train])
val_input = torch.Tensor(training_input[num_train:])
val_output = torch.Tensor(training_output[num_train:])
eval_input = torch.Tensor(eval_input)

# Model hyperparameters
DROPOUT = 0.46
L2_WEIGHT = 5e-7
LR_INIT = 1.5e-2
CD_RATIO = 0.27
HIDDEN_DIM = 40
USE_GPU = False
MAX_GPUS = 2

RUN_NAME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_model'
RUN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), './runs/')
if not os.path.isdir(RUN_DIR):
    os.mkdir(RUN_DIR)

# Train model
runner = NLBRunner(
    model_init=NLBRNN,
    model_cfg={'input_dim': train_input.shape[2], 'hidden_dim': HIDDEN_DIM, 'output_dim': train_output.shape[2], 'dropout': DROPOUT},
    data=(train_input, train_output, val_input, val_output, eval_input),
    train_cfg={'lr': LR_INIT, 'alpha': L2_WEIGHT, 'cd_ratio': CD_RATIO},
    use_gpu=USE_GPU,
    dataset_name=dataset_name,
    num_gpus=MAX_GPUS,
)

model_dir = os.path.join(RUN_DIR, RUN_NAME)
os.mkdir(os.path.join(RUN_DIR, RUN_NAME))
train_log = runner.train(n_iter=10000, patience=1000, save_path=os.path.join(model_dir, 'model.ckpt'), verbose=True)

# Save results
import pandas as pd
train_log = pd.DataFrame(train_log)
train_log.to_csv(os.path.join(model_dir, 'train_log.csv'))
