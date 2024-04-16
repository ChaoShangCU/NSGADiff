# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import os
from tool.utils import available_devices, format_devices
device = available_devices(threshold=10000, n_devices=1)
os.environ["CUDA_VISIBLE_DEVICES"] = format_devices(device)
import copy
import utils
import argparse
from configs.datasets_config import get_dataset_info
from os.path import join
from qm9 import dataset
from util.utils import EMA
import torch
import time
import pickle
from torch import optim
import logging
import wandb

from tool.utils import set_logger
from energys_fingerprint.training_energy import train_epoch, get_model

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='predict_fingerprint')
parser.add_argument('--model', type=str, default='egnn_dynamics',
                    help='our_dynamics | schnet | simple_dynamics | '
                         'kernel_dynamics | egnn_dynamics |gnn_dynamics')
parser.add_argument('--probabilistic_model', type=str, default='diffusion',
                    help='diffusion')
parser.add_argument('--diffusion_steps', type=int, default=1000)
parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
                    help='learned, cosine')
parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5,
                    )
parser.add_argument('--n_epochs', type=int, default=2000)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-16, metavar='N',
                        help='weight decay')
parser.add_argument('--brute_force', type=eval, default=False,
                    help='True | False')
parser.add_argument('--actnorm', type=eval, default=True,
                    help='True | False')
parser.add_argument('--break_train_epoch', type=eval, default=False,
                    help='True | False')
parser.add_argument('--dp', type=eval, default=True,
                    help='True | False')
parser.add_argument('--condition_time', type=eval, default=True,
                    help='True | False')
parser.add_argument('--clip_grad', type=eval, default=True,
                    help='True | False')
parser.add_argument('--trace', type=str, default='hutch',
                    help='hutch | exact')
# EGNN args -->
parser.add_argument('--n_layers', type=int, default=7,
                    help='number of layers')
parser.add_argument('--inv_sublayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--nf', type=int, default=192,
                    help='number of layers')
parser.add_argument('--tanh', type=eval, default=True,
                    help='use tanh in the coord_mlp')
parser.add_argument('--attention', type=eval, default=True,
                    help='use attention in the EGNN')
parser.add_argument('--norm_constant', type=float, default=1,
                    help='diff/(|diff| + norm_constant)')
parser.add_argument('--sin_embedding', type=eval, default=False,
                    help='whether using or not the sin embedding')
# <-- EGNN args
parser.add_argument('--ode_regularization', type=float, default=1e-3)
parser.add_argument('--dataset', type=str, default='qm9',
                    help='qm9 | qm9_second_half (train only on the last 50K samples of the training dataset)')
parser.add_argument('--datadir', type=str, default='qm9/temp',
                    help='qm9 directory')
parser.add_argument('--filter_n_atoms', type=int, default=None,
                    help='When set to an integer value, QM9 will only contain molecules of that amount of atoms')
parser.add_argument('--dequantization', type=str, default='deterministic',
                    help='uniform | variational | argmax_variational | deterministic')
parser.add_argument('--n_report_steps', type=int, default=1)
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--save_model', type=eval, default=True,
                    help='save model')
parser.add_argument('--generate_epochs', type=int, default=1,
                    help='save model')
parser.add_argument('--num_workers', type=int, default=0, help='Number of worker for the dataloader')
parser.add_argument('--data_augmentation', type=eval, default=False, help='use attention in the EGNN')
parser.add_argument("--conditioning", nargs='+', default=[],
                    help='arguments : homo | lumo | alpha | gap | mu | Cv' )
parser.add_argument('--resume', type=str, default= None,
                    help='')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='')
parser.add_argument('--ema_decay', type=float, default=0.999,
                    help='Amount of EMA decay, 0 means off. A reasonable value'
                         ' is 0.999.')
parser.add_argument('--augment_noise', type=float, default=0)
parser.add_argument('--normalize_factors', type=eval, default=[1, 8, 1],
                    help='normalize factors for [x, categorical, integer]')
parser.add_argument('--remove_h', action='store_true')
parser.add_argument('--include_charges', type=eval, default=False,
                    help='include atom charge or not')
parser.add_argument('--load_charges', type=eval, default=True,
                    help='load atom charge or not')
parser.add_argument('--normalization_factor', type=float, default=1,
                    help="Normalize the sum aggregation of EGNN")
parser.add_argument('--aggregation_method', type=str, default='sum',
                    help='"sum" or "mean"')
args = parser.parse_args()

#set workpath
workpath = os.path.join('pretrained_models', args.exp_name)
os.makedirs(workpath,exist_ok=True)
set_logger(workpath, 'logs.txt')
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32

dataset_info = get_dataset_info(args.dataset, args.remove_h)

if args.resume is not None:
    exp_name = args.exp_name
    resume = args.resume
    wandb_usr = args.wandb_usr
    normalization_factor = args.normalization_factor
    aggregation_method = args.aggregation_method

    with open(join(args.resume, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)

    args.resume = resume
    args.break_train_epoch = False

    args.exp_name = exp_name
    args.start_epoch = args.start_epoch
    args.wandb_usr = wandb_usr

    # Careful with this -->
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = normalization_factor
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = aggregation_method

    logging.info(args)

utils.create_folders(args)

dataloaders, charge_scale = dataset.retrieve_dataloaders(args)
args.context_node_nf = 0

# Create EGNN flow
model, nodes_dist, prop_dist = get_model(args, device, dataset_info, dataloaders['train'])
model = model.to(device)
optim = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.n_epochs)

gradnorm_queue = utils.Queue()
gradnorm_queue.add(3000)  # Add large value that will be flushed.

import torch
from torch.utils.data import Dataset, DataLoader
import os

charge_dict ={ 'C': 6, 'F': 9, 'H': 1, 'O': 8,'N':7}


import torch
from torch.utils.data import Dataset, DataLoader
import os

def process_xyz(datafile, charge_dict, included_species):
    xyz_lines = [line.decode('UTF-8') for line in datafile.readlines()]

    num_atoms = int(xyz_lines[0])
    mol_xyz = xyz_lines[2:num_atoms+2]

    atom_charges, atom_positions = [], []
    for line in mol_xyz:
        atom, posx, posy, posz = line.replace('*^', 'e').split()
        atom_charges.append(charge_dict[atom])
        atom_positions.append([float(posx), float(posy), float(posz)])

    # Convert atom charges to tensor and generate one-hot encoding
    atom_charges_tensor = torch.tensor(atom_charges, dtype=torch.long)
    one_hot = atom_charges_tensor.unsqueeze(-1) == included_species.unsqueeze(0)

    molecule = {
        'num_atoms': num_atoms,
        'charges': atom_charges_tensor,
        'positions': torch.tensor(atom_positions, dtype=torch.float),
        'one_hot': one_hot.to(torch.float)  # Convert boolean to float for the one-hot encoding
    }

    return molecule
all_charges = []
source_directory = '/home/chao/3dmolgen/data/small_mol_5'
for filename in os.listdir(source_directory):
    if filename.endswith('.xyz'):
        file_path = os.path.join(source_directory, filename)
        with open(file_path, 'rb') as datafile:
            xyz_lines = [line.decode('UTF-8') for line in datafile.readlines()]
            for line in xyz_lines[2:int(xyz_lines[0])+2]:
                atom = line.split()[0]
                all_charges.append(charge_dict[atom])

# Calculate included_species based on all encountered charges
all_charges_tensor = torch.tensor(all_charges, dtype=torch.long)
included_species = torch.unique(all_charges_tensor, sorted=True)
def batch_stack(props):
    """
    Stack a list of torch.tensors so they are padded to the size of the
    largest tensor along each axis.

    Parameters
    ----------
    props : list of Pytorch Tensors
        Pytorch tensors to stack\

    Returns
    -------
    props : Pytorch tensor
        Stacked pytorch tensor.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        return torch.tensor(props)
    elif props[0].dim() == 0:
        return torch.stack(props)
    else:
        return torch.nn.utils.rnn.pad_sequence(props, batch_first=True, padding_value=0)


def drop_zeros(props, to_keep):
    """
    Function to drop zeros from batches when the entire dataset is padded to the largest molecule size.

    Parameters
    ----------
    props : Pytorch tensor
        Full Dataset


    Returns
    -------
    props : Pytorch tensor
        The dataset with  only the retained information.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        return props
    elif props[0].dim() == 0:
        return props
    else:
        return props[:, to_keep, ...]


class PreprocessData:
    def __init__(self, load_charges=True):
        self.load_charges = load_charges

    def add_trick(self, trick):
        self.tricks.append(trick)

    def collate_fn(self, batch):
        """
        Collation function that collates datapoints into the batch format for cormorant

        Parameters
        ----------
        batch : list of datapoints
            The data to be collated.

        Returns
        -------
        batch : dict of Pytorch tensors
            The collated data.
        """
        batch = {prop: batch_stack([mol[prop] for mol in batch]) for prop in batch[0].keys()}

        to_keep = (batch['charges'].sum(0) > 0)

        batch = {key: drop_zeros(prop, to_keep) for key, prop in batch.items()}

        atom_mask = batch['charges'] > 0
        batch['atom_mask'] = atom_mask

        #Obtain edges
        batch_size, n_nodes = atom_mask.size()
        edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)

        #mask diagonal
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask

        #edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
        batch['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)

        if self.load_charges:
            batch['charges'] = batch['charges'].unsqueeze(2)
        else:
            batch['charges'] = torch.zeros(0)
        return batch

source_directory = '/home/chao/3dmolgen/data/small_mol_5'
output_file = '/home/chao/3dmolgen/data/small_mol_5/pre_dictionary.txt'

# Initialize the list to hold all molecule dictionaries

molecule_data_list=[]
with open(output_file, 'w') as out_file:
    for filename in os.listdir(source_directory):
        if filename.endswith('.xyz'):
            file_path = os.path.join(source_directory, filename)
            with open(file_path, 'rb') as datafile:
                molecule_data = process_xyz(datafile,charge_dict,included_species)
                molecule_data_list.append(molecule_data)
from torch.utils.data import Dataset, DataLoader

class ListDataset(Dataset):
    def __init__(self, data_list):
        """
        Args:
            data_list (list): List of data points, each data point being a dictionary.
        """
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # Here, you return the data point as is, or you could transform it to tensor if needed
        data_point = self.data_list[idx]
        # Assuming data_point is a dictionary, you might want to return it directly,
        # or process it further depending on your requirements.
        return data_point

# Assuming molecule_data_list is your list of molecule dictionaries
dataset = ListDataset(molecule_data_list)
preprocessor = PreprocessData(load_charges=True)
batch_processed_data = preprocessor.collate_fn(molecule_data_list)
# Create a DataLoader from your custom dataset
mol_loader = DataLoader(dataset, batch_size=200, shuffle=True, collate_fn=preprocessor.collate_fn)
for i, batch in enumerate(mol_loader):
    print(f"Batch {i}:")
    # Assuming the batch is a dictionary of tensors, which is common for complex data structures
    if isinstance(batch, dict):
        for key, value in batch.items():
            print(f"  {key}: {value.size()}")
    # If the batch is a tensor or a list/tuple of tensors
    elif isinstance(batch, torch.Tensor):
        print(f"  Tensor size: {batch.size()}")
    elif isinstance(batch, (list, tuple)):
        print("  Elements:", len(batch))
        for tensor in batch:
            print(f"    Tensor size: {tensor.size()}")
    else:
        print("  Unknown batch type:", type(batch))

    # Optional: break after the first batch to just see one example
    break

#wandb.init(project="trainfingerprint", entity="cs2485")

def main():
    if args.resume is not None:
        flow_state_dict = torch.load(join(args.resume, 'model.npy'))
        optim_state_dict = torch.load(join(args.resume, 'optim.npy'))
        model.load_state_dict(flow_state_dict)
        optim.load_state_dict(optim_state_dict)

    # Initialize dataparallel if enabled and possible.
    if args.dp and torch.cuda.device_count() > 1:
        print(f'Training using {torch.cuda.device_count()} GPUs')
        model_dp = torch.nn.DataParallel(model.cpu())
        model_dp = model_dp.cuda()
    else:
        model_dp = model

    # Initialize model copy for exponential moving average of params.
    if args.ema_decay > 0:
        if args.resume is not None:
            model_ema = copy.deepcopy(model)
            ema_state_dict = torch.load(
                join(args.resume, 'model_ema.npy'))
            model_ema.load_state_dict(ema_state_dict)
        else:
            model_ema = copy.deepcopy(model)

        ema = EMA(args.ema_decay)

        if args.dp and torch.cuda.device_count() > 1:
            model_ema_dp = torch.nn.DataParallel(model_ema)
        else:
            model_ema_dp = model_ema
    else:
        ema = None
        model_ema = model
        model_ema_dp = model_dp

    for epoch in range(args.start_epoch, args.n_epochs):
        start_epoch = time.time()
        train_epoch(args=args, loader=mol_loader, epoch=epoch, model=model, model_dp=model_dp,
                    model_ema=model_ema, ema=ema, device=device, dtype=dtype,
                    optim=optim, gradnorm_queue=gradnorm_queue, lr_scheduler=lr_scheduler)
        print(f"Epoch took {time.time() - start_epoch:.1f} seconds.")
        logging.info(f"Epoch took {time.time() - start_epoch:.1f} seconds.")

        if epoch % 250 == 0:
            utils.save_model(optim, os.path.join(workpath, 'optim_%d.npy' % (epoch)))
            utils.save_model(model, os.path.join(workpath, 'model_%d.npy' % (epoch)))
            if args.ema_decay > 0:
                utils.save_model(model_ema, os.path.join(workpath, 'model_ema_%d.npy' % (epoch)))
            with open(os.path.join(workpath, 'args_%d.pickle' % (epoch)), 'wb') as f:
                pickle.dump(args, f)

            utils.save_model(optim, os.path.join(workpath, 'optim.npy'))
            utils.save_model(model, os.path.join(workpath, 'model.npy'))
            if args.ema_decay > 0:
                utils.save_model(model_ema, os.path.join(workpath, 'model_ema.npy'))
            with open(os.path.join(workpath, 'args.pickle'), 'wb') as f:
                pickle.dump(args, f)

if __name__ == "__main__":
    main()
