import argparse
import os
from tool.utils import available_devices,format_devices
device = available_devices(threshold=11000,n_devices=1)
os.environ["CUDA_VISIBLE_DEVICES"] = format_devices(device)
import numpy as np
import torch
import pickle
from configs.datasets_config import get_dataset_info
from qm9 import dataset
import qm9.visualizer as vis
import os
from tool.utils import set_logger,timenow
from tool.reproducibility import set_seed
import logging
from util.utils import assert_mean_zero_with_mask, assert_correctly_masked
from qm9.models import DistributionProperty,DistributionNodes
from energys_fingerprint.models import EGNN_energy_QM9
from energys_fingerprint.en_diffusion import EnergyDiffusion
from models_fingerprint.models import EGNN_dynamics_QM9
from models_fingerprint.en_diffusion import EnVariationalDiffusion

def get_model(device, dataset_info, dataloader_train):
    histogram = dataset_info['n_nodes']
    in_node_nf = len(dataset_info['atom_decoder']) + int(args.include_charges)
    nodes_dist = DistributionNodes(histogram)

    prop_dist = None
    if len(args.conditioning) > 0:
        prop_dist = DistributionProperty(dataloader_train, args.conditioning)

    if args.condition_time:
        dynamics_in_node_nf = in_node_nf + 1
    else:
        print('Warning: dynamics model is _not_ conditioned on time.')
        dynamics_in_node_nf = in_node_nf

    net_dynamics = EGNN_dynamics_QM9(
        in_node_nf=dynamics_in_node_nf, context_node_nf=args.context_node_nf,
        n_dims=3, device=device, hidden_nf=args.nf,
        act_fn=torch.nn.SiLU(), n_layers=args.n_layers,
        attention=args.attention, tanh=args.tanh, mode=args.model, norm_constant=args.norm_constant,
        inv_sublayers=args.inv_sublayers, sin_embedding=args.sin_embedding,
        normalization_factor=args.normalization_factor, aggregation_method=args.aggregation_method)

    

    if args.probabilistic_model == 'diffusion':
        vdm = EnVariationalDiffusion(
            dynamics=net_dynamics,
            in_node_nf=in_node_nf,
            n_dims=3,
            timesteps=args.diffusion_steps,
            noise_schedule=args.diffusion_noise_schedule,
            noise_precision=args.diffusion_noise_precision,
            loss_type=args.diffusion_loss_type,
            norm_values=args.normalize_factors,
            include_charges=args.include_charges
            )
        
        )
        return vdm, guidance, nodes_dist, prop_dist

    else:
        raise ValueError(args.probabilistic_model)


def sample(args, device, generative_model,guidance,l, dataset_info,
           nodesxsample=torch.tensor([10]), context=None,
           fix_noise=False):
    max_n_nodes = dataset_info['max_n_nodes']  # this is the maximum node_size in QM9

    assert int(torch.max(nodesxsample)) <= max_n_nodes
    batch_size = len(nodesxsample)

    node_mask = torch.zeros(batch_size, max_n_nodes)
    for i in range(batch_size):
        node_mask[i, 0:nodesxsample[i]] = 1

    # Compute edge_mask
    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.view(batch_size * max_n_nodes * max_n_nodes, 1).to(device)
    node_mask = node_mask.unsqueeze(2).to(device)

    if args.probabilistic_model == 'diffusion':
        x, h = generative_model.sample(batch_size, max_n_nodes, node_mask, edge_mask, context, fix_noise=fix_noise,guidance=guidance,l=l)

        assert_correctly_masked(x, node_mask)
        assert_mean_zero_with_mask(x, node_mask)

        one_hot = h['categorical']
        charges = h['integer']

        assert_correctly_masked(one_hot.float(), node_mask)
        if args.include_charges:
            assert_correctly_masked(charges.float(), node_mask)

    else:
        raise ValueError(args.probabilistic_model)

    return one_hot, charges, x, node_mask

import openbabel as ob
import pybel
from ase.data import atomic_masses

def compute_fingerprint(poss,numberss,num_atomss):
    fingerprint_1024 = []
    fingerprint_bits = []
    ids = len(num_atomss)
    for i in range(ids):
        pos = poss[i, :]
        numbers = numberss[i, :].squeeze()
        num_atoms = num_atomss[i]

        numbers = numbers[:num_atoms]
        pos = pos[:num_atoms]

        # minius compute mass
        m = atomic_masses[numbers]
        com = np.dot(m, pos) / m.sum()
        pos = pos - com

        # order atoms by distance to center of mass
        d = torch.sum(pos ** 2, dim=1)
        center_dists = torch.sqrt(torch.maximum(d, torch.zeros_like(d)))
        idcs_sorted = torch.argsort(center_dists)
        pos = pos[idcs_sorted]
        numbers = numbers[idcs_sorted]

        # Open Babel OBMol representation
        obmol = ob.OBMol()
        obmol.BeginModify()
        # set positions and atomic numbers of all atoms in the molecule
        for p, n in zip(pos, numbers):
            obatom = obmol.NewAtom()
            obatom.SetAtomicNum(int(n))
            obatom.SetVector(*p.tolist())
        # infer bonds and bond order
        obmol.ConnectTheDots()
        obmol.PerceiveBondOrders()
        obmol.EndModify()
        _fp = pybel.Molecule(obmol).calcfp()
        fp_bits = {*_fp.bits}
        fingerprint_bits.append(fp_bits)

        fp_32 = np.array(_fp.fp, dtype=np.uint32)
        # convert fp to 1024bit
        fp_1024 = np.array(fp_32, dtype='<u4')
        fp_1024 = torch.FloatTensor(
            np.unpackbits(fp_1024.view(np.uint8), bitorder='little'))
        fingerprint_1024.append(fp_1024)

    return fingerprint_bits,fingerprint_1024

def tanimoto(fp1_bitss,fp2_bitss):
    s_sum = 0.0
    bs = len(fp1_bitss)
    for i in range(bs):
        fp1_bits = fp1_bitss[i]
        fp2_bits = fp2_bitss[i]
        n_equal = len(fp1_bits.intersection(fp2_bits))
        if len(fp1_bits) + len(fp2_bits) == 0:  # edge case with no set bits
            s = 1.0
        else:
            s = n_equal / (len(fp1_bits)+len(fp2_bits)-n_equal)
        s_sum += s
    return s_sum/bs

def h_to_charges(one_hots):
    # one_hot = one_hots[:,0]
    bs,n_nodes,_ = one_hots.size()
    charges = torch.zeros(bs,n_nodes,dtype=torch.int64)
    name = torch.tensor([1,6,7,8,9],dtype=torch.int64)
    index = (one_hots == 1).nonzero(as_tuple=True)
    charge  = torch.index_select(name, 0, index[2])
    charges[index[0],index[1]] = charge
    return charges.unsqueeze(2)

def read_small_mol_file(filepath):
    molecules_dict = {}
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split(", Num_atoms: ")
            cid, fp_str = parts[0].split(": ")
            num_atoms = int(parts[1])
            fp_1024 = torch.tensor([float(bit) for bit in fp_str], dtype=torch.float32)
            # Use CID as the key and store a dictionary with fp_1024 and num_atoms as the value
            molecules_dict[cid] = {'fp_1024': fp_1024, 'num_atoms': num_atoms}
    return molecules_dict

 # Assuming we're dealing with batch data and want the mean loss
import torch
from torch.utils.data import Dataset, DataLoader
import os

charge_dict ={
    
   'At':85,'Sm':62,   'Ar':18, 'Ag': 47, 'Eu': 63,'Al': 13, 'As': 33, 'Au': 79, 'B': 5, 'Ba': 56, 'Be': 4, 'Br': 35, 'C': 6, 
    'Ca': 20, 'Cd': 48, 'Ce': 58, 'Cl': 17, 'Co': 27, 'Cr': 24, 'Cs': 55, 'Cu': 29, 'F': 9, 
    'Fe': 26, 'Ga': 31, 'Gd': 64, 'Ge': 32, 'H': 1, 'He': 2, 'Hg': 80, 'I': 53, 'Ir': 77, 
    'K': 19, 'La': 57, 'Li': 3, 'Mg': 12, 'Mn': 25, 'Mo': 42, 'N': 7, 'Na': 11, 'Nb': 41, 
    'Ni': 28, 'O': 8, 'Os': 76, 'P': 15, 'Pb': 82, 'Pd': 46, 'Po': 84, 'Pr': 59, 'Pt': 78, 
    'Rb': 37, 'Ru': 44, 'S': 16, 'Sb': 51, 'Sc': 21, 'Se': 34, 'Si': 14, 'Sn': 50, 'Te': 52, 
    'Ti': 22, 'Tl': 81, 'V': 23, 'W': 74, 'Xe': 54, 'Y': 39, 'Yb': 70, 'Zn': 30, 'Zr': 40
}


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
source_directory = '/home/chao/3dmolgen/small_mols'
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

source_directory = '/home/chao/3dmolgen/small_mols'
output_file = '/home/chao/3dmolgen/small_mols/pre_dictionary.txt'

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

def train(dataloaders, device, dataset_info, args_gen, model, optimizer, guidance, l, loss_fn, epochs, result_path=None, save=False, dtype=torch.float32):
    max_nodes = 50
    for epoch in range(epochs):
        model_dp.train()
        model.train()
        total_loss = 0
        count = 0
        for i, data in enumerate(dataloaders):
            optimizer.zero_grad()

            # compute fingerprint
            fp_bits, fp_1024 = compute_fingerprint(data['positions'], data['charges'], data['num_atoms'])
            fp_1024 = torch.stack(fp_1024)
            fp_1024 = fp_1024.to(device, dtype)
            fp_1024 = fp_1024.unsqueeze(1).repeat(1, max_nodes, 1)

            # generate molecules conditioned on fingerprint
            nodesxsample = data['num_atoms']
            batch_size = len(nodesxsample)
            one_hot, charges, x, node_mask = sample(args_gen, device, model, guidance, l, dataset_info, nodesxsample=nodesxsample, context=fp_1024)
            charges = h_to_charges(one_hot.cpu().detach())
            gen_fp_bits, _ = compute_fingerprint(x.cpu().detach(), charges, data['num_atoms'])
            simi = tanimoto(fp_bits, gen_fp_bits)


            # Compute loss here (you'll need to define how you calculate your loss based on your specific task)
            # For example, if your task involves predicting charges, you might have:
            # actual_charges = data['charges'].to(device, dtype)
            # loss = loss_fn(predicted_charges, actual_charges)
            # This is a placeholder for whatever your actual loss computation needs to be
            loss = 1-simi  # Ensure you have a target field in your dataset
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size
            count += batch_size

            if save and i % 10 == 0:  # Example condition to save periodically
                save_path = os.path.join(result_path, 'training_samples_epoch_{}'.format(epoch))
                os.makedirs(save_path, exist_ok=True)
                # Assuming vis.save_xyz_file is a method to save your data; adjust as necessary
                vis.save_xyz_file(save_path, one_hot, charges, x, dataset_info, count, name='training', node_mask=node_mask)

            logging.info("Epoch %d, Iteration %d, Loss: %.4f" % (epoch, i, loss.item()))

        average_loss = total_loss / count
        print("Epoch: %d, Average Loss: %.4f" % (epoch, average_loss))
        logging.info("Epoch: %d, Average Loss: %.4f" % (epoch, average_loss))

    # Optionally, save your model here
    if save:
        torch.save(model.state_dict(), os.path.join(result_path, 'model_epoch_{}.pth'.format(epoch)))


def get_args_gen(args_path,argse_path):
    logging.info(f'args_path:{args_path}')
    with open(args_path, 'rb') as f:
        args_gen = pickle.load(f)

    logging.info(f'argse_path:{argse_path}')
    with open(argse_path, 'rb') as f:
        args_en = pickle.load(f)

    # Add missing args!
    if not hasattr(args_gen, 'normalization_factor'):
        args_gen.normalization_factor = 1
    if not hasattr(args_gen, 'aggregation_method'):
        args_gen.aggregation_method = 'sum'

    return args_gen,args_en


def get_generator(model_path, guidance_path,dataloaders, device, args_gen,args_en):
    dataset_info = get_dataset_info(args_gen.dataset, args_gen.remove_h)

    #model
    model, guidance,nodes_dist, prop_dist = get_model(args_gen,args_en, device, dataset_info, dataloaders['train'])
    logging.info(f'model_path:{model_path}')
    model_state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(model_state_dict)
    logging.info(f'energy_path:{guidance_path}')
    energy_state_dict = torch.load(guidance_path, map_location='cpu')
    guidance.load_state_dict(energy_state_dict)

    return model.to(device), guidance.to(device),dataset_info


def get_dataloader(args_gen):
    dataloaders, charge_scale = dataset.retrieve_dataloaders(args_gen,args_gen.shuffle_data)
    return dataloaders



def main_quantitative(args):
    # args
    args_gen, args_en = get_args_gen(args.args_generators_path, args.args_energy_path)
    args_gen.batch_size = args.batch_size
    args_gen.load_charges = True
    args_gen.shuffle_data = False
    #dataloader
    dataloaders = get_dataloader(args_gen)

    #load conditional EDM and fingerprint prediction model
    model, guidance, dataset_info = get_generator(args.generators_path, args.energy_path, dataloaders,
                                                    args.device, args_gen, args_en)
    # compute similarity on the test dataset
    test(dataloaders['test'], args.device, dataset_info, args_gen, model, guidance,args.l,args.result_path,args.save)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='eegsde_qm9_fingerprint')
    parser.add_argument('--l', type=float, default=0.5, help='the sacle of guidance')
    parser.add_argument('--generators_path', type=str)
    parser.add_argument('--args_generators_path', type=str)
    parser.add_argument('--energy_path', type=str)
    parser.add_argument('--args_energy_path', type=str)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--save', type=str, default=True, help='whether save the generated molecules as txt')

    args = parser.parse_args()
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    set_seed(1234)

    args.result_path = os.path.join('outputs', args.exp_name, 'l_' + str(args.l))
    os.makedirs(args.result_path, exist_ok=True)
    set_logger(args.result_path, 'logs.txt')
    main_quantitative(args)
