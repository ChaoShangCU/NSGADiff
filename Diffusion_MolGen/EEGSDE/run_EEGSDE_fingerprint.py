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

def get_model(args, argse, device, dataset_info, dataloader_train):
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

    net_energy = EGNN_energy_QM9(
        in_node_nf=dynamics_in_node_nf, context_node_nf=argse.context_node_nf,
        n_dims=3, device=device, hidden_nf=argse.nf,
        act_fn=torch.nn.SiLU(), n_layers=argse.n_layers,
        attention=argse.attention, tanh=argse.tanh, mode=argse.model, norm_constant=argse.norm_constant,
        inv_sublayers=argse.inv_sublayers, sin_embedding=argse.sin_embedding,
        normalization_factor=argse.normalization_factor, aggregation_method=argse.aggregation_method)

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
        guidance = EnergyDiffusion(
            dynamics=net_energy,
            in_node_nf=in_node_nf,
            n_dims=3,
            timesteps=argse.diffusion_steps,
            noise_schedule=argse.diffusion_noise_schedule,
            noise_precision=argse.diffusion_noise_precision,
            norm_values=argse.normalize_factors,
            include_charges=argse.include_charges
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
decoder = [6, 8, 7, 1, 16, 15, 11, 20, 17, 9, 12, 34, 33, 26, 74, 53, 35, 78, 14, 30, 52, 48, 46, 22, 23, 29, 5,
           42, 58, 51, 19, 3, 24, 50, 25, 13, 80, 28, 27, 39, 82, 37, 55, 40, 4, 56, 76, 59, 77, 47, 70, 81, 85]


def h_to_charges(one_hots):
    # one_hot = one_hots[:,0]
    bs,n_nodes,_ = one_hots.size()
    charges = torch.zeros(bs,n_nodes,dtype=torch.int64)
    
    name = torch.tensor(decoder,dtype=torch.int64)
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
            #print(f"Read CID: {cid}, Num_atoms: {num_atoms}")
            molecules_dict[cid] = {'fp_1024': fp_1024, 'num_atoms': num_atoms}
    return molecules_dict

#def test(dataloaders, device,dataset_info,args_gen, model, guidance,l,result_path=None,save=False,dtype = torch.float32):
    #model.eval()
    #max_nodes = dataset_info['max_n_nodes']
    #res = {'similarity': 0, 'counter': 0, 'similarity_arr':[]}
    #count = 0
    #filepath = "/home/chao/3dmolgen/data/small_mol_5/final_output.txt"  # Correct path for Unix-like systems
    #molecules = read_small_mol_file(filepath)
    #for i, data in enumerate(dataloaders):
        # compute fingerprint
        #fp_bits,fp_1024 = compute_fingerprint(data['positions'], data['charges'], data['num_atoms'])
        #fp_1024 = torch.stack(fp_1024)
        #fp_1024 = fp_1024.to(device, dtype)
        #fp_1024 = fp_1024.unsqueeze(1).repeat(1, max_nodes, 1)
   

    #for cid, molecule_data in molecules.items():
        #fp_1024 = molecule_data['fp_1024'].to(device, dtype).unsqueeze(1).repeat(1, max_nodes, 1)
        #num_atoms = molecule_data['num_atoms']
        #nodesxsample = torch.tensor([num_atoms], dtype=torch.int, device=device)
        
        #batch_size = len(nodesxsample)
        #one_hot, charges, x, node_mask = sample(args_gen, device, model, guidance, l,
                                                #dataset_info, nodesxsample=nodesxsample,
                                                #context=fp_1024)
        #if save:
            #save_path = os.path.join(result_path,'samples',str(cid))
            #os.makedirs(save_path, exist_ok=True)
            #vis.save_xyz_file(save_path, one_hot, charges, x, dataset_info,
                #count, name='samples', node_mask=node_mask)
            #count += batch_size
        #if save:
            # Here, cid is added to the directory or filename to uniquely identify saved outputs for each molecule
            #molecule_save_path = os.path.join(result_path, 'samples', str(cid))  # Use CID in the path
            #os.makedirs(molecule_save_path, exist_ok=True)
    
    # Assuming vis.save_xyz_file saves a file, you can include CID in the filename or use molecule_save_path as the directory
            #filename = f"{cid}_samples.xyz"  # Example: Naming the file with CID prefix
            #full_save_path = os.path.join(molecule_save_path, filename)  # Combining directory and filename
    
            #vis.save_xyz_file(full_save_path, one_hot, charges, x, dataset_info,
                              #count, name='samples', node_mask=node_mask)
           # count += batch_size


        #charges = h_to_charges(one_hot.cpu().detach())
        #gen_fp_bits, _ = compute_fingerprint(x.cpu().detach(), charges, nodesxsample)
        #simi = tanimoto(fp_bits, gen_fp_bits)

        #res['similarity'] += simi * batch_size
        #res['counter'] += batch_size
        #res['similarity_arr'].append(simi)
        #logging.info(f"	CID: {cid} " )

    #average_similarity = res['similarity'] / res['counter']
    #print("similarity on test samples: %.4f" % average_similarity)
    #logging.info(average_similarity)

import itertools
import os
import torch
cid_to_generate = '70912'
def test(cid_to_generate, dataloaders, device, dataset_info, args_gen, model, guidance, l, fp_1024,ni,result_path=None, save=True, dtype=torch.float32):
    print(f"Looking for CID: {cid_to_generate}")
    model.eval()
    max_nodes = dataset_info['max_n_nodes']
    filepath = "/home/chao/3dmolgen/data/small_mol_5/final_output.txt"  # Adjust the file path as needed
    molecules = read_small_mol_file(filepath)
       #print("Available CIDs in the file:")
    #print("Available CIDs in the file:")
    #found = False
    #for cid in molecules.keys():
        #print(f"'{cid}'")
        #if cid == cid_to_generate:
            #print(f"Direct match found for CID: '{cid}'")
            #found = True
            #break

   # if not found:
        #print(f"Molecule with CID {cid_to_generate} not found.")
        #return
    #else:
        #print(f"Molecule with CID {cid_to_generate} found.") 

    molecule_data = molecules[cid_to_generate]
    #fp_1024 = molecule_data['fp_1024'].to(device, dtype).unsqueeze(1).repeat(1, max_nodes, 1)
    fp_1024 = fp_1024
    num_atoms = molecule_data['num_atoms']
    nodesxsample = torch.tensor([num_atoms], dtype=torch.int, device=device)

    one_hot, charges, x, node_mask = sample(args_gen, device, model, guidance, l,
                                            dataset_info, nodesxsample=nodesxsample,
                                            context=fp_1024)
    if save:
        molecule_save_path = os.path.join(result_path, 'samples', str(cid_to_generate))  # Use CID in the path
        print(molecule_save_path)
        os.makedirs(molecule_save_path, exist_ok=True)
        filename = f"{cid_to_generate}_samples.xyz"  # Example: Naming the file with CID prefix
        full_save_path = os.path.join(molecule_save_path, filename)  # Combining directory and filename
        print(full_save_path)

        # Save the generated molecule to a file
        vis.save_xyz_file(full_save_path, one_hot, charges, x, dataset_info, ni, name='samples', node_mask=node_mask)
        print(full_save_path)

    logging.info(f"Generated molecule for CID: {cid_to_generate}")



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
    #test(dataloaders['test'], args.device, dataset_info, args_gen, model, guidance,args.l,args.result_path,args.save)
    if args.fp_1024:
     # Split the string into individual numbers, and convert each one to float
        fp_1024_list = [float(bit) for bit in args.fp_1024.split(',')]
        fp_1024_tensor = torch.tensor(fp_1024_list, dtype=torch.float32).to(args.device).unsqueeze(1).repeat(1, 50, 1)
    else:
    # Handle the case where no fp_1024 is provided
        fp_1024_tensor = None  # Or your default way of getting fp_1024
    ni_n = args.ni if args.ni else 0 

    
    #test(cid_to_generate, dataloaders['test'], args.device, dataset_info, args_gen, model, guidance, args.l,fp_1024 args.result_path, args.save)
    test(cid_to_generate, dataloaders['test'], args.device, dataset_info, args_gen, model, guidance, args.l, fp_1024_tensor,ni_n, args.result_path, args.save)

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
    parser.add_argument('--fp_1024', type=str, help='1024-bit vector as a string of 0s and 1s')
    parser.add_argument('--ni', type=int, help='1024-bit vector as a string of 0s and 1s')



    args = parser.parse_args()
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    set_seed(1234)

    args.result_path = os.path.join('outputs', args.exp_name, 'l_' + str(args.l))
    os.makedirs(args.result_path, exist_ok=True)
    set_logger(args.result_path, 'logs.txt')
    main_quantitative(args)
