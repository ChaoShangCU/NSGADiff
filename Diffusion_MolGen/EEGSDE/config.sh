# config.sh

# Specify the code to be modified in utils.py
code_to_modify_utils="property_norms[property_key]['mean']=-21.00"

# Specify the code to be modified in rdkit_functions.py
code_to_modify_rdkit="with open(f'valid_molecules_Cv.txt', 'w') as file:"

# Specify the path to the Python script
python_script="python run_EEGSDE_single_property.py --exp_name eegsde_CVexp --l 10.0 --property Cv --generators_path pretrained_models/cEDM_Cv/generative_model_ema.npy --args_generators_path pretrained_models/cEDM_Cv/args.pickle --energy_path pretrained_models/predict_Cv/model_ema.npy --args_energy_path pretrained_models/predict_Cv/args.pickle --classifiers_path pretrained_models/evaluate_Cv/best_checkpoint.npy --args_classifiers_path pretrained_models/evaluate_Cv/args.pickle --batch_size 50 --iterations 39 --save True"

