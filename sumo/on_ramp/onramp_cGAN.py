import os
import numpy as np
import torch
import optuna
import subprocess
import xml.etree.ElementTree as ET
import shutil
import logging
from datetime import datetime
import sys
import glob
import json
from onramp_pretrain_cGAN import Discriminator, extract_od_matrix
from onramp_calibrate import clear_directory

main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) # two levels up
sys.path.insert(0, main_path)
import utils_data_read as reader
optuna.logging.set_verbosity(optuna.logging.ERROR)
device="cuda" if torch.cuda.is_available() else "cpu"

# ================ Configuration ====================
SCENARIO = "onramp"
N_TRIALS = 10000
SUMO_DIR = os.path.dirname(os.path.abspath(__file__))  # Current script directory
with open('../config.json', 'r') as config_file:
    config = json.load(config_file)
measurement_locations = ['upstream_0', 'upstream_1', 'merge_0', 'merge_1', 'merge_2', 'downstream_0', 'downstream_1']

# Define parameter ranges for calibration
param_names = ['maxSpeed', 'minGap', 'accel', 'decel', 'tau', 'lcStrategic', 'lcCooperative', 'lcAssertive', 'lcSpeedGain', 'lcKeepRight']
min_val = [30.0, 1.0, 1.0, 1.0, 0.5, 0, 0, 0.0001, 0, 0]  
max_val = [35.0, 3.0, 4.0, 3.0, 2.0, 5, 1, 5,      5, 5] 

# ================ SUMO Simulation Functions ====================
def run_sumo(sim_config):
    """Run a SUMO simulation with the given configuration."""
    command = ['sumo', '-c', sim_config, '--no-step-log', '--xml-validation', 'never']
    subprocess.run(command, check=True)

def create_temp_config(param, trial_number):
    """Create temporary SUMO configuration files for each trial."""
    output_dir = os.path.join('temp', str(trial_number))
    os.makedirs(output_dir, exist_ok=True)
    
    # Update .rou.xml file with new parameters
    rou_tree = ET.parse(f"{SCENARIO}.rou.xml")
    rou_root = rou_tree.getroot()
    for vtype in rou_root.findall('vType'):
        if vtype.get('id') == 'trial':
            for key, val in param.items():
                vtype.set(key, str(val))
            break
    new_rou_file_path = os.path.join(output_dir, f"{trial_number}_{SCENARIO}.rou.xml")
    rou_tree.write(new_rou_file_path, encoding='UTF-8', xml_declaration=True)
    
    # Copy other necessary files
    shutil.copy(f"{SCENARIO}.net.xml", os.path.join(output_dir, f"{trial_number}_{SCENARIO}.net.xml"))
    shutil.copy('detectors.add.xml', os.path.join(output_dir, f"{trial_number}_detectors.add.xml"))
    
    # Update .sumocfg file
    sumocfg_tree = ET.parse(f"{SCENARIO}.sumocfg")
    sumocfg_root = sumocfg_tree.getroot()
    input_element = sumocfg_root.find('input')
    if input_element is not None:
        input_element.find('route-files').set('value', f"{trial_number}_{SCENARIO}.rou.xml")
        input_element.find('net-file').set('value', f"{trial_number}_{SCENARIO}.net.xml")
        input_element.find('additional-files').set('value', f"{trial_number}_detectors.add.xml")
    new_sumocfg_file_path = os.path.join(output_dir, f"{trial_number}_{SCENARIO}.sumocfg")
    sumocfg_tree.write(new_sumocfg_file_path, encoding='UTF-8', xml_declaration=True)
    
    return new_sumocfg_file_path, output_dir


# ================ Objective Function to train the generator (SUMO)====================
def objective(trial):
    """Objective function for optimization with cGAN."""
    # Define the parameters to be optimized
    driver_param = {
        param_name: trial.suggest_uniform(param_name, min_val[i], max_val[i])
        for i, param_name in enumerate(param_names)
    } # TODO: add constraints using stability and RDC
    
    # Update SUMO configuration or route files with these parameters
    temp_config_path, temp_path = create_temp_config(driver_param, trial.number)

    # Run SUMO simulation
    run_sumo(temp_config_path)
    
    # Extract simulated traffic patterns (flow, density, speed)
    simulated_output = reader.extract_sim_meas(["trial_"+ location for location in measurement_locations],
                                        file_dir = temp_path)
    
    # Reshape simulated output and OD matrix for the discriminator
    traffic_pattern = np.stack([simulated_output['flow'], simulated_output['density'], simulated_output['speed']], axis=-1)
    traffic_pattern = np.expand_dims(traffic_pattern, axis=0)  # Add batch dimension
    traffic_pattern = torch.tensor(traffic_pattern, dtype=torch.float32)
    
    # TODO: supply SUMO with ground truth OD
    od_matrix = extract_od_matrix(os.path.join(os.getcwd(), "onramp_gt.rou.xml"))
    od_matrix = np.expand_dims(od_matrix, axis=0)  # Add batch dimension
    od_matrix = torch.tensor(od_matrix, dtype=torch.float32)
    
    # Evaluate realism using the discriminator
    realism_score = discriminator(traffic_pattern, od_matrix).item() # should be 0-1
    realism_score = torch.sigmoid(torch.tensor(realism_score))
    
    # Adversarial loss: encourage the discriminator to classify the simulated pattern as real
    adversarial_loss = -torch.log(realism_score)
    # logging.info(f'Realism score: {realism_score}, adversarial loss: {adversarial_loss.item()}')
    
    # Clear temporary files
    clear_directory(os.path.join("temp", str(trial.number)))
    
    return adversarial_loss.item()


def logging_callback(study, trial):
    # if trial.state == optuna.trial.TrialState.COMPLETE:
    #     logging.info(f'Trial {trial.number} succeeded: value={trial.value}, params={trial.params}')
    if trial.state == optuna.trial.TrialState.FAIL:
        logging.error(f'Trial {trial.number} failed: exception={trial.user_attrs.get("exception")}')
    
    # Ensure at least one trial is completed before accessing best_trial
    if len(study.trials) > 0 and any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        if study.best_trial.number == trial.number:
            logging.info(f'Current Best Trial: {study.best_trial.number}, best value: {study.best_value}')
            logging.info(f'Current Best Parameters: {study.best_params}')

# ================ Main Script ====================
if __name__ == "__main__":
    # Initialize logging
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = '_log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f'{current_time}_optuna_log.txt')
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

    # Load pre-trained discriminator
    discriminator = Discriminator(traffic_shape=(10, 7, 10, 3), od_shape=(10, 2, 1)) #(num_samples, space, time, features)
    discriminator.load_state_dict(torch.load("discriminator.pth"))
    discriminator.to(device)  # Move model to the appropriate device

    # Run optimization
    sampler = optuna.samplers.TPESampler(seed=10)
    pruner = optuna.pruners.SuccessiveHalvingPruner()
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=16, callbacks=[logging_callback])

    # Save results
    print('Best parameters:', study.best_params)

