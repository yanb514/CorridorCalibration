'''
Use optuna for optimization
Faster than Differential Evolution
Optuna allows
- initial guess
- parallel workers
- log progress
'''
import optuna
from optuna.pruners import MedianPruner
import subprocess
import os
import os
import xml.etree.ElementTree as ET
import numpy as np
import sys
import shutil
import pickle
import logging
from datetime import datetime
import json
import pandas as pd
from pathlib import Path
import tempfile

main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) # two levels up
sys.path.insert(0, main_path)
import utils_data_read as reader

# ================ CONFIGURATION ====================
SCENARIO = "i24b"
with open('../config.json', 'r') as config_file:
    config = json.load(config_file)

# SUMO_EXE = config['SUMO_PATH'] # customize SUMO_PATH in config.json
SUMO_EXE = os.getenv("SUMO_PATH", config['SUMO_PATH'])  # Fallback to config

N_TRIALS = config["N_TRIALS"]  # config["N_TRIALS"] # optimization trials
N_JOBS = config["N_JOBS"]  # config["N_JOBS"] # cores
EXP = config["EXP"] # experiment label
RDS_DIR = config[SCENARIO]["RDS_DIR"] # directory of the RDS data
DEFAULT_PARAMS = config["DEFAULT_PARAMS"]
# ================================================


if "1" in EXP:
    params_range = config["PARAMS_RANGE"]["cf"]
elif "2" in EXP:
    params_range = config["PARAMS_RANGE"]["lc"]
elif "3" in EXP:
    params_range = {**config["PARAMS_RANGE"]["cf"], **config["PARAMS_RANGE"]["lc"]}
param_names, ranges = zip(*params_range.items())
min_val, max_val = zip(*ranges)

if "a" in EXP:
    MEAS = "flow"
elif "b" in EXP:
    MEAS = "speed"
elif "c" in EXP:
    MEAS = "occupancy"

initial_guess = {key: DEFAULT_PARAMS[key] for key in param_names if key in DEFAULT_PARAMS}


def extract_detector_locations(csv_file):
    """
    Reads a detector measurement CSV file and extracts a list of unique detector locations.

    Parameters:
    - csv_file: Path to the detector measurement CSV file.

    Returns:
    - det_locations: A list of unique detector locations in the format "milemarker-eastbound_lane".
    """
    # Read the CSV file
    df = pd.read_csv(csv_file, delimiter=';')
    
    # Extract unique detector IDs from the 'Detector' column
    detector_ids = df['Detector'].unique()
    
    # Convert detector IDs to the desired format (e.g., "555-eastbound_0")
    det_locations = [detector_id for detector_id in detector_ids]
    
    return det_locations


def run_sumo(sim_config, tripinfo_output=None, fcd_output=None):
    """Run a SUMO simulation with the given configuration."""
    # command = ['sumo', '-c', sim_config, '--tripinfo-output', tripinfo_output, '--fcd-output', fcd_output]

    command = [SUMO_EXE, '-c', sim_config, 
               '--no-step-log',  '--xml-validation', 'never',  '--no-warnings',
               '--lateral-resolution', '0.5']
    if tripinfo_output is not None:
        command.extend(['--tripinfo-output', tripinfo_output])
        
    if fcd_output is not None:
        command.extend([ '--fcd-output', fcd_output])
        
    try:
        subprocess.run(command, check=False, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        logging.error(f"SUMO failed: {e}")
        raise


def update_sumo_configuration(param):
    """
    Update the SUMO configuration file with the given parameters.
    All parameters in .rou.xml not present in the given param will be removed
    
    Parameters:
        param (dict): Dictionary of parameter values {attribute_name: value}
    """
    # Define the path to your rou.xml file
    file_path = SCENARIO + '.rou.xml'
    
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Find the vType element with id="hdv"
    for vtype in root.findall('vType'):
        if vtype.get('id') == 'hdv':
            # Remove all existing attributes
            for attr in list(vtype.attrib.keys()):
                # These params are not calibrated
                if attr not in ["id", "length", "carFollowModel","emergencyDecel", "laneChangeModel", "latAlignment", "lcKeepRight", "lcOvertakeRight"]:
                    del vtype.attrib[attr]
            
            # Set new attributes from param
            for key, val in param.items():
                vtype.set(key, str(val))
            break
    
    # Write the updated XML content back to the file
    tree.write(file_path, encoding='UTF-8', xml_declaration=True)



def create_temp_config(param: dict, temp_dir: Path, trial_number: int) -> Path:
    """
    Creates temporary SUMO configuration files with modified parameters in a specified directory.
    
    Args:
        param: Dictionary of parameter values to apply
        temp_dir: Path to temporary directory for file storage
        trial_number: Trial number for unique filenames
        
    Returns:
        Path to the created SUMO configuration file
    """
    # Base filenames
    scenario_files = {
        'rou': f"{SCENARIO}.rou.xml",
        'net': f"{SCENARIO}.net.xml",
        'add': "12-15_detectors.xml",
        'sumocfg': f"{SCENARIO}.sumocfg"
    }
    print(scenario_files)

    # Create new filenames with trial number prefix
    new_files = {
        key: temp_dir / f"{trial_number}_{filename}"
        for key, filename in scenario_files.items()
    }

    # 1. Process vehicle type parameters in rou.xml
    try:
        # Parse original routing file
        rou_tree = ET.parse(scenario_files['rou'])
        rou_root = rou_tree.getroot()

        # Find and modify the HDV vehicle type
        for vtype in rou_root.findall('vType'):
            if vtype.get('id') == 'hdv':
                # Update the attributes with the provided parameters
                for key, val in param.items():
                    vtype.set(key, str(val))
                break

        # Write modified routing file
        rou_tree.write(new_files['rou'], encoding='UTF-8', xml_declaration=True)
    
    except FileNotFoundError as e:
        raise RuntimeError(f"Missing required file: {scenario_files['rou']}") from e

    # 2. Copy static files (network and detector definitions)
    try:
        shutil.copy(scenario_files['net'], new_files['net'])
        shutil.copy(scenario_files['add'], new_files['add'])
    except FileNotFoundError as e:
        raise RuntimeError(f"Missing scenario file: {e.filename}") from e

    # 3. Update sumo configuration file
    try:
        sumocfg_tree = ET.parse(scenario_files['sumocfg'])
        sumocfg_root = sumocfg_tree.getroot()
        
        # Update file references in the configuration
        input_elem = sumocfg_root.find('input')
        if input_elem is not None:
            input_elem.find('route-files').set('value', new_files['rou'].name)
            input_elem.find('net-file').set('value', new_files['net'].name)
            input_elem.find('additional-files').set('value', new_files['add'].name)
        
        # Write modified configuration
        sumocfg_tree.write(new_files['sumocfg'], encoding='UTF-8', xml_declaration=True)
    
    except FileNotFoundError as e:
        raise RuntimeError(f"Missing sumo config file: {scenario_files['sumocfg']}") from e

    return new_files['sumocfg']


def objective(trial):
    """Objective function for optimization."""
    # Define the parameters to be optimized
    driver_param = {
        param_name: trial.suggest_uniform(param_name, min_val[i], max_val[i])
        for i, param_name in enumerate(param_names)
    }
    with tempfile.TemporaryDirectory(prefix=f"trial_{trial.number}_") as tmp_dir:
        temp_dir = Path(tmp_dir)
        config_path = create_temp_config(driver_param, temp_dir, trial.number)
        
        # Run SUMO - files automatically write to temp_dir
        run_sumo(config_path)
        
        # Process results
        output_xml = temp_dir / "out.xml"
        simulated_output = reader.extract_sim_meas_i24b(det_locations, xml_file=output_xml)

    
    # --- RMSE ---
    diff = simulated_output[MEAS] - measured_output[MEAS] # measured output may have nans
    error = np.sqrt(np.nanmean(diff.flatten()**2))
    # --- RMSPE ---
    # relative_diff = (simulated_output[MEAS][:, :end_idx] - np.nan_to_num(measured_output[MEAS][:, start_idx:end_idx_rds], nan=0)) \
    #              / np.nan_to_num(measured_output[MEAS][:, start_idx:end_idx_rds], nan=0.1) # ensures NaN values in measured_output are replaced with 1 to avoid division by zero or NaN issues.
    # error = np.sqrt(np.nanmean((relative_diff**2).flatten()))

    # clear_directory(os.path.join("temp", str(trial.number)))
    # logging.info(f'Trial {trial.number}: param={driver_param}, error={error}')
    
    return error

def logging_callback(study, trial):

    if study.best_trial.number == trial.number:
        logging.info(f'Current Best Trial: {study.best_trial.number}')
        logging.info(f'Current Best Value: {study.best_value}')
        logging.info(f'Current Best Parameters: {study.best_params}')


def clear_directory(directory_path):
    """
    Clear all files within the specified directory.
    
    Parameters:
        directory_path (str): The path to the directory to be cleared.
    """
    try:
        shutil.rmtree(directory_path)
        print(f"Directory {directory_path} and all its contents have been removed.")
    except FileNotFoundError:
        print(f"Directory {directory_path} does not exist.")
    except Exception as e:
        print(f"Error removing directory {directory_path}: {e}")


if __name__ == "__main__":

    # ================================= prepare RDS data for model calibration
    det_locations = extract_detector_locations(RDS_DIR)
    det_locations = [det_loc for det_loc in det_locations if "westbound" in det_loc] # filter westbound only
    measured_output = reader.rds_to_matrix_i24b(rds_file=RDS_DIR, det_locations=det_locations)

    # ================================= run default 
    # update_sumo_configuration(initial_guess)
    # run_sumo(sim_config=SCENARIO+".sumocfg")
    
    # # ================================= Create a study object and optimize the objective function
    # clear_directory("temp")
    # current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # log_dir = '_log'
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)
    # log_file = os.path.join(log_dir, f'{current_time}_optuna_log_{EXP}_{N_TRIALS}_{N_JOBS}.txt')
    # logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

    # sampler = optuna.samplers.TPESampler(seed=10)
    # study = optuna.create_study(direction='minimize', 
    #                             sampler=sampler,
    #                             pruner=MedianPruner()  
    #                             )
    # study.enqueue_trial(initial_guess)
    # study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS, callbacks=[logging_callback])
    # # try:
    # #     fig = optuna.visualization.plot_optimization_history(study)
    # #     fig.show()
    # # except:
    # #     pass
    
    # # Get the best parameters
    # best_params = study.best_params
    # print('Best parameters:', best_params)
    # with open(f'calibration_result/study_{EXP}.pkl', 'wb') as f:
    #     pickle.dump(study, f)

    # ================================= run best param 
    # best_params = {'maxSpeed': 34.81165096351248, 'minGap': 1.5938065104844015, 'accel': 3.022346538786358, 'decel': 1.0428115924602528, 'tau': 0.5317315149807879, 'lcSublane': 0.002929023905628436, 'maxSpeedLat': 1.2782516047399735, 'lcAccelLat': 1.8797774681211579, 'minGapLat': 0.6986271674187223, 'lcStrategic': 1.7903870628938083, 'lcCooperative': 0.9306777856904532, 'lcPushy': 0.7751464178714065, 'lcImpatience': 0.9811640612008771, 'lcSpeedGain': 0.3439350590925962}
    # update_sumo_configuration(best_params)
    # run_sumo(sim_config=SCENARIO+".sumocfg", fcd_output=SCENARIO+".out.xml")
