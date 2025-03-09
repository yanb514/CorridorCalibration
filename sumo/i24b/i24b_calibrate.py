'''
Use optuna for optimization
Faster than Differential Evolution
Optuna allows
- initial guess
- parallel workers
- log progress
'''
import optuna
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

main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) # two levels up
sys.path.insert(0, main_path)
import utils_data_read as reader

# ================ CONFIGURATION ====================
with open('../config.json', 'r') as config_file:
    config = json.load(config_file)

computer_name = os.environ.get('HOSTNAME', 'Unknown')

if "CSI" in computer_name:
    SUMO_EXE = config['SUMO_PATH']["CSI"]
elif "VMS" in computer_name:
    SUMO_EXE = config['SUMO_PATH']["VMS"]
else: # run on SOL
    SUMO_EXE = config['SUMO_PATH']["SOL"]

SCENARIO = "i24b"
EXP = "3b" # experiment label
N_TRIALS = 10000 # config["N_TRIALS"] # optimization trials
N_JOBS = 120 # config["N_JOBS"] # cores
RDS_DIR = "detector_measurements_i24wRDSdetectors.csv"
# RDS_DIR = os.path.join("../..", "data/RDS/I24_WB_52_60_11132023.csv")
# ================================================

# follows convention e.g., 56_7_0, milemarker 56.7, lane 1

DEFAULT_PARAMS = {
    "maxSpeed": 30.55,
    "minGap": 2.5,
    "accel": 1.5,
    "decel": 2,
    "tau": 1.4,
    "emergencyDecel": 4.0,
    "laneChangeModel": "SL2015",
    "lcSublane": 1.0,
    "latAlignment": "arbitrary",
    "maxSpeedLat": 1.4,
    "lcAccelLat": 0.7,
    "minGapLat": 0.4,
    "lcStrategic": 10.0,
    "lcCooperative": 1.0,
    "lcPushy": 0.4,
    "lcImpatience": 0.9,
    "lcSpeedGain": 1.5,
    "lcKeepRight": 0.0,
    "lcOvertakeRight": 0.0
}

if "1" in EXP:
    param_names = ['maxSpeed', 'minGap', 'accel', 'decel', 'tau']
    min_val = [30.0, 1.0, 1.0, 1.0, 0.5]  
    max_val = [35.0, 3.0, 4.0, 3.0, 2.0] 
elif "2" in EXP:
    param_names = ['lcSublane', 'maxSpeedLat', 'lcAccelLat', 'minGapLat', 'lcStrategic', 
                   'lcCooperative', 'lcPushy','lcImpatience','lcSpeedGain', ]
    min_val = [0,  0,  0,  0, 0,  0, 0, 0, 0, 0, 0]  
    max_val = [10, 10, 5,  5, 10, 1, 1, 1, 1, 1, 1] 
elif "3" in EXP:
    param_names = ['maxSpeed', 'minGap', 'accel', 'decel', 'tau', 
                   'lcSublane', 'maxSpeedLat', 'lcAccelLat', 'minGapLat', 'lcStrategic', 
                   'lcCooperative', 'lcPushy','lcImpatience','lcSpeedGain']
    min_val = [30.0, 1.0, 1.0, 1.0, 0.5, 0,  0,  0,  0, 0,  0, 0, 0, 0]  
    max_val = [35.0, 3.0, 4.0, 3.0, 2.0, 10, 10, 5,  5, 10, 1, 1, 1, 1] 

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
        print(f"SUMO simulation failed with error: {e}")
    except OSError as e:
        print(f"Execution failed: {e}")


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


def create_temp_config(param, trial_number):
    """
    Update the SUMO configuration file with the given parameters and save it as a new file.
    create new .rou.xml and .sumocfg files for each trial
    
    Parameters:
        param (dict): List of parameter values [maxSpeed, minGap, accel, decel, tau]
        trial_number (int): The trial number to be used for naming the new file.
    """
    
    # Define the path to your original rou.xml and sumocfg files
    original_rou_file_path = SCENARIO + '.rou.xml'
    original_net_file_path = SCENARIO + '.net.xml'
    original_sumocfg_file_path = SCENARIO + '.sumocfg'
    original_add_file_path = '12-15_detectors.xml'
    
    # Create the directory for the new files if it doesn't exist
    output_dir = os.path.join('temp', str(trial_number))
    os.makedirs(output_dir, exist_ok=True)
    
    # ==================== .Parse the original rou.xml file ==========================
    rou_tree = ET.parse(original_rou_file_path)
    rou_root = rou_tree.getroot()

    # Find the vType element with id="hdv"
    for vtype in rou_root.findall('vType'):
        if vtype.get('id') == 'hdv':
            # Update the attributes with the provided parameters
            for key, val in param.items():
                vtype.set(key, str(val))
            break

    new_rou_file_path = os.path.join(output_dir, f"{trial_number}_{SCENARIO}.rou.xml")
    rou_tree.write(new_rou_file_path, encoding='UTF-8', xml_declaration=True)

    # ==================== copy original net.xml file ==========================
    shutil.copy(original_net_file_path, os.path.join(output_dir, f"{trial_number}_{SCENARIO}.net.xml"))

    # ==================== copy original add.xml file ==========================
    new_add_file_path = os.path.join(output_dir, f"{trial_number}_{original_add_file_path}")
    shutil.copy(original_add_file_path, new_add_file_path)
    
    #  ==================== parse original sumocfg.xml file ==========================
    sumocfg_tree = ET.parse(original_sumocfg_file_path)
    sumocfg_root = sumocfg_tree.getroot()
    input_element = sumocfg_root.find('input')
    if input_element is not None:
        input_element.find('route-files').set('value', f"{trial_number}_{SCENARIO}.rou.xml")
        input_element.find('net-file').set('value', f"{trial_number}_{SCENARIO}.net.xml")
        input_element.find('additional-files').set('value',  f"{trial_number}_{original_add_file_path}")

    new_sumocfg_file_path = os.path.join(output_dir, f"{trial_number}_{SCENARIO}.sumocfg")
    sumocfg_tree.write(new_sumocfg_file_path, encoding='UTF-8', xml_declaration=True)
    
    return new_sumocfg_file_path, output_dir


def objective(trial):
    """Objective function for optimization."""
    # Define the parameters to be optimized
    driver_param = {
        param_name: trial.suggest_uniform(param_name, min_val[i], max_val[i])
        for i, param_name in enumerate(param_names)
    }
    # print(driver_param, trial.number)
    
    # Update SUMO configuration or route files with these parameters
    temp_config_path, temp_path = create_temp_config(driver_param, trial.number)

    # Run SUMO simulation
    run_sumo(temp_config_path)
    
    # Extract simulated traffic volumes
    simulated_output = reader.extract_sim_meas_i24b(det_locations, xml_file=f"{temp_path}/out.xml") 
    
    # --- RMSE ---
    diff = simulated_output[MEAS] - measured_output[MEAS] # measured output may have nans
    error = np.sqrt(np.nanmean(diff.flatten()**2))
    # --- RMSPE ---
    # relative_diff = (simulated_output[MEAS][:, :end_idx] - np.nan_to_num(measured_output[MEAS][:, start_idx:end_idx_rds], nan=0)) \
    #              / np.nan_to_num(measured_output[MEAS][:, start_idx:end_idx_rds], nan=0.1) # ensures NaN values in measured_output are replaced with 1 to avoid division by zero or NaN issues.
    # error = np.sqrt(np.nanmean((relative_diff**2).flatten()))

    clear_directory(os.path.join("temp", str(trial.number)))
    # logging.info(f'Trial {trial.number}: param={driver_param}, error={error}')
    
    return error

def logging_callback(study, trial):
    # if trial.state == optuna.trial.TrialState.COMPLETE:
    #     logging.info(f'Trial {trial.number} succeeded: value={trial.value}, params={trial.params}')
    # elif trial.state == optuna.trial.TrialState.FAIL:
    #     logging.error(f'Trial {trial.number} failed: exception={trial.user_attrs.get("exception")}')
    
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
    # det_locations = extract_detector_locations(RDS_DIR)
    # det_locations = [det_loc for det_loc in det_locations if "westbound" in det_loc] # filter westbound only
    # measured_output = reader.rds_to_matrix_i24b(rds_file=RDS_DIR, det_locations=det_locations)

    # ================================= run default 
    # update_sumo_configuration(initial_guess)
    # run_sumo(sim_config=SCENARIO+".sumocfg")
    

    # ================================= Create a study object and optimize the objective function
    # clear_directory("temp")
    # current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # log_dir = '_log'
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)
    # log_file = os.path.join(log_dir, f'{current_time}_optuna_log_{EXP}_{N_TRIALS}_{N_JOBS}.txt')
    # logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

    # sampler = optuna.samplers.TPESampler(seed=10)
    # study = optuna.create_study(direction='minimize', sampler=sampler)
    # study.enqueue_trial(initial_guess)
    # study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS, callbacks=[logging_callback])
    # try:
    #     fig = optuna.visualization.plot_optimization_history(study)
    #     fig.show()
    # except:
    #     pass
    
    # # Get the best parameters
    # best_params = study.best_params
    # print('Best parameters:', best_params)
    # with open(f'calibration_result/study_{EXP}.pkl', 'wb') as f:
    #     pickle.dump(study, f)

    # ================================= run best param 
    best_params = {'maxSpeed': 34.81165096351248, 'minGap': 1.5938065104844015, 'accel': 3.022346538786358, 'decel': 1.0428115924602528, 'tau': 0.5317315149807879, 'lcSublane': 0.002929023905628436, 'maxSpeedLat': 1.2782516047399735, 'lcAccelLat': 1.8797774681211579, 'minGapLat': 0.6986271674187223, 'lcStrategic': 1.7903870628938083, 'lcCooperative': 0.9306777856904532, 'lcPushy': 0.7751464178714065, 'lcImpatience': 0.9811640612008771, 'lcSpeedGain': 0.3439350590925962}
    update_sumo_configuration(best_params)
    run_sumo(sim_config=SCENARIO+".sumocfg", fcd_output=SCENARIO+".out.xml")
