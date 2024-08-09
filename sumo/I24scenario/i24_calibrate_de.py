import traci
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
import multiprocessing
import functools
import uuid
import json
from scipy.optimize import differential_evolution, OptimizeResult

main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) # two levels up
sys.path.insert(0, main_path)
import utils_data_read as reader
import utils_vis as vis
import macro

# ================ I24 scenario ====================
SCENARIO = "I24_scenario"
EXP = "3b"
MAXITER = 100 # DE
POPSIZE = 15 # DE
NUM_WORKERS = 64

with open('../config.json', 'r') as config_file:
    config = json.load(config_file)

computer_name = os.environ.get('COMPUTERNAME', 'Unknown')
if "CSI" in computer_name:
    SUMO_EXE = config['SUMO_EXE']
    
    
elif "VMS" in computer_name:
    SUMO_EXE = config['SUMO_EXE_PATH']

RDS_DIR = os.path.join("../..", "data/RDS/I24_WB_52_60_11132023.csv")


measurement_locations = [
                        # '56_7_0', '56_7_1', '56_7_2', '56_7_3', '56_7_4', 
                         '56_3_0', '56_3_1', '56_3_2', '56_3_3', '56_3_4',
                         '56_0_0', '56_0_1', '56_0_2', '56_0_3', '56_0_4',
                         '55_3_0', '55_3_1', '55_3_2', '55_3_3',
                         '54_6_0', '54_6_1', '54_6_2', '54_6_3',
                         '54_1_0', '54_1_1', '54_1_2', '54_1_3' ]

initial_guesses = {'maxSpeed': 31.534820558874827, 'minGap': 1.860096631767026, 'accel': 1.0708978903827724, 'decel': 3.8918676775882215, 'tau': 1.7949543267839752, 'lcStrategic': 1.414,
                    'lcCooperative': 1.0,
                    'lcAssertive': 1.0,
                    'lcSpeedGain': 3.76,
                    'lcKeepRight': 0.0,
                    'lcOvertakeRight': 0.877}
if "1" in EXP:
    param_names = ['maxSpeed', 'minGap', 'accel', 'decel', 'tau']
    min_val = [25.0, 0.5, 1.0, 1.0, 0.5]  
    max_val = [43.0, 3.0, 4.0, 4.0, 3] 
elif "2" in EXP:
    param_names = ['lcStrategic', 'lcCooperative', 'lcAssertive', 'lcSpeedGain']
    min_val = [0, 0, 0.0001, 0]  
    max_val = [5, 1, 5,      5] 
elif "3" in EXP:
    param_names = ['maxSpeed', 'minGap', 'accel', 'decel', 'tau', 'lcStrategic', 'lcCooperative', 'lcAssertive', 'lcSpeedGain']
    min_val = [25.0, 0.5, 1.0, 1.0, 0.5, 0, 0, 0.0001, 0]  
    max_val = [43.0, 3.0, 4.0, 4.0, 3.0, 5, 1, 5,      5] 
if "a" in EXP:
    MEAS = "volume"
elif "b" in EXP:
    MEAS = "speed"
elif "c" in EXP:
    MEAS = "occupancy"

default_params =  {'maxSpeed': 34.91628705652602,
                    'minGap': 2.9288888706657783,
                    'accel': 1.0031145478483796,
                    'decel': 2.9618821510422406,
                    'tau': 1.3051261247487569,
                    'lcStrategic': 1.414,
                    'lcCooperative': 1.0,
                    'lcAssertive': 1.0,
                    'lcSpeedGain': 3.76,
                    'lcKeepRight': 0.0,
                    'lcOvertakeRight': 0.877}

initial_guess = [initial_guesses[key] for key in param_names]

# Set up logging
log_dir = '_log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file = os.path.join(log_dir, f'DE_log_{EXP}_{MAXITER}_{POPSIZE}_{NUM_WORKERS}.log')
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# =============================================

def run_sumo(sim_config, tripinfo_output=None, fcd_output=None):
    """Run a SUMO simulation with the given configuration."""
    # command = ['sumo', '-c', sim_config, '--tripinfo-output', tripinfo_output, '--fcd-output', fcd_output]

    command = [SUMO_EXE, '-c', sim_config]
    if tripinfo_output is not None:
        command.extend(['--tripinfo-output', tripinfo_output])
        
    if fcd_output is not None:
        command.extend([ '--fcd-output', fcd_output])
        
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"SUMO simulation failed with error: {e}")
    except OSError as e:
        print(f"Execution failed: {e}")



def update_sumo_configuration(param):
    """
    Update the SUMO configuration file with the given parameters.
    
    Parameters:
        param (dict): List of parameter values [maxSpeed, minGap, accel, decel, tau]
    """
    
    # Define the path to your rou.xml file
    file_path = SCENARIO+'.rou.xml'

    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Find the vType element with id="trial"
    for vtype in root.findall('vType'):
        if vtype.get('id') == 'trial':
            # Update the attributes with the provided parameters
            for key, val in param.items():
                vtype.set(key, str(val))
            break

    # Write the updated XML content back to the file
    tree.write(file_path, encoding='UTF-8', xml_declaration=True)
    return

def create_temp_config(param):
    """
    Update the SUMO configuration file with the given parameters and save it as a new file.
    create new .rou.xml and .sumocfg files for each trial
    
    Parameters:
        param (dict): List of parameter values [maxSpeed, minGap, accel, decel, tau]
        trial_number (int): The trial number to be used for naming the new file.
    """
    trial_number = uuid.uuid4().hex # unique
    # Define the path to your original rou.xml and sumocfg files
    original_rou_file_path = SCENARIO + '.rou.xml'
    original_net_file_path = SCENARIO + '.net.xml'
    original_sumocfg_file_path = SCENARIO + '.sumocfg'
    original_add_file_path = 'I24_RDS.add.xml'
    
    # Create the directory for the new files if it doesn't exist
    output_dir = os.path.join('temp', str(trial_number))
    os.makedirs(output_dir, exist_ok=True)
    
    # ==================== .Parse the original rou.xml file ==========================
    rou_tree = ET.parse(original_rou_file_path)
    rou_root = rou_tree.getroot()

    # Find the vType element with id="trial"
    for vtype in rou_root.findall('vType'):
        if vtype.get('id') == 'trial':
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


def objective_de(params, param_names, measurement_locations, measured_output, logger):
    """Objective function for optimization."""
    # Define the parameters to be optimized
    driver_param = {param_name: params[i] for i, param_name in enumerate(param_names)}
    # print(driver_param, trial.number)
    
    # Update SUMO configuration or route files with these parameters
    temp_config_path, temp_path = create_temp_config(driver_param)

    # Run SUMO simulation
    run_sumo(temp_config_path)
    
    # Extract simulated traffic volumes
    simulated_output = reader.extract_sim_meas(measurement_locations=measurement_locations, file_dir=temp_path)
    
    # Align time
    # TODO: SIMULATED_OUTPUT starts at 5AM-8AM, while measured_output is 0-24, both in 5min intervals
    start_idx = 60 #int(5*60/5)
    end_idx = min(simulated_output[MEAS].shape[1], 36)
    end_idx_rds = start_idx + end_idx # at most three hours of simulated measurements
    
    # Calculate the objective function value
    diff = simulated_output[MEAS][:,:end_idx] - measured_output[MEAS][:, start_idx: end_idx_rds] # measured output may have nans
    mask = ~np.isnan(diff)

    # Replace NaNs with 0 in the matrix for norm calculation
    matrix_no_nan = np.where(mask, diff, 0)
    error = np.linalg.norm(matrix_no_nan)
    # logger.info(f'fun={error}, params={driver_param}')

    clear_directory(temp_path)
    
    return error



def log_progress(intermediate_result):
    if isinstance(intermediate_result, OptimizeResult):
        best_solution = intermediate_result.x
        best_value = intermediate_result.fun
        # Log the current best solution and its objective function value
        logger.info(f"Current best solution: {best_solution}, "
                    f"Objective function value: {best_value}, "
                    f"Convergence: {intermediate_result.convergence}")
    else:
        xk, convergence = intermediate_result
        logger.info(f"Current best solution: {xk}, "
                f"Convergence: {convergence}")
        

def parallel_evaluation(func, param_list, num_workers):
    # Evaluate the objective function in parallel
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(func, param_list)
    return results

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


    # ================================= get RDS data
    measured_output = reader.rds_to_matrix(rds_file=RDS_DIR, det_locations=measurement_locations)

    # ================================= run default 
    update_sumo_configuration(initial_guesses)
    run_sumo(sim_config=SCENARIO+".sumocfg")

    # ================================= Create a study object and optimize the objective function
    clear_directory("temp")
    wrapped_objective = functools.partial(objective_de, param_names=param_names, measurement_locations=measurement_locations, 
                                          measured_output=measured_output, logger=logger)
    bounds = [(min_val[i], max_val[i]) for i, _ in enumerate(param_names)]
    result = differential_evolution(wrapped_objective, bounds, 
                                    x0 = initial_guess,
                                    maxiter=MAXITER, popsize=POPSIZE, workers=lambda f, p: parallel_evaluation(f, p, NUM_WORKERS), callback=log_progress)
    print("Optimization result:", result)
    print("Best parameters found:", result.x)
    print("Objective function value at best parameters:", result.fun)
    with open(f'calibration_result/result_{EXP}.pkl', 'wb') as f:
        pickle.dump(result, f)


