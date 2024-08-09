import traci
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
import utils_vis as vis
import utils_data_read as reader
import macro


# ================ on-ramp scenario ====================
SCENARIO = "onramp"
EXP = "1a"
MAXITER = 3 # DE
POPSIZE = 3 # DE
NUM_WORKERS = 2

SUMO_DIR = os.path.dirname(os.path.abspath(__file__)) # current script directory

with open('../config.json', 'r') as config_file:
    config = json.load(config_file)

computer_name = os.environ.get('COMPUTERNAME', 'Unknown')
if "CSI" in computer_name:
    SUMO_EXE = config['SUMO_EXE']
    
elif "VMS" in computer_name:
    SUMO_EXE = config['SUMO_EXE_PATH']

measurement_locations = ['upstream_0', 'upstream_1', 
                            'merge_0', 'merge_1', 'merge_2', 
                            'downstream_0', 'downstream_1']

if "1" in EXP:
    param_names = ['maxSpeed', 'minGap', 'accel', 'decel', 'tau']
    min_val = [25.0, 0.5, 1.0, 1.0, 0.5]  
    max_val = [40.0, 3.0, 4.0, 4.0, 2.0] 
elif "2" in EXP:
    param_names = ['lcStrategic', 'lcCooperative', 'lcAssertive', 'lcSpeedGain']
    min_val = [0, 0, 0.0001, 0]  
    max_val = [5, 1, 5,      5] 
elif "3" in EXP:
    param_names = ['maxSpeed', 'minGap', 'accel', 'decel', 'tau', 'lcStrategic', 'lcCooperative', 'lcAssertive', 'lcSpeedGain']
    min_val = [25.0, 0.5, 1.0, 1.0, 0.5, 0, 0, 0.0001, 0]  
    max_val = [40.0, 3.0, 4.0, 4.0, 2.0, 5, 1, 5,      5] 
if "a" in EXP:
    MEAS = "volume"
elif "b" in EXP:
    MEAS = "speed"
elif "c" in EXP:
    MEAS = "occupancy"


# Set up logging
log_dir = '_log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file = os.path.join(log_dir, f'DE_log_{EXP}.log')
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
        
    subprocess.run(command, check=True)




def get_vehicle_ids_from_routes(route_file):
    tree = ET.parse(route_file)
    root = tree.getroot()

    vehicle_ids = []
    for route in root.findall('.//vehicle'):
        vehicle_id = route.get('id')
        vehicle_ids.append(vehicle_id)

    return vehicle_ids



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
    # print(trial_number)
    # Define the path to your original rou.xml and sumocfg files
    original_rou_file_path = SCENARIO + '.rou.xml'
    original_net_file_path = SCENARIO + '.net.xml'
    original_sumocfg_file_path = SCENARIO + '.sumocfg'
    original_add_file_path = 'detectors.add.xml'
    
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
    # Convert the parameter vector to a dictionary
    driver_param = {param_name: params[i] for i, param_name in enumerate(param_names)}
    
    # Update SUMO configuration or route files with these parameters
    temp_config_path, temp_path = create_temp_config(driver_param)
    
    # Run SUMO simulation
    run_sumo(temp_config_path)
    
    # Extract simulated traffic volumes
    simulated_output = reader.extract_sim_meas(
        ["trial_" + location for location in measurement_locations],
        file_dir=temp_path
    )
    
    # Calculate the objective function value
    error = np.linalg.norm(simulated_output[MEAS] - measured_output[MEAS])
    logger.info(f'fun={error}, params={driver_param}')
    
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

    default_params = { "maxSpeed": 55.5, "minGap": 2.5, "accel": 2.6, "decel": 4.5, "tau": 1.0,
        "lcStrategic": 1.0,
        "lcCooperative": 1.0,
        "lcAssertive": 0.5,
        "lcSpeedGain": 1.0,
        "lcKeepRight": 0.5,
        "lcOvertakeRight": 1.0}
    update_sumo_configuration(default_params)

    # ================================= run ground truth and generate synthetic measurements
    run_sumo(sim_config=SCENARIO+"_gt.sumocfg") #, fcd_output ="trajs_gt.xml")
    measured_output = reader.extract_sim_meas(measurement_locations)

    # # =============================== optimize the objective function
    clear_directory("temp")
    wrapped_objective = functools.partial(objective_de, param_names=param_names, measurement_locations=measurement_locations, 
                                          measured_output=measured_output, logger=logger)
    bounds = [(min_val[i], max_val[i]) for i, _ in enumerate(param_names)]
    result = differential_evolution(wrapped_objective, bounds, 
                                    maxiter=MAXITER, popsize=POPSIZE, workers=lambda f, p: parallel_evaluation(f, p, NUM_WORKERS), callback=log_progress)
    print("Optimization result:", result)
    print("Best parameters found:", result.x)
    print("Objective function value at best parameters:", result.fun)
    with open(f'calibration_result/result_{EXP}.pkl', 'wb') as f:
        pickle.dump(result, f)


    # # ================================ visualize time-space using best parameters
    # best_params =  {"maxSpeed": 30.55,
    #     "minGap": 2.5,
    #     "accel": 1.5,
    #     "decel": 2.0,
    #     "tau": 1.4,
    #     "lcStrategic": 1.0,
    #     "lcCooperative": 1.0,
    #     "lcAssertive": 0.5,
    #     "lcSpeedGain": 1.0,
    #     "lcKeepRight": 0.5,
    #     "lcOvertakeRight": 1.0}
    # update_sumo_configuration(best_params)
    # run_sumo(sim_config=SCENARIO+".sumocfg")#, fcd_output ="trajs_best.xml")
    # vis.visualize_fcd("trajs_best.xml") # lanes=["E0_0", "E0_1", "E1_0", "E1_1", "E2_0", "E2_1", "E2_2", "E4_0", "E4_1"]

    # run_sumo(sim_config=SCENARIO+".sumocfg")
    # sim_output = reader.extract_sim_meas(measurement_locations=["trial_"+ location for location in measurement_locations])
    
     
    # ================================= compare GT meas. vs. simulation with custom params.======================
    # run_sumo(sim_config=SCENARIO+"_gt.sumocfg") #, fcd_output ="trajs_gt.xml")
    # best_params =  {'maxSpeed': 31.44813279984895, 'minGap': 1.8669305739182382, 'accel': 2.2398476082518677, 'decel': 2.5073714738472153, 'tau': 1.3988475504128757, 'lcStrategic': 0.8624217521963465, 'lcCooperative': 0.9789774143646455, 'lcAssertive': 0.43478229746049984, 'lcSpeedGain': 1.1383219615950644, 'lcKeepRight': 4.030227753894549, 'lcOvertakeRight': 0.9240310635518598}

    # update_sumo_configuration(best_params)
    # run_sumo(sim_config = SCENARIO+".sumocfg")
    # vis.plot_sim_vs_sim(sumo_dir, measurement_locations, quantity="speed")
    
    # ============== compute macroscopic properties ==================
    # base_name = SCENARIO+""
    # fcd_name = "fcd_"+base_name+"_cflc_rho"
    # run_sumo(sim_config = base_name+".sumocfg", fcd_output =fcd_name+".out.xml")
    
    # reader.fcd_to_csv_byid(xml_file=fcd_name+".out.xml", csv_file=fcd_name+".csv")
    # macro.reorder_by_id(fcd_name+".csv", bylane=False)
    # macro_data = macro.compute_macro(fcd_name+"_byid.csv", dx=10, dt=10, save=True, plot=True)