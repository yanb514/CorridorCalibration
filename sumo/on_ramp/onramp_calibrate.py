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
import json
from datetime import datetime

main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) # two levels up
sys.path.insert(0, main_path)
import utils_data_read as reader
import macro
import utils_vis as vis


# ================ on-ramp scenario setup ====================
SCENARIO = "onramp"
EXP = "3c"
SUMO_DIR = os.path.dirname(os.path.abspath(__file__)) # current script directory

with open('../config.json', 'r') as config_file:
    config = json.load(config_file)

computer_name = os.environ.get('COMPUTERNAME', 'Unknown')
if "CSI" in computer_name:
    SUMO_EXE = config['SUMO_EXE']
    
elif "VMS" in computer_name:
    SUMO_EXE = config['SUMO_EXE_PATH']

if "1" in EXP:
    param_names = ['maxSpeed', 'minGap', 'accel', 'decel', 'tau']
    min_val = [30.0, 1.0, 1.0, 1.0, 0.5]  
    max_val = [35.0, 3.0, 4.0, 3.0, 2.0] 
elif "2" in EXP:
    param_names = ['lcStrategic', 'lcCooperative', 'lcAssertive', 'lcSpeedGain', 'lcKeepRight']
    min_val = [0, 0, 0.0001, 0, 0]  
    max_val = [5, 1, 5,      5, 5] 
elif "3" in EXP:
    param_names = ['maxSpeed', 'minGap', 'accel', 'decel', 'tau', 'lcStrategic', 'lcCooperative', 'lcAssertive', 'lcSpeedGain', 'lcKeepRight']
    min_val = [30.0, 1.0, 1.0, 1.0, 0.5, 0, 0, 0.0001, 0, 0]  
    max_val = [35.0, 3.0, 4.0, 3.0, 2.0, 5, 1, 5,      5, 5] 
if "a" in EXP:
    MEAS = "volume"
elif "b" in EXP:
    MEAS = "speed"
elif "c" in EXP:
    MEAS = "occupancy"


measurement_locations = ['upstream_0', 'upstream_1', 
                            'merge_0', 'merge_1', 'merge_2', 
                            'downstream_0', 'downstream_1']


# ================ Configure the logging module ====================
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_dir = '_log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file = os.path.join(log_dir, f'{current_time}_optuna_log_{EXP}.txt')
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')




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





def write_vehicle_trajectories_to_csv(readfilename, writefilename):
    # Start SUMO simulation with TraCI
    traci.start(["sumo", "-c", readfilename+".sumocfg"])
    
    # Replace "your_routes_file.rou.xml" with the actual path to your SUMO route file
    route_file_path = readfilename+".rou.xml"
    # Get a list of vehicle IDs from the route file
    predefined_vehicle_ids = get_vehicle_ids_from_routes(route_file_path)

    # Print the list of vehicle IDs
    print("List of Predefined Vehicle IDs:", predefined_vehicle_ids)


    # Open the CSV file for writing
    with open(writefilename, 'w') as csv_file:
        # Write header
        # Column 1:	Vehicle ID
        # Column 2:	Frame ID
        # Column 3:	Lane ID
        # Column 4:	LocalY
        # Column 5:	Mean Speed
        # Column 6:	Mean Acceleration
        # Column 7:	Vehicle length
        # Column 8:	Vehicle Class ID
        # Column 9:	Follower ID
        # Column 10: Leader ID

        csv_file.write("VehicleID, Time, LaneID, LocalY, MeanSpeed, MeanAccel, VehLength, VehClass, FollowerID, LeaderID\n")
        # vehicle_id = "carflow1.131"
        # Run simulation steps
        step = 0
        while traci.simulation.getMinExpectedNumber() > 0:
            # Get simulation time
            simulation_time = traci.simulation.getTime()

            # Get IDs of all vehicles
            vehicle_ids = traci.vehicle.getIDList()

            # Iterate over all vehicles
            for vehicle_id in vehicle_ids:
                # Get vehicle position and speed
                position = traci.vehicle.getPosition(vehicle_id)
                laneid = traci.vehicle.getLaneID(vehicle_id)
                speed = traci.vehicle.getSpeed(vehicle_id)
                accel = traci.vehicle.getAcceleration(vehicle_id)
                cls = traci.vehicle.getVehicleClass(vehicle_id)

                # Write data to the CSV file - similar to NGSIM schema
                csv_file.write(f"{vehicle_id} {simulation_time} {laneid} {position[0]} {speed} {accel} {-1} {cls} {-1} {-1}\n")

            # try to overwite acceleration of one vehicle
            # if 300< step <400:
            #     traci.vehicle.setSpeed(vehicle_id, 0)
            # Simulate one step
            traci.simulationStep()
            step += 1

    # Close connection
    traci.close()
    print("Complete!")

    return



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


def objective(trial):
    """Objective function for optimization."""
    # Define the parameters to be optimized
    driver_param = {
        param_name: trial.suggest_uniform(param_name, min_val[i], max_val[i])
        for i, param_name in enumerate(param_names)
    }
    print(driver_param, trial.number)
    
    # Update SUMO configuration or route files with these parameters
    temp_config_path, temp_path = create_temp_config(driver_param, trial.number)

    # Run SUMO simulation
    # run_sumo(SCENARIO+'.sumocfg')
    run_sumo(temp_config_path)
    
    # Extract simulated traffic volumes
    simulated_output = reader.extract_sim_meas(["trial_"+ location for location in measurement_locations],
                                        file_dir = temp_path)
    
    # Calculate the objective function value
    error = np.linalg.norm(simulated_output[MEAS] - measured_output[MEAS])
    
    clear_directory(os.path.join("temp", str(trial.number)))
    logging.info(f'Trial {trial.number}: param={driver_param}, error={error}')
    
    return error


def logging_callback(study, trial):
    if trial.state == optuna.trial.TrialState.COMPLETE:
        logging.info(f'Trial {trial.number} succeeded: value={trial.value}, params={trial.params}')
    elif trial.state == optuna.trial.TrialState.FAIL:
        logging.error(f'Trial {trial.number} failed: exception={trial.user_attrs.get("exception")}')
    
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

    # ================================= initialize calibration: reset default parameters 
    default_params =  { "maxSpeed": 55.5, "minGap": 2.5, "accel": 2.6, "decel": 4.5, "tau": 1.0, "lcStrategic": 1.0, "lcCooperative": 1.0,"lcAssertive": 1, "lcSpeedGain": 1.0, "lcKeepRight": 1.0, "lcOvertakeRight": 0}
    update_sumo_configuration(default_params)

    # ================================= run ground truth and generate synthetic measurements
    run_sumo(sim_config=SCENARIO+"_gt.sumocfg") #, fcd_output ="trajs_gt.xml")
    measured_output = reader.extract_sim_meas(measurement_locations)


    # =============================== Create a study object and optimize the objective function
    clear_directory("temp")
    sampler = optuna.samplers.TPESampler(seed=10)
    pruner = optuna.pruners.SuccessiveHalvingPruner()
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=100, n_jobs=16, callbacks=[logging_callback])
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.show()
    except:
        pass

    # Get the best parameters and save the optimization result
    best_params = study.best_params
    print('Best parameters:', best_params)
    with open(f'calibration_result/study_{EXP}.pkl', 'wb') as f:
        pickle.dump(study, f)
        print(f"Save optimization result as study_{EXP}.pkl in /calibration_result")

    # # ================================ visualize time-space using best parameters
    # update_sumo_configuration(best_params)
    # run_sumo(sim_config=SCENARIO+".sumocfg")#, fcd_output ="trajs_best.xml")
    # vis.visualize_fcd("trajs_best.xml") # lanes=["E0_0", "E0_1", "E1_0", "E1_1", "E2_0", "E2_1", "E2_2", "E4_0", "E4_1"]
    # sim_output = reader.extract_sim_meas(measurement_locations=["trial_"+ location for location in measurement_locations])
    
     
    # ================================= compare GT meas. vs. simulation with custom params.======================
    # best_params =  {'maxSpeed': 31.44813279984895, 'minGap': 1.8669305739182382, 'accel': 2.2398476082518677, 'decel': 2.5073714738472153, 'tau': 1.3988475504128757, 'lcStrategic': 0.8624217521963465, 'lcCooperative': 0.9789774143646455, 'lcAssertive': 0.43478229746049984, 'lcSpeedGain': 1.1383219615950644, 'lcKeepRight': 4.030227753894549, 'lcOvertakeRight': 0.9240310635518598}
    # update_sumo_configuration(best_params)
    # run_sumo(sim_config = SCENARIO+".sumocfg")
    # vis.plot_sim_vs_sim(sumo_dir, measurement_locations, quantity="speed")
    
    # ============== compute & save macroscopic properties ==================
    # best_params = {'maxSpeed': 30.53284221198521, 'minGap': 2.7958695360441843, 'accel': 2.4497572915690244, 'decel': 2.4293815796265275, 'tau': 1.374376527326827, 'lcStrategic': 1.3368371035725628, 'lcCooperative': 0.9994681517674497, 'lcAssertive': 0.35088886304156547, 'lcSpeedGain': 1.901166989734572, 'lcKeepRight': 0.7531568339763854},

    # update_sumo_configuration(best_params)
    # base_name = SCENARIO+""
    # fcd_name = "fcd_"+base_name+"_"+EXP
    # run_sumo(sim_config = base_name+".sumocfg", fcd_output =fcd_name+".out.xml")
    # reader.fcd_to_csv_byid(xml_file=fcd_name+".out.xml", csv_file=fcd_name+".csv")
    # macro.reorder_by_id(fcd_name+".csv", bylane=False)
    # fcd_name = "fcd_onramp_cflc_v"
    # macro_data = macro.compute_macro(fcd_name+"_byid.csv",  dx=10, dt=10, start_time=0, end_time=480, start_pos =0, end_pos=1300, save=True, plot=True)

    # vis.scatter_fcd(fcd_name+".out.xml")