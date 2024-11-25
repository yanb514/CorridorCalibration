"""
This file generates the training and validation results from macrosopic data stored in .pkl
"""
import pickle
import numpy as np
import os
import os
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import csv
import json
import shutil
import glob

main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) # two levels up
sys.path.insert(0, main_path)
import utils_vis as vis
import utils_data_read as reader
import utils_macro as macro
import i24_calibrate as i24



SCENARIO = "I24_scenario"
with open('../config.json', 'r') as config_file:
    config = json.load(config_file)

computer_name = os.environ.get('COMPUTERNAME', 'Unknown')
if "CSI" in computer_name:
    SUMO_EXE = config['SUMO_EXE']
    
    
elif "VMS" in computer_name:
    SUMO_EXE = config['SUMO_EXE_PATH']

RDS_DIR = os.path.join("../..", "data/RDS/I24_WB_52_60_11132023.csv")
SUMO_DIR = os.path.dirname(os.path.abspath(__file__)) # current script directory
asm_file =  os.path.join("../..", "data/2023-11-13-ASM.csv")

measurement_locations = [
                         '56_7_0', '56_7_1', '56_7_2', '56_7_3', #'56_7_4', 
                         '56_3_0', '56_3_1', '56_3_2', '56_3_3', #'56_3_4',
                         '56_0_0', '56_0_1', '56_0_2', '56_0_3', #'56_0_4',
                         '55_3_0', '55_3_1', '55_3_2', '55_3_3',
                         '54_6_0', '54_6_1', '54_6_2', '54_6_3',
                        #  '54_1_0', '54_1_1', '54_1_2', '54_1_3' 
                         ]

best_param_map = {
    'default': {'maxSpeed': 34.91628705652602,'minGap': 2.9288888706657783,'accel': 1.0031145478483796, 'decel': 2.9618821510422406,'tau': 1.3051261247487569,'lcStrategic': 1.414, 'lcCooperative': 1.0,'lcAssertive': 1.0,'lcSpeedGain': 3.76,'lcKeepRight': 0.0,'lcOvertakeRight': 0.877},
    "1a": {'maxSpeed': 40.73100503684973, 'minGap': 0.7554695612797295, 'accel': 1.5138893454833222, 'decel': 3.1736271864613324, 'tau': 1.758647213750675},
    "1b": {'maxSpeed': 25.285732074182814, 'minGap': 1.0030275914354692, 'accel': 1.0523656379385362, 'decel': 3.9018744110714643, 'tau': 2.93682397740524},
    "1c": {'maxSpeed': 42.367511096833326, 'minGap': 2.9993342836983476, 'accel': 1.9468168020463612, 'decel': 1.0061776541678609, 'tau': 2.9998938201490217},
    "2a": {'lcStrategic': 4.13327296799944, 'lcCooperative': 0.18415903165741201, 'lcAssertive': 2.8928435373483725, 'lcSpeedGain': 2.234576625106279},
    "2b": {'lcStrategic': 2.188441136054435, 'lcCooperative': 0.08301719911828441, 'lcAssertive': 4.5274083995909224, 'lcSpeedGain': 4.8748255095752695},
    "2c": {'lcStrategic': 3.6399880110545846, 'lcCooperative': 0.7193235256529892, 'lcAssertive': 0.000705838535847945, 'lcSpeedGain': 1.1101360440720436},
    "3a": {'maxSpeed': 41.050542277462505, 'minGap': 1.0778474458689624, 'accel': 2.9361340135234197, 'decel': 1.7060856568918261, 'tau': 1.801505055872516, 'lcStrategic': 1.5691496860555882, 'lcCooperative': 0.009112636537354599, 'lcAssertive': 4.547730711265782, 'lcSpeedGain': 1.3161135569989806},
    # "3b": {'maxSpeed': 30.394124775941144, 'minGap': 1.1135192000015561, 'accel': 1.0488617301700136, 'decel': 3.894432099649597, 'tau': 2.5172146404021585, 'lcStrategic': 4.765478517631743, 'lcCooperative': 0.6036708111679133, 'lcAssertive': 4.85055235578156, 'lcSpeedGain': 4.421559372617682}, # 5hr
    # "3b": {'maxSpeed': 32.628733908202186, 'minGap': 1.341402259779661, 'accel': 3.6534628045970376, 'decel': 2.161073526928775, 'tau': 1.4679912202190781, 'lcStrategic': 0.09167743180863308, 'lcCooperative': 0.5573275937293471, 'lcAssertive': 2.6157238740399222, 'lcSpeedGain': 1.5338412219673294}, # with buffer
    "3b": {'maxSpeed': 35.46631186967486, 'minGap': 2.94390525189298, 'accel': 2.3655452248462066, 'decel': 3.207636533267079, 'tau': 1.6603715426741472, 'lcStrategic': 0.5091719209419492, 'lcCooperative': 0.8854467728463593, 'lcAssertive': 4.815590691539943, 'lcSpeedGain': 1.1639111046014528},
    "3c": {'maxSpeed': 41.67790488305487, 'minGap': 2.944472588322073, 'accel': 1.0413394249871004, 'decel': 1.0015581105726545, 'tau': 2.8756688917028996, 'lcStrategic': 3.203048947163652, 'lcCooperative': 0.6794410211168292, 'lcAssertive': 0.11095047910487453, 'lcSpeedGain': 3.1717237808418135}
}


rerun_labels = ["1c", "3c"]

mainline = ["E0_1", "E0_2", "E0_3", "E0_4",
            "E1_2", "E1_3", "E1_4", "E1_5",
            "E3_1", "E3_2", "E3_3", "E3_4",
            "E5_0", "E5_1", "E5_2", "E5_3",
            "E7_1", "E7_2", "E7_3", "E7_4",
            "E8_0", "E8_1", "E8_2", "E8_3"
            ] # since ASM is only processed on lane 1-4 (SUMO reversed lane idx)


def detector_rmse(exp_label):

    '''
    sumo_dir: directory for DETECTOR.out.xml files
    measurement_locations: a list of detectors
    quantity: "volume", "speed" or "occupancy"
    '''

    # Read and extract data
    print("Training RMSE (detectors)")
    measured_output = reader.rds_to_matrix(rds_file=RDS_DIR, det_locations=measurement_locations)


    column_names = ["flow", "speed"]
    simulated_output = {column_name: [] for column_name in column_names}
    for meas in measurement_locations:
        flow_arr = []
        speed_arr = []
        filename = os.path.join(SUMO_DIR, f"det_{meas}_{exp_label}.csv")
        with open(filename, mode='r') as file:
            csvreader = csv.DictReader(file)
            
            for row in csvreader:
                # for column_name in column_names:
                    # data_dict[column_name].append(float(row[column_name]))
                flow_arr.append(float(row["flow"]))
                speed_arr.append(float(row["speed"]))
        simulated_output['flow'].append(flow_arr)
        simulated_output['speed'].append(speed_arr)

    for key in simulated_output.keys():
        simulated_output[key] = np.array(simulated_output[key]) #n_det x n_time

    simulated_output['speed']*=  2.23694 # to mph
    simulated_output['density'] = simulated_output['flow'] / simulated_output['speed']

    # Align time
    # TODO: SIMULATED_OUTPUT starts at 5AM-8AM, while measured_output is 0-24, both in 5min intervals
    start_idx = 60 #int(5*60/5)
    end_idx = min(simulated_output["speed"].shape[1], 36)
    end_idx_rds = start_idx + end_idx # at most three hours of simulated measurements
    
    # Calculate the objective function value
    diff = simulated_output["flow"][:,:end_idx] - measured_output["flow"][:, start_idx: end_idx_rds] # measured output may have nans
    error = np.sqrt(np.nanmean(diff.flatten()**2))
    print("Volume q (nveh/hr): {:.2f}".format(error))  #veh/hr


    diff = simulated_output["speed"][:,:end_idx] - measured_output["speed"][:, start_idx: end_idx_rds] # measured output may have nans
    error = np.sqrt(np.nanmean(diff.flatten()**2))
    print("Speed v (mph): {:.2f} ".format(error)) # mph

    diff = simulated_output["density"][:,:end_idx] - measured_output["density"][:, start_idx: end_idx_rds] # measured output may have nans
    error = np.sqrt(np.nanmean(diff.flatten()**2))
    print("Density rho: {:.2f} veh/mile/lane".format(error))
    return

def macro_rmse(asm_file, macro_data):
    '''
    Compare simulated macro with RDS AMS in the selected temporal-spatial range
    asm is dx=0.1 mi, dt=10 sec
    macro_data units (Edie's def):
        Q: veh/sec
        V: m/s
        Rho: veh/m
    ASM RDS unit 
        Q: veh/30 sec
        V: mph
        Rho: veh/(0.1mile)
    Final unit:
        Q: veh/hr/lane
        V: mph
        Rho: -
    '''
    dx =160.934
    dt =30
    
    hours = 5
    length = int(hours * 3600/dt)-1 #360

    # simulated data
    Q, Rho, V = macro_data["flow"][:length,:], macro_data["density"][:length,:], macro_data["speed"][:length,:]
    Q = Q.T * 3600/4 # veh/hr/lane
    V = V.T * 2.23694 # mph
    Rho = Rho.T
    n_space, n_time = Q.shape
    print(n_space, n_time)
    V = np.flipud(V)
    Rho = np.flipud(Rho)


    # Initialize an empty DataFrame to store the aggregated ASM data
    aggregated_data = pd.DataFrame()

    # Define a function to process each chunk
    def process_chunk(chunk):
        # Calculate aggregated volume, occupancy, and speed for each row
        chunk['total_volume'] = chunk[['lane1_volume', 'lane2_volume', 'lane3_volume', 'lane4_volume']].mean(axis=1)*120 # convert from veh/30s to veh/hr
        chunk['total_occ'] = chunk[['lane1_occ',  'lane2_occ','lane3_occ',  'lane4_occ']].mean(axis=1)
        chunk['total_speed'] = chunk[['lane1_speed',  'lane2_speed', 'lane3_speed','lane4_speed']].mean(axis=1)
        return chunk[['unix_time', 'milemarker', 'total_volume', 'total_occ', 'total_speed']]

    # Read the CSV file in chunks and process each chunk
    chunk_size = 10000  # Adjust the chunk size based on your memory capacity
    for chunk in pd.read_csv(asm_file, chunksize=chunk_size):
        processed_chunk = process_chunk(chunk)
        aggregated_data = pd.concat([aggregated_data, processed_chunk], ignore_index=True)

    # Define the range of mile markers to plot
    milemarker_min = 54.1
    milemarker_max = 57.6
    start_time = aggregated_data['unix_time'].min()+3600 # data starts at 4AM CST, but we want to start at 5AM
    end_time = start_time + hours*3600 # only select the first x hours

    # Filter milemarker within the specified range
    filtered_data = aggregated_data[
        (aggregated_data['milemarker'] >= milemarker_min) &
        (aggregated_data['milemarker'] <= milemarker_max) &
        (aggregated_data['unix_time'] >= start_time) &
        (aggregated_data['unix_time'] <= end_time)
    ]
    # Convert unix_time to datetime if needed and extract hour (UTC to Central standard time in winter)
    filtered_data['unix_time'] = pd.to_datetime(filtered_data['unix_time'], unit='s') - pd.Timedelta(hours=6)
    filtered_data.set_index('unix_time', inplace=True)
    resampled_data = filtered_data.groupby(['milemarker', pd.Grouper(freq='30s')]).agg({
        'total_volume': 'mean',     # Sum for total volume (veh/30sec)
        'total_speed': 'mean'      # Mean for total speed
    }).reset_index()

    # Pivot the data for heatmaps
    volume_pivot = resampled_data.pivot(index='milemarker', columns='unix_time', values='total_volume').values[:n_space, :n_time] # convert from veh/30s/lane to veh/hr/lane
    speed_pivot = resampled_data.pivot(index='milemarker', columns='unix_time', values='total_speed').values[:n_space, :n_time]
    density_pivot = volume_pivot/speed_pivot # veh/mile/lane

    volume_pivot = np.flipud(volume_pivot)


    # OCC = Rho * 5 *100

    # visualize for debugging purpose
    # plt.figure(figsize=(13, 6))
    # plt.subplot(1, 2, 1)
    # sns.heatmap(density_pivot, cmap='viridis', vmin=0) # veh/hr/lane

    # plt.subplot(1, 2, 2)
    # sns.heatmap(Rho, cmap='viridis', vmin=0)

    # plt.tight_layout()
    # plt.show()

   
    print("Validation RMSE (macro simulation data)")
    diff = volume_pivot - Q
    norm = np.sqrt(np.nanmean(diff.flatten()**2))
    print("Volume q: {:.2f} veh/hr/lane".format(norm))  
          
    diff = speed_pivot - V
    norm = np.sqrt(np.nanmean(diff.flatten()**2))
    print("Speed v: {:.2f} mph".format(norm))

    diff = density_pivot - Rho
    norm = np.sqrt(np.nanmean(diff.flatten()**2))
    print("Density rho: {:.2f} veh/mile/lane".format(norm))


    return

def training_rmspe(exp_label):

    '''
    Compare simulated detector meas with RDS
    sumo_dir: directory for DETECTOR.out.xml files
    measurement_locations: a list of detectors
    quantity: "volume", "speed" or "occupancy"
    compute RMSPE: the scale doesn't matter
    '''

    # Read and extract data
    print("Training RMSPE (detectors)")
    sim_dir = f'simulation_result/{exp_label}'

    # get RDS measurements
    measured_output = reader.rds_to_matrix(rds_file=RDS_DIR, det_locations=measurement_locations)
    # read simulated measurements from det_XXX.out.xml
    sim_output = reader.extract_sim_meas(measurement_locations=[location for location in measurement_locations], file_dir=sim_dir)
    # select the same time ranges in measurement and sim 5AM-10AM
    start_idx = 60 #int(5*60/5)
    end_idx = min(sim_output["speed"].shape[1], 5*12)-1 # the last time interval of the simulated measurements are usually inaccurate (boundary issue)
    end_idx_rds = start_idx + end_idx # at most three hours of simulated measurements

    keys = ["volume", "speed", "occupancy"]
    print_labels = ["Volume q: ", "Speed v: ", "Occupancy o: "]

    for i, key in enumerate(keys):
        diff = sim_output[key][:, 2:end_idx] - np.nan_to_num(measured_output[key][:, start_idx+2:end_idx_rds], nan=0)
        denom = measured_output[key][:, start_idx+2:end_idx_rds]
        relative_diff = np.divide(diff, denom, out=np.full_like(diff, np.nan), where=denom != 0) # Avoid division by zero: set entries in relative_diff to np.nan where denom is zero
        error = np.sqrt(np.nanmean((relative_diff**2).flatten()))
        print(print_labels[i]+"{:.2f}".format(error))

    return

def validation_rmspe(exp_label):
    '''
    Compare simulated macro with RDS AMS in the selected temporal-spatial range
    asm is dx=0.1 mi, dt=10 sec, but flow is aggregated at 30sec (veh/30s), speed (mph), occupancy (%)
    macro_data units (Edie's def):
        Q: veh/sec
        V: m/s
        Rho: veh/m
    ASM RDS unit 
        Q: veh/30 sec
        V: mph
        Rho: veh/(0.1mile)
    Final unit:
        Q: veh/hr/lane
        V: mph
        Rho: -
    '''
    dt = 30
    hours = 5
    length = int(hours * 3600/dt)-1 #360

    macro_pkl = rf'simulation_result/{exp_label}/macro_fcd_i24_{exp_label}_mainline.pkl'
    try:
        with open(macro_pkl, 'rb') as file:
            macro_data = pickle.load(file)
    except FileNotFoundError:
        print("no file: ", macro_pkl)
        pass

    # simulated data
    Q, Rho, V = macro_data["flow"][:length,:], macro_data["density"][:length,:], macro_data["speed"][:length,:]# , macro_data["occupancy"][:length,:]
    Q = Q.T * 3600/4 # veh/sec -> veh/hr/lane
    V = V.T * 2.23694 # m/s -> mph
    # O = O.T
    Rho = Rho.T * 1609/4 # veh/m -> veh/mile/lane
    n_space, n_time = Q.shape
    V = np.flipud(V)
    Rho = np.flipud(Rho)

    # Initialize an empty DataFrame to store the aggregated ASM data
    aggregated_data = pd.DataFrame()

    # Define a function to process each chunk
    def process_chunk(chunk):
        # Calculate aggregated volume, occupancy, and speed for each row
        chunk['total_volume'] = chunk[['lane1_volume', 'lane2_volume', 'lane3_volume', 'lane4_volume']].mean(axis=1)*120 # convert from veh/10s to veh/hr/lane (ASM data averaged at 30sec intervals but sampled at 10s )
        chunk['total_occ'] = chunk[['lane1_occ',  'lane2_occ','lane3_occ',  'lane4_occ']].mean(axis=1)
        chunk['total_speed'] = chunk[['lane1_speed',  'lane2_speed', 'lane3_speed','lane4_speed']].mean(axis=1)
        return chunk[['unix_time', 'milemarker', 'total_volume', 'total_occ', 'total_speed']]

    # Read the CSV file in chunks and process each chunk
    chunk_size = 10000  # Adjust the chunk size based on your memory capacity
    # flag = 0
    for chunk in pd.read_csv(asm_file, chunksize=chunk_size):
        processed_chunk = process_chunk(chunk) # flow: veh/hr/lane
        # if flag == 0:
        #     print(processed_chunk["unix_time"][1000:1050]-1.699870e+09)
        #     flag = 1
        aggregated_data = pd.concat([aggregated_data, processed_chunk], ignore_index=True)

    # Define the range of mile markers to plot
    milemarker_min = 54.1
    milemarker_max = 57.6
    start_time = aggregated_data['unix_time'].min()+3600 # data starts at 4AM CST, but we want to start at 5AM
    end_time = start_time + hours*3600 # only select the first x hours

    # Filter milemarker within the specified range
    filtered_data = aggregated_data[
        (aggregated_data['milemarker'] >= milemarker_min) &
        (aggregated_data['milemarker'] <= milemarker_max) &
        (aggregated_data['unix_time'] >= start_time) &
        (aggregated_data['unix_time'] <= end_time)
    ]
    # Convert unix_time to datetime if needed and extract hour (UTC to Central standard time in winter)
    filtered_data['unix_time'] = pd.to_datetime(filtered_data['unix_time'], unit='s') - pd.Timedelta(hours=6)
    filtered_data.set_index('unix_time', inplace=True) # filtered_data is every 10sec, flow:
    # print(filtered_data.head(40))
    resampled_data = filtered_data.groupby(['milemarker', pd.Grouper(freq='30s')]).agg({
        'total_volume': 'mean',     # Sum for total volume (veh/30sec)
        'total_speed': 'mean'      # Mean for total speed
    }).reset_index()
    # print(resampled_data.head())

    # Pivot the data for heatmaps
    volume_pivot = resampled_data.pivot(index='milemarker', columns='unix_time', values='total_volume').values[:n_space, :n_time] # convert from veh/30s/lane to veh/hr/lane
    speed_pivot = resampled_data.pivot(index='milemarker', columns='unix_time', values='total_speed').values[:n_space, :n_time]
    density_pivot = volume_pivot/speed_pivot # veh/mile/lane
    volume_pivot = np.flipud(volume_pivot)


    # OCC = Rho * 5 *100

    # visualize for debugging purpose
    # plt.figure(figsize=(13, 6))
    # plt.subplot(1, 2, 1)
    # sns.heatmap(speed_pivot, cmap='viridis', vmin=0) # veh/hr/lane

    # plt.subplot(1, 2, 2)
    # sns.heatmap(V, cmap='viridis', vmin=0, vmax=80)

    # plt.tight_layout()
    # plt.show()
    # print(speed_pivot, V)
    # print(density_pivot, Rho)
    # print(volume_pivot, Q)


   
    # print("Validation RMSE (macro simulation data)")
    # diff = volume_pivot - Q
    # norm = np.sqrt(np.nanmean(diff.flatten()**2))
    # print("Volume q: {:.2f} veh/hr/lane".format(norm))  
          
    # diff = speed_pivot - V
    # norm = np.sqrt(np.nanmean(diff.flatten()**2))
    # print("Speed v: {:.2f} mph".format(norm))

    # diff = density_pivot - Rho
    # norm = np.sqrt(np.nanmean(diff.flatten()**2))
    # print("Density rho: {:.2f} veh/mile/lane".format(norm))

    keys = ["volume", "speed", "density"]
    print_labels = ["Volume q: ", "Speed v: ", "Density rho: "]
    sims = [Q, V, Rho]
    meas = [volume_pivot, speed_pivot, density_pivot]

    for i, key in enumerate(keys):
        diff = sims[i] - np.nan_to_num(meas[i], nan=0)
        denom = meas[i]
        relative_diff = np.divide(diff, denom, out=np.full_like(diff, np.nan), where=denom != 0) # Avoid division by zero: set entries in relative_diff to np.nan where denom is zero
        error = np.sqrt(np.nanmean((relative_diff**2).flatten()))
        # print(key, meas[i])
        print(print_labels[i]+"{:.2f}".format(error))

    return

def run_with_param(parameter, exp_label="", rerun=True, plot_ts=False, plot_det=False, plot_macro=False):
    '''
    rerun SUMO using provided parameters
    generate FCD and detector data
    convert FCD to macro data
    save macro data
    '''
    fcd_name = "fcd_i24_" + exp_label
    folder_path = f'simulation_result/{exp_label}'
    fcd_file = fcd_name+".xml"
    traj_file = fcd_name+"_mainline.csv"
    trajectory_file_name = traj_file.split(".")[0]

    if rerun: # save things in simulation_result/
        i24.update_sumo_configuration(best_param_map['default'])
        i24.update_sumo_configuration(parameter)
        i24.run_sumo(sim_config = "I24_scenario.sumocfg", fcd_output =fcd_name+"_full.xml")

        # filter fcd data with start time and end time
        # SUMO simulates 4AM-10AM, filter 5-10AM
        reader.filter_trajectory_data(input_file=fcd_name+"_full.xml", output_file=fcd_name+".xml", 
                                      start_time=3600, end_time=21600)

        # read detector outputs
        # for meas in measurement_locations:
        #     reader.det_to_csv(xml_file=f"det_{meas}.out.xml", suffix="_"+EXP) 

        # Generate trajectories in mainline.csv
        reader.parse_and_reorder_xml(xml_file=fcd_name+".xml", output_csv=fcd_name+"_mainline.csv", link_names=mainline)

        # Edie's into macro data
        macro_data = macro.compute_macro_generalized(fcd_name+"_mainline.csv", dx=160.934, dt=30, start_time=0, end_time=18000, start_pos =0, end_pos=5730,
                                        save=True, plot=False) # plot later
        
        # Move simulated files to simulation_result/EXP_LABEL   
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created directory: {folder_path}")

        # move fcd
        if os.path.exists(fcd_file):
            shutil.move(fcd_file, os.path.join(folder_path, fcd_file))

        # move detector outputs
        files_to_move = glob.glob('*.out.xml') + glob.glob(f'*_{exp_label}.csv')
        for file_name in files_to_move:
            if os.path.exists(file_name):
                shutil.move(file_name, os.path.join(folder_path, file_name))

        # move fcd_name+"_mainline.csv"
        if os.path.exists(traj_file):
            shutil.move(traj_file, os.path.join(folder_path, traj_file))

        # save macro
        macro_name = f'macro_fcd_i24_{exp_label}_mainline.pkl'
        # if os.path.exists('calibration_result/'+macro_name):
        shutil.move('calibration_result/'+macro_name, os.path.join(folder_path, macro_name))
        
    # plot time-space diagram for this simulation
    if plot_ts:
        vis.visualize_fcd(folder_path+"/"+fcd_name+".xml") #, lanes=mainline)  # plot mainline only

    # plot RDS and simulated detector measurements on the same plots
    if plot_det:
        fig = None
        axes = None
        quantity = "speed"
        experiments = ["RDS", exp_label]
        for exp_label in experiments:
            fig, axes = vis.plot_line_detectors(folder_path, RDS_DIR, measurement_locations, quantity, fig, axes, exp_label) # read csv files, continuously adding plots to figure
        plt.show()

    # plot macro 3 plots
    if plot_macro:
        # macro_name = rf'simulation_result/{exp_label}/macro_{trajectory_file_name}.pkl'
        macro_pkl = rf'simulation_result/{exp_label}/macro_fcd_i24_{exp_label}_mainline.pkl'
        with open(macro_pkl, 'rb') as file:
            macro_sim = pickle.load(file, encoding='latin1')
        macro.plot_macro(macro_sim, dx=160.934, dt=30, hours=5)
    return

if __name__ == "__main__":

    
    # EXP = "1a"
    for EXP in ["3b"]: #["1a","1b", "1c","2a","2b","2c","3a","3b","3c"]:
        # reset parameters
        # i24.update_sumo_configuration(best_param_map['default'])
        # i24.update_sumo_configuration(best_param_map[EXP])
        run_with_param(best_param_map[EXP], exp_label=EXP, rerun=True, plot_ts=False, plot_det=0, plot_macro=True)
        # training_rmspe(EXP)
        # validation_rmspe(EXP)

    # vis.read_asm(asm_file)
