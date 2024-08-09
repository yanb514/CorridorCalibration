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
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import json

main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) # two levels up
sys.path.insert(0, main_path)
import utils_vis as vis
import utils_data_read as reader
import i24_calibrate_de as i24
import macro


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

measurement_locations = [
                        # '56_7_0', '56_7_1', '56_7_2', '56_7_3', '56_7_4', 
                         '56_3_0', '56_3_1', '56_3_2', '56_3_3', #'56_3_4',
                         '56_0_0', '56_0_1', '56_0_2', '56_0_3', #'56_0_4',
                         '55_3_0', '55_3_1', '55_3_2', '55_3_3',
                         '54_6_0', '54_6_1', '54_6_2', '54_6_3',
                         '54_1_0', '54_1_1', '54_1_2', '54_1_3' ]

best_param_map = {
    'default': {'maxSpeed': 34.91628705652602,
                    'minGap': 2.9288888706657783,
                    'accel': 1.0031145478483796,
                    'decel': 2.9618821510422406,
                    'tau': 1.3051261247487569,
                    'lcStrategic': 1.414,
                    'lcCooperative': 1.0,
                    'lcAssertive': 1.0,
                    'lcSpeedGain': 3.76,
                    'lcKeepRight': 0.0,
                    'lcOvertakeRight': 0.877},
    "1a": {'maxSpeed': 38.27084380173686, 'minGap': 1.3161582136286667, 'accel': 3.991592675681226, 'decel': 1.8080341736427255, 'tau': 1.0066512077261067},
    "1b": {'maxSpeed': 31.534820558874827, 'minGap': 1.860096631767026, 'accel': 1.0708978903827724, 'decel': 3.8918676775882215, 'tau': 1.7949543267839752},
    "1c": {'maxSpeed': 31.101934010592, 'minGap': 0.6280180394989212, 'accel': 1.6756620591108398, 'decel': 1.001024011369078, 'tau': 2.589065833625148},
    "2a": {'lcStrategic': 0.8488669891308404, 'lcCooperative': 0.9095876997465676, 'lcAssertive': 4.84304250702997, 'lcSpeedGain': 1.9154077824532103},
    "2b": {'lcStrategic': 0.0034590132917030228, 'lcCooperative': 0.3142842765112314, 'lcAssertive': 2.9067266300117014, 'lcSpeedGain': 3.607046567855465},
    "2c": {'lcStrategic': 0.004112939077306229, 'lcCooperative': 0.0595973542386749, 'lcAssertive': 2.808169034162054, 'lcSpeedGain': 2.8840926392852104},
    "3a":  {'maxSpeed': 28.084544621059674, 'minGap': 1.2048496168998633, 'accel': 3.9515854431696535, 'decel': 1.0029304345650172, 'tau': 1.1454276680332822, 'lcStrategic': 3.5550037767428475, 'lcCooperative': 0.7586428298068486, 'lcAssertive': 4.574845459240965, 'lcSpeedGain': 4.378730584964673},
    "3b": {'maxSpeed': 41.23472014948679, 'minGap': 2.3231364774022953, 'accel': 1.778106163419937, 'decel': 3.8816489744532974, 'tau': 1.8591112580311417, 'lcStrategic': 0.0883885681423903, 'lcCooperative': 0.34206423150441595, 'lcAssertive': 2.712768298941477, 'lcSpeedGain': 1.0397729328884315},
    "3c":{'maxSpeed': 27.192848569516737, 'minGap': 0.600389480872267, 'accel': 1.2318358355858519, 'decel': 1.0020979338277094, 'tau': 2.047389930933602, 'lcStrategic': 2.4633036268216455, 'lcCooperative': 0.8815112972395069, 'lcAssertive': 4.59649247215181, 'lcSpeedGain': 3.987299355366019}
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
    
    hours = 3
    length = int(hours * 3600/dt) #360

    # simulated data
    Q, Rho, V = macro_data["flow"][:length,:], macro_data["density"][:length,:], macro_data["speed"][:length,:]
    Q = Q.T * 3600/4 # veh/hr/lane
    V = V.T * 2.23694 # mph
    Rho = Rho.T
    n_space, n_time = Q.shape
    size = Q.size
    V = np.flipud(V)
    Rho = np.flipud(Rho)


    # Initialize an empty DataFrame to store the aggregated results
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
    
    end_time = start_time + 3*3600 # only select the first 3 hours

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


if __name__ == "__main__":

    
    EXP = "3c"


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
    
    # ================== rerun with new parameters
    i24.run_sumo(sim_config=SCENARIO+".sumocfg")

    # ================================ visualize time-space using best parameters
    asm_file =  os.path.join("../..", "data/2023-11-13-ASM.csv")

    for EXP in rerun_labels: #["1a","1b", "1c","2a","2b","2c","3a","3b","3c"]:
        i24.update_sumo_configuration(best_param_map['default'])
        best_params = best_param_map[EXP]
        i24.update_sumo_configuration(best_params)
        base_name = SCENARIO+""
        fcd_name = "fcd_"+base_name+"_"+EXP
        
        # ============ rerun simulation if necessary
        i24.run_sumo(sim_config = base_name+".sumocfg", fcd_output =fcd_name+".out.xml")
        for meas in measurement_locations:
            reader.det_to_csv(xml_file=f"det_{meas}.out.xml", suffix="_"+EXP)

        reader.fcd_to_csv_byid(xml_file=fcd_name+".out.xml", csv_file=fcd_name+".csv")
        macro.reorder_by_id(fcd_name+".csv", link_names=mainline, lane_name="mainline")
        macro_data = macro.compute_macro(fcd_name+"_mainline.csv", dx=160.934, dt=30, start_time=0, end_time=10801, start_pos =0, end_pos=5730,
                                        save=True, plot=True)


    # ============ plot flow, lane-specific, detector location
    fig = None
    axes = None
    # quantity = "flow"
    # experiments = ["RDS", "1a", "2a", "3a"]
    # quantity = "density"
    # experiments = ["RDS", "1c", "2c", "3c"]
    quantity = "speed"
    experiments = ["RDS", "1b", "2b", "3b"]
    for exp_label in experiments:
        param = best_param_map[exp_label]
        i24.update_sumo_configuration(best_param_map["default"])
        i24.update_sumo_configuration(param)
        i24.run_sumo(sim_config = "onramp.sumocfg")
        if exp_label == "RDS":
            path = RDS_DIR
        else:
            path = SUMO_DIR
        fig, axes = vis.plot_line_detectors(path, measurement_locations, quantity, fig, axes, exp_label) # continuously adding plots to figure
    plt.savefig(rf'..\..\figures\i24_detector_{quantity}.png')
    plt.show()

    # ============ plot time-space macroscopic grid 3x3 ===============
    quantity = "speed"
    for i, exp_label in enumerate(["1a", "1b", "1c","2a","2b","2c","3a","3b","3c"]):
        macro_pkl = rf'macro_fcd_I24_scenario_{exp_label}_mainline.pkl'
        try:
            with open(macro_pkl, 'rb') as file:
                macro_sim = pickle.load(file)
            
            fig, axes = vis.plot_macro_grid(macro_sim, quantity, dx=160.934, dt=30, fig=fig, axes=axes, ax_idx=i, label=exp_label)
            # if exp_label in rerun_labels:
            #     print(exp_label)
            #     detector_rmse(exp_label)
            #     macro_rmse(asm_file, macro_sim)
        except FileNotFoundError:
            print("no file: ", macro_pkl)
            pass
    plt.savefig(rf'..\..\figures\macro_i24_{quantity}.png')
    plt.show()

    # vis.read_asm(asm_file)
    # vis.plot_rds_vs_sim(RDS_DIR, SUMO_DIR, measurement_locations, quantity="occupancy")