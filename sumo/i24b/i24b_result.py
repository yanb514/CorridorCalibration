'''
This script evaluates the calibrated results in SUMO
Generates plots and basic benchmarking statistics
'''
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
import i24b_calibrate as i24
import warnings
warnings.filterwarnings("ignore")

# ================ CONFIGURATION ====================
with open('../config.json', 'r') as config_file:
    config = json.load(config_file)

SUMO_EXE = os.getenv("SUMO_PATH", config['SUMO_PATH'])  # Fallback to config

SCENARIO = "i24b"
RDS_DIR = config[SCENARIO]["RDS_DIR"] # directory of the RDS data
SUMO_DIR = os.path.dirname(os.path.abspath(__file__)) # current script directory


best_param_map = { # TODO: read from calibration_result
    'default': config["DEFAULT_PARAMS"],
    "3b": {'maxSpeed': 34.72185955968814, 'minGap': 1.295900993995676, 'accel': 2.3132922370096063, 'decel': 1.8443042297988792, 'tau': 1.681989476492903, 'lcSublane': 8.031266669695496, 'maxSpeedLat': 5.551852173272676, 'lcAccelLat': 1.4899061645158596, 'minGapLat': 2.437640158503997, 'lcStrategic': 6.2630157083167015, 'lcCooperative': 2.490876662995877, 'lcPushy': 0.6017363873188605, 'lcImpatience': 0.37524842223505894, 'lcSpeedGain': 0.1475478299582167},
    }

def run_with_param(parameter, exp_label="", rerun=True, lane_by_lane_macro=False, plot_ts=False, plot_det=False, plot_macro=False):
    '''
    rerun SUMO using provided parameters
    generate FCD and detector data
    ** convert FCD to macro data
    ** save macro data
    '''
    fcd_name = f"fcd_{SCENARIO}_" + exp_label
    folder_path = f'simulation_result/{exp_label}'
    fcd_file = fcd_name+".xml"
    traj_file = fcd_name+"_mainline.csv"
    trajectory_file_name = traj_file.split(".")[0]

    if rerun: # save things in simulation_result/
        print("Running SUMO...")
        i24.update_sumo_configuration(best_param_map['default'])
        i24.update_sumo_configuration(parameter)
        i24.run_sumo(sim_config = f"{SCENARIO}.sumocfg") #, fcd_output =fcd_name+"_full.xml")

        # # filter fcd data with start time and end time
        # # SUMO simulates 4AM-10AM, filter 5-10AM
        # reader.filter_trajectory_data(input_file=fcd_name+"_full.xml", output_file=fcd_name+".xml", 
        #                               start_time=3600, end_time=21600)

        # # Generate trajectories in mainline.csv
        # reader.parse_and_reorder_xml(xml_file=fcd_name+".xml", output_csv=fcd_name+"_mainline.csv", link_names=mainline)

        # # Edie's into macro data
        # macro_data = macro.compute_macro_generalized(fcd_name+"_mainline.csv", dx=DX, dt=DT, start_time=0, end_time=18000, start_pos =0, end_pos=5730,
        #                                 save=True, plot=False) # plot later
        
        # if lane_by_lane_macro:
        #     link_dict = {
        #         "lane1": ["E0_4","E1_5","E3_4","E5_3","E7_4","E8_3"], # left-most lane
        #         "lane2": ["E0_3","E1_4","E3_3","E5_2","E7_3","E8_2"],
        #         "lane3": ["E0_2","E1_3","E3_2","E5_1","E7_2","E8_1"],
        #         "lane4": ["E0_1","E1_2","E3_1","E5_0","E7_1","E8_0"] # right-most lane
        #     }
        #     # Generate lane-specific trajectories and save as {fcd_name}_lane1.csv etc.
        #     reader.parse_and_reorder_xml(xml_file=fcd_name+".xml", output_csv=fcd_name+".csv", link_names=link_dict)

        #     # Edie's into macro data
        #     for key in link_dict:
        #         macro_data = macro.compute_macro_generalized(fcd_name+f"_{key}.csv", dx=DX, dt=DT, start_time=0, end_time=18000, start_pos =0, end_pos=5730,
        #                                 save=True, plot=0) # plot later
        
        # # Move simulated files to simulation_result/EXP_LABEL   
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)
        #     print(f"Created directory: {folder_path}")

        # # move fcd
        # if os.path.exists(fcd_file):
        #     shutil.move(fcd_file, os.path.join(folder_path, fcd_file))

        # move detector outputs
        files_to_move = glob.glob('*.out.xml') + glob.glob(f'*_{exp_label}.csv')
        for file_name in files_to_move:
            if os.path.exists(file_name):
                shutil.move(file_name, os.path.join(folder_path, file_name))

        # # move all csv files that contains {fcd_name}
        # files_to_move = glob.glob(f'*{fcd_name}*.csv')
        # for file_name in files_to_move:
        #     if os.path.exists(file_name):
        #         shutil.move(file_name, os.path.join(folder_path, file_name))

        # # move all .pkl files that contains {fcd_name}
        # files_to_move = glob.glob(f'*{fcd_name}*.pkl')
        # for file_name in files_to_move:
        #     if os.path.exists('calibration_result/'+file_name):
        #         shutil.move('calibration_result/'+file_name, os.path.join(folder_path, file_name))

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
            fig, axes = vis.plot_line_detectors(folder_path, RDS_DIR, det_locations, quantity, fig, axes, exp_label) # read csv files, continuously adding plots to figure
        plt.show()

    # plot macro 3 plots
    # if plot_macro:
    #     # macro_name = rf'simulation_result/{exp_label}/macro_{trajectory_file_name}.pkl'
    #     macro_pkl = rf'simulation_result/{exp_label}/macro_fcd_i24_{exp_label}_mainline.pkl'
    #     with open(macro_pkl, 'rb') as file:
    #         macro_sim = pickle.load(file, encoding='latin1')
    #     macro.plot_macro(macro_sim, dx=DX, dt=DT, hours=5)
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
    print(f"{exp_label} Training RMSPE: ")
    sim_dir = f'simulation_result/{exp_label}'

    # get RDS measurements
    measured_output = reader.rds_to_matrix(rds_file=RDS_DIR, det_locations=measurement_locations)
    # read simulated measurements from det_XXX.out.xml
    sim_output = reader.extract_sim_meas(measurement_locations=[location for location in measurement_locations], file_dir=sim_dir)
    
    # select the same time ranges in measurement and sim 5AM-10AM
    start_idx_rds = 60 #int(5*60/5)
    start_idx_sumo = 12 # sumo starts at 4AM to allow some buffer
    length = 5*12-1 #5hr

    keys = ["volume", "speed", "occupancy"]
    print_labels = ["Volume q: ", "Speed v: ", "Occupancy o: "]

    epsilon = 1e-6  # Small constant to stabilize logarithmic transformation

    for i, key in enumerate(keys):
        sim1_vals = sim_output[key][:, start_idx_sumo:start_idx_sumo+length]
        sim2_vals = measured_output[key][:, start_idx_rds: start_idx_rds+length]
        
        log_sim1 = np.log(sim1_vals + epsilon)
        log_sim2 = np.log(sim2_vals + epsilon)
        
        log_diff = log_sim1 - log_sim2
        error = np.sqrt(np.nanmean((log_diff**2).flatten()))
        print(print_labels[i] + "{:.2f}".format(error))

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
    print(f"{exp_label} Validation RMSPE: ")
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
    for chunk in pd.read_csv(ASM_FILE, chunksize=chunk_size):
        processed_chunk = process_chunk(chunk) # flow: veh/hr/lane
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
    volume_rds = resampled_data.pivot(index='milemarker', columns='unix_time', values='total_volume').values[:n_space, :n_time] # convert from veh/30s/lane to veh/hr/lane
    speed_rds = resampled_data.pivot(index='milemarker', columns='unix_time', values='total_speed').values[:n_space, :n_time]
    density_rds = volume_rds/speed_rds # veh/mile/lane
    volume_rds = np.flipud(volume_rds)

    # visualize for debugging purpose
    # plt.figure(figsize=(6, 6))
    # plt.subplot(1, 2, 1)
    # plt.imshow(density_rds, cmap='viridis', vmin=0) # veh/hr/lane

    # plt.subplot(1, 2, 2)
    # plt.imshow(Rho, cmap='viridis', vmin=0, vmax=80)

    # plt.tight_layout()
    # plt.show()

    keys = ["volume", "speed", "density"]
    print_labels = ["Volume q: ", "Speed v: ", "Density rho: "]
    sims = [Q, V, Rho]
    meas = [volume_rds, speed_rds, density_rds]

    epsilon = 1e-6  # Small constant to stabilize logarithmic transformation

    for i, key in enumerate(keys):
        sim1_vals = sims[i]
        sim2_vals = meas[i]
        log_sim1 = np.log(sim1_vals + epsilon)
        log_sim2 = np.log(sim2_vals + epsilon)
        log_diff = log_sim1 - log_sim2
        error = np.sqrt(np.nanmean((log_diff**2).flatten()))
        print(print_labels[i] + "{:.2f}".format(error))

    return



if __name__ == "__main__":

    # ================================= prepare RDS data for model calibration
    det_locations = i24.extract_detector_locations(RDS_DIR)
    det_locations = [det_loc for det_loc in det_locations if "westbound" in det_loc] # filter westbound only

    measured_output = reader.rds_to_matrix_i24b(rds_file=RDS_DIR, det_locations=det_locations)

    # ===== rerun and save data ============= 
    for EXP in ["3b"]:
        run_with_param(best_param_map[EXP], exp_label=EXP, rerun=1, lane_by_lane_macro=0,plot_ts=0, plot_det=0, plot_macro=0)
        training_rmspe(EXP)
        validation_rmspe(EXP)
        print("\n")

    # ===== plot detector line plot ============= 
    save_path = "det_sim.png"
    fig = None
    axes = None
    quantity = "speed"
    experiments = ["RDS", "3b"]
    for exp_label in experiments:
        folder_path = f'simulation_result/{exp_label}'
        fig, axes = vis.plot_line_detectors(folder_path, RDS_DIR, det_locations, quantity, fig, axes, exp_label) # read csv files, continuously adding plots to figure
    # save
    fig.savefig(save_path, dpi=300, bbox_inches='tight')  # Save with high resolution
    # plt.show()

    # ===== plot 9-grid plot ============= 
    # quantity = "speed"
    # save_path = r'C:\Users\yanbing.wang\Documents\CorridorCalibration\figures\TRC-i24\grid_speed.png'
    # fig=None
    # axes=None
    # for i,exp_label in enumerate(["1a","1b", "1c","2a","2b","2c","3a","3b","3c"]):
    #     macro_pkl = rf'simulation_result/{exp_label}/macro_fcd_i24_{exp_label}_mainline.pkl'
    #     with open(macro_pkl, 'rb') as file:
    #         macro_data = pickle.load(file, encoding='latin1')
    #     fig, axes = vis.plot_macro_grid(macro_data, 
    #                         quantity, 
    #                         dx=160.934, dt=30, 
    #                         fig=fig, axes=axes,
    #                         ax_idx=i, label=exp_label)
    # fig.savefig(save_path, dpi=300, bbox_inches='tight') 
    # plt.show()

    # ======= plot ASM RDS data ===========
    # vis.read_asm(ASM_FILE)
    # plt.show()

    # ======= travel time lane-specific =====
    # fig = None
    # ax = None
    # for i,exp_label in enumerate(["rds"]):
    #     fig, ax = vis.plot_travel_time(fig, ax, exp_label)
    #     ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
    #     ax.xaxis.set_major_locator(plt.matplotlib.dates.HourLocator(interval=1))
    #     ax.set_ylim([0,1550])
    #     ax.legend(loc='upper right', fontsize=16)
    #     ax.set_xlabel("Departure time")
    #     ax.set_ylabel("Travel time (sec)")
    # plt.tight_layout(rect=[0, 0, 1, 1])
    # plt.show()

    # ======= travel time lane-specific 9 grid =====
    # fig = None
    # axes = None
    # save_path = r'C:\Users\yanbing.wang\Documents\CorridorCalibration\figures\TRC-i24\grid_travel_time.png'
    # for i,exp_label in enumerate(["1a","1b", "1c","2a","2b","2c","3a","3b","3c"]):
    #     if exp_label in ["RDS", "rds"]:
    #         macro_data = None # read from ASM file in the function
    #     else:
    #         fig, axes = vis.plot_travel_time_grid(fig, axes, i, exp_label)
    # fig.savefig(save_path, dpi=300, bbox_inches='tight') 
    # plt.show()
