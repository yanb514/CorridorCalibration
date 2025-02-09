"""
This file generates the training and validation results from macrosopic data stored in .pkl
"""
import pickle
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) # two levels up
sys.path.insert(0, main_path)
import utils_vis as vis
import utils_data_read as reader
import onramp_calibrate as onramp
import utils_macro as macro
from collections import defaultdict
import xml.etree.ElementTree as ET


pre = "macro_fcd_onramp"

best_param_map = {
    "gt": { "maxSpeed": 30.55, "minGap": 2.5, "accel": 1.5, "decel": 2, "tau": 1.4, "lcStrategic": 1.0, "lcCooperative": 1.0,"lcAssertive": 0.5, "lcSpeedGain": 1.0, "lcKeepRight": 0.5},
    "default":  { "maxSpeed": 32.33, "minGap": 2.5, "accel": 2.6, "decel": 4.5, "tau": 1.0, "lcStrategic": 1.0, "lcCooperative": 1.0,"lcAssertive": 0.5, "lcSpeedGain": 1.0, "lcKeepRight": 1.0},
    "1a": {'maxSpeed': 30.30259970810396, 'minGap': 2.7534154198947003, 'accel': 1.7048215182278634, 'decel': 1.1045635653229056, 'tau': 1.423368852985237},
    "1b":  {'maxSpeed': 30.174360517924566, 'minGap': 2.7567104401644023, 'accel': 1.3056948937388382, 'decel': 1.0339046789440762, 'tau': 1.4675601739522783},
    "1c": {'maxSpeed': 30.16239221599735, 'minGap': 2.818761285099028, 'accel': 1.8654642734403721, 'decel': 1.1648315742640447, 'tau': 0.5260090373730051},
    "2a": {'lcStrategic': 0.6395627001183368, 'lcCooperative': 0.3642540170229058, 'lcAssertive': 0.08989548250927029, 'lcSpeedGain': 0.4695680888532741, 'lcKeepRight': 4.035116423813789},
    "2b": {'lcStrategic': 0.46070505058093153, 'lcCooperative': 0.36561260090219133, 'lcAssertive': 0.08206888287016509, 'lcSpeedGain': 1.6306187210090093, 'lcKeepRight': 0.3586279593581411},
    "2c":  {'lcStrategic': 0.11640222497083824, 'lcCooperative': 0.49394590298899477, 'lcAssertive': 0.24040835673634667, 'lcSpeedGain': 1.7820156343693352, 'lcKeepRight': 2.721390561408376},
    "3a": {'maxSpeed': 30.116481051532638, 'minGap': 2.9246837509525085, 'accel': 3.6138980697235565, 'decel': 2.999289823239752, 'tau': 1.8452201128675738, 'lcStrategic': 3.0487969842704445, 'lcCooperative': 0.5119595169145059, 'lcAssertive': 0.09087476505204747, 'lcSpeedGain': 0.48288326756878197, 'lcKeepRight': 1.2791339423376002},
    "3b":  {'maxSpeed': 31.15811238953671, 'minGap': 2.9296054625794694, 'accel': 1.8746961428409035, 'decel': 2.7579099400317237, 'tau': 1.195216410299756, 'lcStrategic': 0.7404221362409035, 'lcCooperative': 0.49611840700450593, 'lcAssertive': 0.08853824678040209, 'lcSpeedGain': 0.7633522724278734, 'lcKeepRight': 0.4078627936600342}, # new resu
    "3c": {'maxSpeed': 33.754985273472734, 'minGap': 1.9865826126204338, 'accel': 3.0466653512966855, 'decel': 2.8407628417205637, 'tau': 1.5194700473522247, 'lcStrategic': 0.7093501609745403, 'lcCooperative': 0.5958295428510783, 'lcAssertive': 0.4372298978360208, 'lcSpeedGain': 3.033059771569888, 'lcKeepRight': 3.4745560963764546},
    "cgan": {'maxSpeed': 32.155468058027324, 'minGap': 1.7304751150744706, 'accel': 3.13277526615182, 'decel': 2.2556560753351427, 'tau': 1.376101486623159, 'lcStrategic': 0.7670997584501396, 'lcCooperative': 0.9983358594382765, 'lcAssertive': 0.5716922651157786, 'lcSpeedGain': 1.615401763830451, 'lcKeepRight': 4.302565070438945},
    "gt_sublane": {"carFollowModel": "IDM",
        "maxSpeed": 30.55,
        "minGap": 2.5,
        "accel": 1.5,
        "decel": 2,
        "tau": 1.4,
        "emergencyDecel": 4.0,
        # "speedFactor": 0.8,
        # "speedDev": 0.2,
        "laneChangeModel": "SL2015",
        "lcSublane": 1.0,
        "latAlignment": "arbitrary",
        "maxSpeedLat": 1.4,
        "lcAccelLat": 0.7,
        "minGapLat": 0.4,
        "lcStrategic": 10.0,
        "lcCooperative": 1.0,
        "lcPushy": 0.4,          # Replaces lcAssertive in sublane model
        "lcImpatience": 0.9,
        "lcSpeedGain": 1.5,
        "lcKeepRight": 0.0,
        "lcOvertakeRight": 0.0
    }
}

mainline=["E0_0", "E0_1", "E1_0", "E1_1", "E2_0", "E2_1", "E2_2", "E4_0", "E4_1"]

def training_rmspe():

    '''
    sumo_dir: directory for DETECTOR.out.xml files
    measurement_locations: a list of detectors
    quantity: "volume", "speed" or "occupancy"
    compute RMSPE: the scale doesn't matter
    
    '''

    # Read and extract data
    print("Training RMSPE (detectors)")
    sim1_dict = reader.extract_sim_meas(measurement_locations=[location for location in measurement_locations], file_dir=sumo_dir)
    sim2_dict = reader.extract_sim_meas(measurement_locations=["trial_" + location for location in measurement_locations], file_dir=sumo_dir)

    keys = ["volume", "speed", "occupancy"]
    print_labels = ["Volume q: ", "Speed v: ", "Occupancy o: "]

    epsilon = 1e-6  # Small constant to stabilize logarithmic transformation

    for i, key in enumerate(keys):
        sim1_vals = sim1_dict[key]
        sim2_vals = sim2_dict[key]
        
        # Logarithmic transformation with stabilization
        log_sim1 = np.log(sim1_vals + epsilon)
        log_sim2 = np.log(sim2_vals + epsilon)
        
        # Compute the difference in log-space
        log_diff = log_sim1 - log_sim2
        
        # Root Mean Square Error in log-space
        error = np.sqrt(np.nanmean((log_diff**2).flatten()))
        
        print(print_labels[i] + "{:.2f}".format(error))




    return

def validation_rmspe(exp_name):

    with open("calibration_result/"+pre+"_"+exp_name+".pkl", 'rb') as file:
        macro_sim = pickle.load(file)

    with open("calibration_result/"+pre+"_gt.pkl", 'rb') as file:
        macro_gt = pickle.load(file)

    print("Validation RMSPE (macro simulation data)")
    size1 = min(macro_gt["flow"].shape[0], macro_sim["flow"].shape[0])
    size2 = min(macro_gt["flow"].shape[1], macro_sim["flow"].shape[1])

    keys = ["flow", "speed", "density"]
    print_labels = ["Flow q: ", "Speed v: ", "Density rho: "]

    epsilon = 1e-6  # Small constant to stabilize logarithmic transformation

    for i, key in enumerate(keys):
        sim1_vals = macro_gt[key][:size1,:size2]
        sim2_vals = macro_sim[key][:size1,:size2]
        
        # Logarithmic transformation with stabilization
        log_sim1 = np.log(sim1_vals + epsilon)
        log_sim2 = np.log(sim2_vals + epsilon)
        
        # Compute the difference in log-space
        log_diff = log_sim1 - log_sim2
        
        # Root Mean Square Error in log-space
        error = np.sqrt(np.nanmean((log_diff**2).flatten()))
        
        print(print_labels[i] + "{:.2f}".format(error))
    return

def run_with_param(parameter, exp_label="", rerun=True, plot_ts=False, plot_det=False, plot_macro=False):
    '''
    rerun SUMO using provided parameters
    generate FCD and detector data
    convert FCD to macro data
    save macro data
    '''
    fcd_name = "fcd_onramp_" + exp_label
    # onramp.run_sumo(sim_config = "onramp_gt.sumocfg")
    if rerun:
        onramp.update_sumo_configuration(parameter)
        if exp_label in ["GT", "gt"]:
            onramp.run_sumo(sim_config = "onramp_gt.sumocfg", fcd_output =fcd_name+".xml")
        else:
            onramp.run_sumo(sim_config = "onramp.sumocfg", fcd_output =fcd_name+".xml")

        sim_output = reader.extract_sim_meas(measurement_locations=[location for location in measurement_locations])
        reader.parse_and_reorder_xml(xml_file=fcd_name+".xml", output_csv=fcd_name+".csv") #, link_names=mainline)
        macro_data = macro.compute_macro(fcd_name+".csv", dx=10, dt=10, start_time=0, end_time=480, start_pos=0, end_pos=1300, 
                                        save=True, plot=plot_macro)
        # if plot_macro:
        #     macro.plot_macro_sim(macro_data)

    # plotting
    if plot_ts:
        vis.visualize_fcd(fcd_name+".xml") #, lanes=mainline)  # plot mainline only
    if plot_det:
        # vis.plot_sim_vs_sim(sumo_dir, measurement_locations, quantity="speed")
        fig, axes = vis.plot_line_detectors_sim(sumo_dir, measurement_locations, quantity="speed", label="gt") # continuously adding plots to figure
        fig, axes = vis.plot_line_detectors_sim(sumo_dir, measurement_locations, quantity="speed", fig=fig, axes=axes, label=exp_label)
        plt.show()
    return


def travel_time(fcd_name, rerun_macro=False):
    '''
    To be removed
    Get lane-specific travel time given varying departure time
    '''
    if rerun_macro:
        lane1 = ["E0_1", "E1_1", "E2_2", "E4_1"]
        lane2 = ["E0_0", "E1_0", "E2_1", "E4_0"]
        reader.parse_and_reorder_xml(xml_file=fcd_name+".xml", output_csv=fcd_name+"_lane1.csv", link_names=lane1)
        reader.parse_and_reorder_xml(xml_file=fcd_name+".xml", output_csv=fcd_name+"_lane2.csv", link_names=lane2)
        macro_lane1 = macro.compute_macro(fcd_name+"_lane1.csv", dx=10, dt=10, start_time=0, end_time=480, start_pos=0, end_pos=1300, 
                                            save=True, plot=False)  
        macro_lane2 = macro.compute_macro(fcd_name+"_lane2.csv", dx=10, dt=10, start_time=0, end_time=480, start_pos=0, end_pos=1300, 
                                            save=True, plot=False)
        # macro.plot_macro_sim(macro_lane1)
        # macro.plot_macro_sim(macro_lane2)

        # with open("calibration_result/"+pre+"_gt.pkl", 'rb') as file:
            # macro_gt = pickle.load(file)

    tt = defaultdict(list) # key: lane, value: "departure_time", "travel_time"
    departure_time = np.linspace(0, 400, 40)

    for lane in [1,2]:
        with open(f"calibration_result/macro_fcd_onramp_{EXP}_lane{lane}.pkl", 'rb') as file:
            macro_data = pickle.load(file)

        for t0 in departure_time:
            t_arr, x_arr = macro.gen_VT(macro_data, t0=t0, x0=0)
            if x_arr[-1]-x_arr[0] >= 1299:
                tt[lane].append(t_arr[-1]-t_arr[0])
        num = len(tt[lane])
        plt.plot(departure_time[:num], tt[lane], label=f"lane {lane}")
    plt.xlabel("Departure time (sec)")
    plt.ylabel("Travel time (sec)")
    plt.legend()
    plt.show()


def lane_delay(lanearea_xml):
    '''
    To be removed
    Plot lane-specific quantity in lanearea detector output
    '''
    LANES = ["lane_1", "lane_2"]

    # Parse the XML file
    tree = ET.parse(lanearea_xml)
    root = tree.getroot()

    # Initialize a dictionary to store time-series data for each lane
    lane_data = {lane: 0 for lane in LANES}

    # Iterate over each interval element in the XML
    for interval in root.findall('interval'):
        lane_id = interval.attrib['id']
        if lane_id in LANES:
            # begin_time = float(interval.attrib['begin'])
            mean_time_loss = float(interval.attrib['meanTimeLoss'])
            veh_seen = float(interval.attrib["nVehSeen"])

            # Append time and meanTimeLoss to the corresponding lane
            # lane_data[lane_id]['time'].append(begin_time)
            # lane_data[lane_id][quantity].append(mean_time_loss*veh_seen)
            lane_data[lane_id] += mean_time_loss*veh_seen

    for lane_id, val in lane_data.items():
        print(f"Total delay (veh x sec) in {lane_id}: {val}")
    # Plot the time-series for each lane
    # plt.figure(figsize=(10, 6))
    # for lane_id, data in lane_data.items():
    #     plt.plot(data['time'], data[quantity], label=f"Lane {lane_id}")

    # # Add plot details
    # plt.xlabel("Time (s)")
    # plt.ylabel(quantity)
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    return


if __name__ == "__main__":

    sumo_dir = os.path.dirname(os.path.abspath(__file__)) # current script directory
    measurement_locations = ['upstream_0', 'upstream_1', 
                             'merge_0', 'merge_1', 
                             'downstream_0', 'downstream_1']

    # ============ generate RMSPE results =========================
    # EXP = "1c"
    # run_with_param(best_param_map[EXP], exp_label=EXP, rerun=1, plot_ts=False, plot_det=1, plot_macro=1)
    # training_rmspe()
    # validation_rmspe(EXP)
    # lane_delay("lanearea.out.xml")

    # ============ plot line detectors =========================
    quantity = "speed" # speed, volume, occupancy
    # save_path = r'C:\Users\yanbing.wang\Documents\CorridorCalibration\figures\TRC-i24\synth_detector_occ.png'
    save_path = "gt_detector.png"
    fig=None
    axes=None
    for i,exp_label in enumerate(["gt_sublane"]):
        onramp.update_sumo_configuration(best_param_map[exp_label])
        onramp.run_sumo(sim_config = "onramp.sumocfg")
        fig, axes = vis.plot_line_detectors_sim(sumo_dir, measurement_locations, quantity=quantity, fig=fig, axes=axes, label=exp_label)
    fig.savefig(save_path, dpi=100, bbox_inches='tight') 
    # plt.show()


    # ===== plot 9-grid plot ============= 
    # quantity = "flow" # speed, flow, density
    # save_path = rf'C:\Users\yanbing.wang\Documents\CorridorCalibration\figures\TRC-i24\synth_grid_{quantity}.png'
    # fig=None
    # axes=None
    # for i,exp_label in enumerate(["1a","1b", "1c","2a","2b","2c","3a","3b","3c"]):
    #     macro_pkl = rf'calibration_result/macro_fcd_onramp_{exp_label}.pkl'
    #     with open(macro_pkl, 'rb') as file:
    #         macro_data = pickle.load(file, encoding='latin1')
    #     fig, axes = vis.plot_macro_sim_grid(macro_data, 
    #                         quantity, 
    #                         dx=10, dt=10,
    #                         fig=fig, axes=axes,
    #                         ax_idx=i, label=exp_label)
    # fig.savefig(save_path, dpi=100, bbox_inches='tight') 
    # plt.show()

    # ===== plot 1x3-macro plot ============= 
    # exp_label = "1c"
    # save_path = rf'C:\Users\yanbing.wang\Documents\CorridorCalibration\figures\TRC-i24\synth_macro_{exp_label}.png'
    # macro_pkl = rf'calibration_result/macro_fcd_onramp_{exp_label}.pkl'
    # with open(macro_pkl, 'rb') as file:
    #     macro_data = pickle.load(file, encoding='latin1')
    # fig, ax = macro.plot_macro_sim(macro_data, dx=10, dt=10)
    # fig.savefig(save_path, dpi=100, bbox_inches='tight') 
    # plt.show()