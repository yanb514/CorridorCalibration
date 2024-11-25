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
import macro
from collections import defaultdict
import xml.etree.ElementTree as ET


pre = "macro_fcd_onramp"
# suf  = "_byid.pkl"
# exp = {
#     "gt": pre + "_gt",
#     "default": pre + "" + suf,
#     "1a": pre + "_cf_q" + suf,
#     "1b": pre + "_cf_v" + suf,
#     "1c": pre + "_cf_rho" + suf,
#     "2a": pre + "_lc_q" + suf,
#     "2b": pre + "_lc_v" + suf,
#     "2c": pre + "_lc_rho" + suf,
#     "3a": pre + "_cflc_q" + suf,
#     "3b": pre + "_cflc_v" + suf,
#     "3c": pre + "_cflc_rho" + suf,
# }

best_param_map = {
    "gt": { "maxSpeed": 30.55, "minGap": 2.5, "accel": 1.5, "decel": 2, "tau": 1.4, "lcStrategic": 1.0, "lcCooperative": 1.0,"lcAssertive": 0.5, "lcSpeedGain": 1.0, "lcKeepRight": 0.5},
    "default":  { "maxSpeed": 32.33, "minGap": 2.5, "accel": 2.6, "decel": 4.5, "tau": 1.0, "lcStrategic": 1.0, "lcCooperative": 1.0,"lcAssertive": 0.5, "lcSpeedGain": 1.0, "lcKeepRight": 1.0},
    "1a":{'maxSpeed': 30.438177087377383, 'minGap': 2.7154211528218135, 'accel': 1.0969376713390915, 'decel': 2.1563832118867414, 'tau': 1.4505762714817776},
    "1b": {'maxSpeed': 30.497289567282024, 'minGap': 2.859372370601303, 'accel': 1.1086621104873673, 'decel': 1.9781537645876819, 'tau': 1.3856158933432625},
    "1c": {'maxSpeed': 32.61769788158533, 'minGap': 2.4452488415195335, 'accel': 1.0000135604576716, 'decel': 2.911121855619264, 'tau': 1.564551421039515},
    "2a": {'lcStrategic': 1.047192894872194, 'lcCooperative': 0.8645387614240766, 'lcAssertive': 0.39033097381529464, 'lcSpeedGain': 0.7680291002087158, 'lcKeepRight': 4.395423080752877, 'lcOvertakeRight': 0.44198548511444324},
    "2b": {'lcStrategic': 0.47837275159543946, 'lcCooperative': 0.8599243307840726, 'lcAssertive': 0.1909699035864018, 'lcSpeedGain': 4.287017983890513, 'lcKeepRight': 1.6517538483664194, 'lcOvertakeRight': 0.8233156865096709},
    "2c": {'lcStrategic': 0.3894270091843165, 'lcCooperative': 0.7366477001268105, 'lcAssertive': 0.17652970576044152, 'lcSpeedGain': 2.9021162967920486, 'lcKeepRight': 2.598242165430954, 'lcOvertakeRight': 0.21302179905397123},
    "3a": {'maxSpeed': 31.44813279984895, 'minGap': 1.8669305739182382, 'accel': 2.2398476082518677, 'decel': 2.5073714738472153, 'tau': 1.3988475504128757, 'lcStrategic': 0.8624217521963465, 'lcCooperative': 0.9789774143646455, 'lcAssertive': 0.43478229746049984, 'lcSpeedGain': 1.1383219615950644, 'lcKeepRight': 4.030227753894549},
    "3b": {'maxSpeed': 31.605877951781565, 'minGap': 2.4630185481679043, 'accel': 1.6173674534215892, 'decel': 2.4864299905414677, 'tau': 1.4482507669327735, 'lcStrategic': 1.414282922055993, 'lcCooperative': 0.9998246130488315, 'lcAssertive': 0.5454520350957692, 'lcSpeedGain': 3.7567851330319795, 'lcKeepRight': 0.3604351181518853},
    "3c": {'maxSpeed': 30.53284221198521, 'minGap': 2.7958695360441843, 'accel': 2.4497572915690244, 'decel': 2.4293815796265275, 'tau': 1.374376527326827, 'lcStrategic': 1.3368371035725628, 'lcCooperative': 0.9994681517674497, 'lcAssertive': 0.35088886304156547, 'lcSpeedGain': 1.901166989734572, 'lcKeepRight': 0.7531568339763854},

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

    # sim1_dict["density"] = sim1_dict["volume"]/sim1_dict["speed"]
    # sim2_dict["density"] = sim2_dict["volume"]/sim2_dict["speed"]

    keys = ["volume", "speed", "occupancy"]
    print_labels = ["Volume q: ", "Speed v: ", "Occupancy o: "]
    small_vals = [0.1, 0.1, 0.001]
    for i, key in enumerate(keys):
        relative_diff = (sim1_dict[key] - sim2_dict[key]) \
                    / np.nan_to_num(sim1_dict[key], nan=small_vals[i]) # ensures NaN values in measured_output are replaced with 1 to avoid division by zero or NaN issues.
        error = np.sqrt(np.nanmean((relative_diff**2).flatten()))
        print(print_labels[i]+"{:.2f}".format(error))

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
    small_vals = [0.1, 0.1, 0.001]
    
    for i, key in enumerate(keys):
        denominator = np.nan_to_num(macro_gt[key], nan=small_vals[i])
        denominator = np.where(denominator == 0, small_vals[i], denominator)
        relative_diff = (macro_gt[key][:size1,:size2] - macro_sim[key][:size1,:size2]) \
                    / denominator # ensures NaN values in measured_output are replaced with 1 to avoid division by zero or NaN issues.
        # print(denominator)
        error = np.sqrt(np.nanmean((relative_diff**2).flatten()))
        print(print_labels[i]+"{:.2f}".format(error))

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
        onramp.run_sumo(sim_config = "onramp.sumocfg", fcd_output =fcd_name+".xml")

        sim_output = reader.extract_sim_meas(measurement_locations=[location for location in measurement_locations])
        reader.parse_and_reorder_xml(xml_file=fcd_name+".xml", output_csv=fcd_name+".csv") #, link_names=mainline)
        macro_data = macro.compute_macro(fcd_name+".csv", dx=10, dt=10, start_time=0, end_time=480, start_pos=0, end_pos=1300, 
                                        save=True, plot=plot_macro)
        # macro.plot_macro_sim(macro_data)

    # plotting
    if plot_ts:
        vis.visualize_fcd(fcd_name+".xml") #, lanes=mainline)  # plot mainline only
    if plot_det:
        # vis.plot_sim_vs_sim(sumo_dir, measurement_locations, quantity="speed")
        fig, axes = vis.plot_line_detectors_sim(sumo_dir, measurement_locations, quantity="volume", label="gt") # continuously adding plots to figure
        fig, axes = vis.plot_line_detectors_sim(sumo_dir, measurement_locations, quantity="volume", fig=fig, axes=axes, label=exp_label)
        plt.show()
    return


def travel_time(fcd_name, rerun_macro=False):
    '''
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

    EXP = "default"
    run_with_param(best_param_map[EXP], exp_label=EXP, rerun=True, plot_ts=False, plot_det=False, plot_macro=False)
    # training_rmspe()
    # validation_rmspe(EXP)
    # lane_delay("lanearea.out.xml")



    # plt.show()
