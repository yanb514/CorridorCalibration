"""
This file generates the training and validation results from macrosopic data stored in .pkl
"""
import pickle
import numpy as np
import os
import os
import numpy as np
import sys
import matplotlib.pyplot as plt

main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) # two levels up
sys.path.insert(0, main_path)
import utils_vis as vis
import utils_data_read as reader
import onramp_calibrate as onramp
import macro




pre = "macro_fcd_onramp"
suf  = "_byid.pkl"
exp = {
    "gt": pre + "_gt" + suf,
    "default": pre + "" + suf,
    "1a": pre + "_cf_q" + suf,
    "1b": pre + "_cf_v" + suf,
    "1c": pre + "_cf_rho" + suf,
    "2a": pre + "_lc_q" + suf,
    "2b": pre + "_lc_v" + suf,
    "2c": pre + "_lc_rho" + suf,
    "3a": pre + "_cflc_q" + suf,
    "3b": pre + "_cflc_v" + suf,
    "3c": pre + "_cflc_rho" + suf,
}

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

def training_rmse(exp_name):

    '''
    sumo_dir: directory for DETECTOR.out.xml files
    measurement_locations: a list of detectors
    quantity: "volume", "speed" or "occupancy"
    '''

    # Read and extract data
    print("Training RMSE (detectors)")
    sim1_dict = reader.extract_sim_meas(measurement_locations=[location for location in measurement_locations], file_dir=sumo_dir)
    sim2_dict = reader.extract_sim_meas(measurement_locations=["trial_" + location for location in measurement_locations], file_dir=sumo_dir)

    sim1_dict["speed"]*=3.6
    sim2_dict["speed"]*=3.6
    sim1_dict["density"] = sim1_dict["volume"]/sim1_dict["speed"]
    sim2_dict["density"] = sim2_dict["volume"]/sim2_dict["speed"]

    diff = sim1_dict["volume"] - sim2_dict["volume"]
    error = np.sqrt(np.nanmean(diff.flatten()**2))
    print("Volume q (vph): {:.2f}".format(error))  #convert veh/s to veh/hr
          
    diff = sim1_dict["speed"] - sim2_dict["speed"]
    error = np.sqrt(np.nanmean(diff.flatten()**2))
    print("Speed v (km/hr): {:.2f}".format(error)) # km/hr 

    diff = sim1_dict["density"] - sim2_dict["density"]
    error = np.sqrt(np.nanmean(diff.flatten()**2))
    print("Density rho (veh/km): {:.2f}".format(error))

    return

def validation_rmse(exp_name):

    with open(exp[exp_name], 'rb') as file:
        macro_sim = pickle.load(file)

    print("Validation RMSE (macro simulation data)")
    size1 = min(macro_gt["flow"].shape[0], macro_sim["flow"].shape[0])
    size2 = min(macro_gt["flow"].shape[1], macro_sim["flow"].shape[1])

    diff = (macro_gt["flow"][:size1,:size2] - macro_sim["flow"][:size1,:size2])*3600
    error = np.sqrt(np.nanmean(diff.flatten()**2))
    print("Volume q (vph): {:.2f}".format(error))  #convert veh/s to veh/hr

    diff = (macro_gt["speed"][:size1,:size2] - macro_sim["speed"][:size1,:size2])*3.6
    error = np.sqrt(np.nanmean(diff.flatten()**2))
    print("Speed v (km/hr): {:.2f}".format(error)) # km/hr 

    diff = (macro_gt["density"][:size1,:size2] - macro_sim["density"][:size1,:size2])*1000
    error = np.sqrt(np.nanmean(diff.flatten()**2))
    print("Density rho (veh/km): {:.2f}".format(error))

    return



def read_opt_result(result_pkl):
    with open(result_pkl, 'rb') as file:
        result = pickle.load(file)
    print("Optimization result:", result)
    print("Best parameters found:", result.x)
    print("Objective function value at best parameters:", result.fun)
    return

if __name__ == "__main__":

    sumo_dir = os.path.dirname(os.path.abspath(__file__)) # current script directory
    measurement_locations = ['upstream_0', 'upstream_1', 
                             'merge_0', 'merge_1', 
                             'downstream_0', 'downstream_1']
    
    onramp.update_sumo_configuration(best_param_map["default"])
    onramp.run_sumo(sim_config = "onramp.sumocfg")
    onramp.run_sumo(sim_config = "onramp_gt.sumocfg")
    # get ground truth detector data
    sim1_dict = reader.extract_sim_meas(measurement_locations=measurement_locations, file_dir=sumo_dir)
        
    with open("calibration_result/"+exp["gt"], 'rb') as file:
        macro_gt = pickle.load(file)
# # "default","1a", "1b", "1c","2a","2b","2c","3a","3b",
#     for i, exp_label in enumerate(["3b", "3c"]):
#         onramp.update_sumo_configuration(best_param_map["default"])
#         onramp.update_sumo_configuration(best_param_map[exp_label])
#         onramp.run_sumo(sim_config = "onramp.sumocfg")
#         print(exp_label)
#         training_rmse(exp_label)
#         validation_rmse(exp_label)


    # ============ lane-specific simualted measurements vs. gt
    fig = None
    axes = None
    # quantity = "volume"
    # # experiments = ["gt", "default", "1a", "2a", "3a"]
    quantity = "speed"
    experiments = ["gt", "default", "1b", "2b", "3b"]
    # quantity = "occupancy"
    # experiments = ["gt", "default", "1c", "2c", "3c"]
    for exp_label in experiments:
        param = best_param_map[exp_label]
        onramp.update_sumo_configuration(best_param_map["default"])
        onramp.update_sumo_configuration(param)
        onramp.run_sumo(sim_config = "onramp.sumocfg")
        print(exp_label)
        training_rmse(exp_label)
        validation_rmse(exp_label)
        fig, axes = vis.plot_line_detectors_sim(sumo_dir, measurement_locations, quantity, fig, axes, exp_label) # continuously adding plots to figure
    plt.show()
    
    # ============ plot time-space macroscopic for all 9 exps in grid 3x3 ===============
    quantity = "speed"
    for i, exp_label in enumerate(["1a", "1b", "1c","2a","2b","2c","3a","3b","3c"]):
        with open("calibration_result/"+exp[exp_label], 'rb') as file:
            macro_sim = pickle.load(file)
        fig, axes = vis.plot_macro_sim_grid(macro_sim, quantity, dx=10, dt=10, fig=fig, axes=axes, ax_idx=i, label=exp_label)
    plt.savefig(rf'..\..\figures\synth_macro_{quantity}.png')
    plt.show()

    # ============ plot macroscopic 3x1 ==========

    with open("calibration_result/"+exp["default"], 'rb') as file:
        macro_sim = pickle.load(file)
    macro.plot_macro_sim(macro_sim)