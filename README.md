# **SUMO-in-the-loop Corridor Calibration**  

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)  
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg) 

This tool is designed to calibrate microscopic traffic flow models using macroscopic (aggregated) data from stationary detectors. It uses a SUMO-in-the-loop calibration framework with the goal of replicating observed macroscopic traffic features. A set of performance measures are selected to evaluate the models' ability to replicate traffic flow characteristics. A case study to calibrate the flow on a segment of Interstate 24 is included.

![corridor setup](i24_corridor_2.png)
---

## **Acknowledgments**  
- The work is sponsored by the U.S. Department of Energy (DOE) Vehicle Technologies Office (VTO) under the Energy Efficient Mobility Systems (EEMS) Program.
- The work can be cited as:
```
@misc{wang2024calibrating,
  title={Calibrating Microscopic Traffic Models with Macroscopic Data},
  author={Wang, Yanbing and de Souza, Felipe and Zhang, Yaozhong and Karbowski, Dominik},
  note={https://ssrn.com/abstract=5065262},
  year={2024}
}
```
---

## **Table of Contents**  
- [Acknowledgments](#acknowledgments)  
- [Features](#features)  
- [Installation](#installation)  
- [Usage](#usage)     

---

## **Features**  
- Automatic calibration of driving behavior parameters, including key car-following and lane-change parameters 
- Utilize a global optimization algorithm 
- Support parallel computation
- Customizable evaluation metrics 
- Scenarios:
  - `onramp`: a synthetic highway on-ramp merging scenario
  - `i24`: I-24 Westbound between postmile 54.1 to 57.4
  - `i24b`: I-24 Westbound between postmile 58.7 to 62.9, to support benchmarking work.
  - `roosevelt` (coming soon): Chicago Roosevelt Rd from Canal to Michigan with SPaT. 

---

## **Installation**  

### Dependencies  
This package is built on Python 3.11, and requires installation of [optuna](https://optuna.org/), and [sumo](https://sumo.dlr.de/docs/Installing/index.html)

## **Usage**  
Follow these steps to prepare a calibration run. Make sure first the files are organized according to the following directory:

### Directory structure
```
CorridorCalibration
│   README.md
│   utils_data_read.py    
│   utils_macro.py
│   utils_vis.py
│
└───sumo
│   │   config.json
│   │
│   └───<SCENARIO>
│       │   SCENARIO_calibrate.py
│       │   SCENARIO_result.py
│       │   SCENARIO.sumocfg
│       │   SCENARIO.net.xml
│       │   SCENARIO.rou.xml
│       │   ...
│       │   
│       └───_log
│       └───calibration_result
│       └───data
│           │   Traffic measurements from stationary sensors
│           │   ...
```

### 1. Data download and processing
The RDS detector data for the I-24 scenario is from Tennessee Department of Transportation, and can be downloaded from the [I-24 MOTION project website](https://i24motion.org). 

The raw RDS data is filtered and processed using:
```python
# write original dat.gz file to a .csv file. Please see inline documentation
utils_data_read.read_and_filter_file()
```
### 2. Configuration setup
Create a `config.json` file under `sumo/` with the following template:
```json
{
    "EXP": "3b",
    "N_TRIALS": 20000,
    "N_JOBS": 120,
    "SUMO_PATH": "REPLACE_WITH_YOUR_LOCAL_SUMO_PATH",
    "onramp":{
        "SIMULATION_TIME": 480,
        "N_ROUTES": 2,
        "N_INTERVALS": 1
    },
    "i24":{
        "SIMULATION_TIME": 21600,
        "N_ROUTES": 5,
        "N_INTERVALS": 12,
        "RDS_DIR": "REPLACE_WITH_YOUR_LOCAL_RDS_PATH"
    },
    "i24b":{
        "RDS_DIR": "REPLACE_WITH_YOUR_LOCAL_RDS_PATH"
    },

    "DEFAULT_PARAMS": {
        "cf":{
            "maxSpeed": 30.55,
            "minGap": 2.5,
            "accel": 1.5,
            "decel": 2,
            "tau": 1.4,
            "emergencyDecel": 4.0
        },
        "lc":{
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
    },
    "PARAMS_RANGE": {
        "cf":{
            "maxSpeed": [30, 35],
            "minGap": [1,3],
            "accel": [1,4],
            "decel": [1,3],
            "tau": [0.5,2]
        },
        "lc":{
            "lcSublane": [0,10],
            "maxSpeedLat": [0,10],
            "lcAccelLat": [0,5],
            "minGapLat": [0,5],
            "lcStrategic": [0,10],
            "lcCooperative": [0,10],
            "lcPushy": [0,1],
            "lcImpatience": [0,1],
            "lcSpeedGain": [0,1]
        }
    }
}
```
Here are a few definitions inside the config setting:
| Setting | Description |
|----------------|-----------------|
| `cf` | Car-following parameters. default CF model is IDM |
| `lc` | Lane-change parameters. default LC model is SL2015 |
| `EXP`       | "1a", "1b", ..., "3b", "3c". 1=cf params, 2=lc params, 3=both using measurement a=flow, b=velocity, c=occupancy. default: "3b" (calibrate both CF and LC params using speed measurements) |
| `N_TRIALS`  | The number of trials for each process in `optuna.study.Study.optimize()` |
| `N_JOBS`    | The number of parallel jobs in `optuna.study.Study.optimize()`   |
| `DEFAULT_PARAMS` | These default values will overwrite (reset) the values in `SCENARIO.rou.xml` at the beginning of each calibration run|
| `PARAMS_RANGE` | [min, max] for each parameter|



### 3. Running calibration  
To run the calibration of any scenario, navigate to the SCENARIO folder and run `SCENARIO_calibrate.py`. For example, to run the `i24b` scenario:
```bash  
cd sumo/i24b
python i24b_calibrate.py
```  
The calibration progress such as current best parameters will be saved in `sumo/i24/_log`.

### 4. Evaluation and plotting (TODO: i24b in progress)
All evaluation related computations are located in `sumo/SCENARIO/SCENARIO_results.py`. Current evaluation & visualization support:
1. Calibrated vs. measured speed at the locations of stationary sensors
![detector speeds](det_b.png)
2. Macroscopic quantities (speed, flow, density)
![macroscopic quantities](asm_5hr.png)
3. Lane-specific travel time
![travel time](travel_time_rds.png)

### Key utility functions
In summary,
- `utils_data_read.py` contains functions to read and process RDS and .xml data
- `utils_vis.py` contains all visualization functions
- `utils_macro.py` contains Edie's method to compute macroscopic traffic quantities from trajectory data

The detailed descriptions of these methods are documented inline. To highlight a few:
- `utils_data_read.parse_and_reorder_xml()` takes the SUMO floating car data (fcd) output `.xml` file, reorders by trajectory and time into NGSIM data format.
- `utils_macro.compute_macro_generalized()` implements the generalized Edie's method, and processes trajectory data into macroscopic quantities for the specified spatial and temporal window.
- `utils_macro.plot_macro()` plots the macroscopic quantities of flow, density and speed computed using `macro.compute_macro_generalized()`.
- `utils_vis.visualize_fcd()` plots the time-space diagram given the fcd file.
- `utils_vis.plot_line_detectors()` plot the aggregated traffic data generated from SUMO at the specified detector locations.

### Using calibrated SUMO
If you only want to work with the calibrated SUMO scenarios without the calibration, you are in good hands!
All calibrated scenarios are located in `sumo/SCENARIO/calibrated`, which contains all the necessary files to run SUMO. You can run `SCENARIO.sumocfg` directly using SUMO-gui, or using command line 
```bash
cd sumo/SCENARIO/calibrated
sumo -c SCENARIO.sumocfg
```

### TODOS
- temp files handling in i24 and onramp scenarios
- add calibrated results for i24b scenario (i24b_results.py and plotting)
- calibrate only westbound?
---
