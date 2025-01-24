# **SUMO-in-the-loop Corridor Calibration**  

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)  
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg) 

This tool is designed to calibrate microscopic traffic flow models using macroscopic (aggregated) data from stationary detectors. It uses a SUMO-in-the-loop calibration framework with the goal of replicating observed macroscopic traffic features. A set of performance measures are selected to evaluate the models' ability to replicate traffic flow characteristics. A case study to calibrate the flow on a segment of Interstate 24 is included.

---

## **Table of Contents**  
- [Features](#features)  
- [Installation](#installation)  
- [Usage](#usage)     
- [License](#license)  
- [Acknowledgments](#acknowledgments)  

---

## **Features**  
- Automatic calibration of driving behavior parameters, including key car-following and lane-change parameters 
- Utilize a global optimization algorithm 
- Support parallel computation
- Customizable evaluation metrics 
- Two scenarios: synthetic highway on-ramp merging (on_ramp), and I-24 Westbound (I24scenario) 

---

## **Installation**  

### Dependencies  
This package is built on Python 3.11, and requires installation of [optuna](https://optuna.org/).

## **Usage**  

### Data download and processing
The RDS detector data for the I-24 scenario is from Tennessee Department of Transportation, and can be downloaded from the [I-24 MOTION project website](https://i24motion.org). 

The raw RDS data is filtered and processed using:
```python
# write original dat.gz file to a .csv file. Please see inline documentation
utils_data_read.read_and_filter_file()
```
### Configuration
First set the configuration used in `sumo/I24scenario/i24_calibrate.py`. Create a `config.json` file under `sumo/` with the following setting:
```json
{
    "SCENARIO": "I24_scenario" or "on_ramp",
    "EXP": "3b", # experiment label "1a", "1b", "1c",...,"3c"
    "N_TRIALS": 20000, # optimization trials
    "N_JOBS": 120, # cores
    "RDS_DIR": "", # directory of the processed RDS .csv file
    "SUMO_EXE": "", # sumo .exe file path
    "SUMO_EXE_PATH": "", # alternative sumo .exe file path
}
```

### Running calibration  
To run the calibration of I-24 scenario:
```bash  
cd sumo/I24scenario
python i24_calibrate.py
```  
The calibration progress such as current best parameters will be saved in `sumo/I24scenario/_log`.

### Evaluation and plotting
All evaluation related computations are located in `sumo/I24scenario/i24_calibrate_results.py`.

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

---

## **License**  
This project is licensed under the [MIT License](LICENSE).  

---

## **Acknowledgments**  
- The work can be cited as:
```
@misc{wang2024calibrating,
  title={Calibrating Microscopic Traffic Models with Macroscopic Data},
  author={Wang, Yanbing and de Souza, Felipe and Zhang, Yaozhong and Karbowski, Dominik},
  note={https://ssrn.com/abstract=5065262},
  year={2024}
}
```
- The work is sponsored by the U.S. Department of Energy (DOE) Vehicle Technologies Office (VTO) under the Energy Efficient Mobility Systems (EEMS) Program.
