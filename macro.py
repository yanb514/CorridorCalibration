import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import csv
import xml.etree.ElementTree as ET
import pickle
from matplotlib.ticker import FuncFormatter
import datetime

def reorder_by_id(trajectory_file, link_names="all", lane_name=""):
    # read the trajectory file to see if it is ordered by time (time increases from previous row)
    # assumption: trajectory_file is either ordered by time or by vehicleID
    # separate files by laneID if bylane is True
    
    prev_time = -1
    ordered_by_time = True
    with open(trajectory_file, mode='r') as file:
        csv_reader = csv.reader(file)
        _ = next(csv_reader)
        for line in csv_reader:
            # Split the line into columns
            # columns = line.strip().split()
            columns = line[0].strip().split()
            # print(columns)
            try:
                curr_time = float(columns[1]) # int(columns[0])
            except IndexError:
                columns = columns[0].split(',')
                # print(columns)
                curr_time = float(columns[1]) 
                
            if curr_time < prev_time:
                print(trajectory_file + " is NOT ordered by time")
                ordered_by_time = False
                return
            prev_time = curr_time
    
    # reorder this file by ID
    if ordered_by_time:
        print(trajectory_file + " is ordered by time, reordering it by vehicleID...")
        with open(trajectory_file, mode='r') as file:
            csv_reader = csv.reader(file)
            headers = next(csv_reader)
            # print(csv_reader)
            rows = [row[0].strip().split() for row in csv_reader]
        # Sort the rows by vehicleID and then by time within each vehicleID
        print("Start time: ", rows[0][1])
        print("End time: ", rows[-1][1])
        print("Start pos: ", rows[0][3]) # TODO may not be accurate
        print("End pos: ", rows[-1][3])

        rows.sort(key=lambda x: (x[0], float(x[1])))

        # if bylane==True:
        #     # Organize rows by laneID
        #     lanes = defaultdict(list)
        #     for row in rows:
        #         lane_id = row[2]  # assuming laneID is in the third column
        #         lanes[lane_id].append(row)
            
        #     for lane_id, lane_rows in lanes.items():
        #         output_file = trajectory_file.replace(".csv", f"_{lane_id}.csv")
        #         with open(output_file, mode='w') as file:
        #             file.write(",".join(headers) + "\n")  # Write the headers
        #             for row in lane_rows:
        #                 file.write(" ".join(str(num) for num in row) + "\n")
        if link_names == "all":
            # Write the sorted rows to a new CSV file
            output_file = trajectory_file.replace(".csv", "_byid.csv")
            with open(output_file, mode='w') as file:
                # csv_writer = csv.writer(file)
                file.write(",".join(str(num) for num in headers)+"\n")  # Write the headers
                # csv_writer.writerows(rows)
                for row in rows:
                    file.write(" ".join(str(num) for num in row)+"\n")

        else: # write selected link names
            # Write the sorted rows to a new CSV file
            output_file = trajectory_file.replace(".csv", f"_{lane_name}.csv")
            with open(output_file, mode='w') as file:
                # csv_writer = csv.writer(file)
                file.write(",".join(str(num) for num in headers)+"\n")  # Write the headers
                # csv_writer.writerows(rows)
                for row in rows:
                    lane_id = row[2]
                    if lane_id in link_names:
                        file.write(" ".join(str(num) for num in row)+"\n")

    return

def compute_macro(trajectory_file, dx, dt, start_time, end_time, start_pos, end_pos, save=False, plot=True):
    '''
    Compute mean speed, density and flow from trajectory data using Edie's definition
    flow Q = TTD/(dx dt)
    density Rho = TTT/(dx dt)
    speed = Q/Rho, or can be computed directory from data
    '''
    # find the spatial and temporal ranges
    start_x = 999999
    end_x = -9999999
    t1 = 999999
    t2 = -999999
    
    first_line = True
    with open(trajectory_file, mode='r') as input_file:
        for line in input_file:
            # Split the line into columns
            columns = line.strip().split()
            # print(columns, type(columns))
            vehicle_id = columns[0] # int(columns[0])
           
            
            if first_line and not vehicle_id.isalpha(): # has all letters, then the first line is a header, skip it
                print("skip the header")
                first_line = False
                continue
            # update boundaries
            try:
                curr_time = float(columns[1]) #* 0.1
            except IndexError:
                columns = columns[0].split(',')
                curr_time = float(columns[1])
            curr_x = float(columns[3])
            start_x = min(start_x, curr_x)
            end_x = max(end_x, curr_x)
            t1 = min(t1, curr_time)
            t2 = max(t2, curr_time)

    print(f"Ranges of {trajectory_file}, start: {t1}, end: {t2}, start_pos: {start_x}, end_pos: {end_x}")
    
    # initialize
    prev_vehicle_id = None
    time_range = end_time - start_time
    space_range = end_pos - start_pos
    TTT_matrix = np.zeros((int(time_range / dt), int( space_range / dx)))
    TTD_matrix = np.zeros((int(time_range / dt), int( space_range / dx)))

    # Read the data file line by line
    first_line = True
    with open(trajectory_file, mode='r') as input_file:
        for line in input_file:
            # Split the line into columns
            columns = line.strip().split()
            if len(columns)==1:
                columns = columns[0].split(',')
            vehicle_id = columns[0] # int(columns[0])

            # if first_line and not vehicle_id.isalpha(): # has all letters, then the first line is a header, skip it
            if first_line and all(isinstance(item, str) for item in columns):
                first_line = False
                print("skip header")
                continue

            # extract current info
            if prev_vehicle_id is None:
                traj = defaultdict(list)

            elif vehicle_id != prev_vehicle_id: # start a new dict
                # write previous traj to matrics
                data_index = pd.DataFrame.from_dict(traj)
                data_index['space_index'] = (data_index['p'] // dx)
                data_index['time_index'] = (data_index['timestamps'] // dt)
                grouped = data_index.groupby(['space_index', 'time_index'])

                for (space, time), group_df in grouped:
                    if (time >= 0 and time < (int(time_range / dt)) and space >= 0 and space < (int(space_range / dx))):
                        TTD_matrix[int(time)][int(space)] += (group_df.p.max() - group_df.p.min()) # meter
                        TTT_matrix[int(time)][int(space)] += (group_df.timestamps.max() - group_df.timestamps.min()) #sec
                # break
                traj = defaultdict(list)

            time = float(columns[1]) #* 0.1#* 0.1
            foll_v_val = float(columns[4])
            foll_p_val = float(columns[3])

            # foll_a_val = float(columns[5])
            traj["timestamps"].append(time)
            traj["v"].append(foll_v_val)
            traj["p"].append(foll_p_val)

            prev_vehicle_id = vehicle_id
            

    Q = TTD_matrix/(dx*dt)
    Rho = TTT_matrix/(dx*dt)
    macro_data = {
        "flow": Q,
        "density": Rho,
        "speed": Q/Rho,
    }
    if save:
        trajectory_file_name = trajectory_file.split(".")[0]
        with open(f'macro_{trajectory_file_name}.pkl', 'wb') as f:  # open a text file
            pickle.dump(macro_data, f) # serialize the list
        print(f'macro_{trajectory_file_name}.pkl  file saved.')
    # Plotting
    if plot:
        plot_macro(macro_data, dx, dt)

    return macro_data



def plot_macro_sim(macro_data, dx=10, dt=10):
    '''
    plot heatmap of Q, Rho and V in one plot
    '''
    fs = 18
    minutes = 10
    length = int(minutes * 60/dt)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = fs
    fig, axs = plt.subplots(1,3, figsize=(20, 6))
    Q, Rho, V = macro_data["flow"][:length,:], macro_data["density"][:length,:], macro_data["speed"][:length,:]

    # flow
    h = axs[0].imshow(Q.T*3600, aspect='auto',vmin=0, vmax=4000)# , vmax=np.max(Q.T*3600)) # 2000 convert veh/s to veh/hr/lane
    colorbar = fig.colorbar(h, ax=axs[0])
    axs[0].set_title("Flow (vph)")

    # density
    h= axs[1].imshow(Rho.T*1609.34, aspect='auto',vmin=0, vmax=600) #, vmax=np.max(Rho.T*1000)) # 200 convert veh/m to veh/mile
    colorbar = fig.colorbar(h, ax=axs[1])
    axs[1].set_title("Density (veh/mile)")

    # speed
    h = axs[2].imshow(V.T * 2.23694, aspect='auto',vmin=0, vmax=60) #, vmax=110) # 110 convert m/s to mph
    colorbar = fig.colorbar(h, ax=axs[2])
    axs[2].set_title("Speed (mph)")

    def time_formatter(x, pos):
        # Calculate the time delta in minutes
        minutes = 5*60 + x * xc # starts at 5am
        # Convert minutes to hours and minutes
        time_delta = datetime.timedelta(minutes=minutes)
        # Convert time delta to string in HH:MM format
        return str(time_delta)[3:]  # Remove seconds part

    # Multiply x-axis ticks by a constant
    xc = dt/60  # convert sec to min
    yc = dx
    for ax in axs:
        ax.invert_yaxis()
        # xticks = ax.get_xticks()
        # ax.set_xticks(xticks)
        # ax.set_xticklabels(["{:.1f}".format(5 + tick * xc /60) for tick in xticks])
        ax.xaxis.set_major_formatter(FuncFormatter(time_formatter))
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        # ax.xaxis.set_major_formatter(FuncFormatter(time_formatter))
        yticks = ax.get_yticks()
        # ax.set_yticks(yticks)
        ax.set_yticklabels([str(int(tick * yc)) for tick in yticks])
        # ax.set_yticklabels([str(int(tick * yc/ 1609.34)) for tick in yticks])
        ax.set_ylabel("Position (m)")
        ax.set_xlabel("Time (min)")
        
    plt.tight_layout()
    plt.show()




def plot_macro(macro_data, dx=10, dt=10):
    '''
    plot heatmap of Q, Rho and V in one plot
    '''
    hours = 3
    length = int(hours * 3600/dt)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 18
    fig, axs = plt.subplots(1,3, figsize=(20, 6))
    Q, Rho, V = macro_data["flow"][:length,:], macro_data["density"][:length,:], macro_data["speed"][:length,:]

    # flow
    h = axs[0].imshow(Q.T*3600, aspect='auto',vmin=0, vmax=8000)# , vmax=np.max(Q.T*3600)) # 2000 convert veh/s to veh/hr/lane
    colorbar = fig.colorbar(h, ax=axs[0])
    axs[0].set_title("Flow (nVeh/hr)")

    # density
    h= axs[1].imshow(Rho.T*1000, aspect='auto',vmin=0) #, vmax=np.max(Rho.T*1000)) # 200 convert veh/m to veh/km
    colorbar = fig.colorbar(h, ax=axs[1])
    axs[1].set_title("Density (veh/km)")

    # speed
    h = axs[2].imshow(V.T * 2.23694, aspect='auto',vmin=0, vmax=80) #, vmax=110) # 110 convert m/s to mph
    colorbar = fig.colorbar(h, ax=axs[2])
    axs[2].set_title("Speed (mph)")

    def time_formatter(x, pos):
        # Calculate the time delta in minutes
        minutes = 5*60 + x * xc # starts at 5am
        # Convert minutes to hours and minutes
        time_delta = datetime.timedelta(minutes=minutes)
        # Convert time delta to string in HH:MM format
        return str(time_delta)[:-3]  # Remove seconds part

    # Multiply x-axis ticks by a constant
    xc = dt/60  # convert sec to min
    yc = dx
    for ax in axs:
        ax.invert_yaxis()
        # xticks = ax.get_xticks()
        # ax.set_xticks(xticks)
        # ax.set_xticklabels(["{:.1f}".format(5 + tick * xc /60) for tick in xticks])
        ax.xaxis.set_major_formatter(FuncFormatter(time_formatter))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        yticks = ax.get_yticks()
        # ax.set_yticks(yticks)
        ax.set_yticklabels(["{:.1f}".format(57.6- tick * yc / 1609.34 ) for tick in yticks])
        # ax.set_yticklabels([str(int(tick * yc/ 1609.34)) for tick in yticks])
        ax.set_xlabel("Time (hour of day)")
        ax.set_ylabel("Milemarker")
        
    plt.tight_layout()
    plt.show()

def compare_macro(macro_data_1, macro_data_2):
    fig, axs = plt.subplots(1,3, figsize=(20, 5))
    Q1, Rho1, V1 = macro_data_1["flow"], macro_data_1["density"], macro_data_1["speed"]
    Q2, Rho2, V2 = macro_data_2["flow"], macro_data_2["density"], macro_data_2["speed"]
    Q, Rho, V = Q1-Q2, Rho1-Rho2, V1-V2

    # flow
    h = axs[0].imshow(Q.T*3600, aspect='auto',vmin=-1500, vmax=1500, cmap="bwr") # convert veh/s to veh/hr
    colorbar = fig.colorbar(h, ax=axs[0])
    axs[0].set_title("Flow difference (veh/hr)")

    # density
    h= axs[1].imshow(Rho.T*1000, aspect='auto',vmin=-100, vmax=100, cmap="bwr") # convert veh/m to veh/km
    colorbar = fig.colorbar(h, ax=axs[1])
    axs[1].set_title("Density difference (veh/km)")

    # speed
    h = axs[2].imshow(V.T * 3.6, aspect='auto',vmin=-10, vmax=10, cmap="bwr") # convert m/s to km/hr
    colorbar = fig.colorbar(h, ax=axs[2])
    axs[2].set_title("Speed difference (km/hr)")


    # Multiply x-axis ticks by a constant
    dx, dt = 10, 10
    xc = dt/60  # Example constant
    yc = dx
    for ax in axs:
        ax.set_xlim([0,8])
        ax.invert_yaxis()
        xticks = ax.get_xticks()
        ax.set_xticklabels([str(int(tick * xc)) for tick in xticks]) # convert to milemarker
        yticks = ax.get_yticks()
        ax.set_yticklabels([str(int(tick * yc)) for tick in yticks])
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Space (m)")
    plt.show()


def idm_fd(theta):
    '''
    Traffic Flow Dynamics Treiber L09 https://www.mtreiber.de/Vkmod_Skript/Lecture09_Micro2.pdf
    Get the homogeenous steady state s_e(v) from IDM parameters (same cars)
    Derive macroscopic FD out of s_e(\rho)
    '''
    v0, s0, T, a, b, l = theta # l is car length, must be homogeneous fleet

    v_arr = np.linspace(0, v0-0.01, 100) # m/s   
    s_e_arr = (s0+v_arr*T) / (np.sqrt(1-(v_arr/v0)**4))
    rho_arr = 1/(s_e_arr+l)
    q_arr = v_arr * rho_arr

    return v_arr, s_e_arr, rho_arr, q_arr

def plot_multiple_idm_fd(thetas, legends=None):
    '''
    thetas is a list of theta's
    '''
    fig, axs = plt.subplots(1,3, figsize=(15,7))
    if legends is None:
        legends = ['' for _ in range(len(thetas))]
    for i, theta in enumerate(thetas):
        v_arr, s_e_arr, rho_arr, q_arr = idm_fd(theta)
        axs[0].plot(s_e_arr, v_arr, label=legends[i])
        axs[1].plot(rho_arr*1000, v_arr)
        axs[2].plot(rho_arr*1000, q_arr*3600 )

    axs[0].set_xlabel('Gap $s$ [m]')
    axs[0].set_ylabel('$v_e$ [m/s]')
    axs[0].set_xlim(left=0)
    axs[0].set_ylim(bottom=0)

    axs[1].set_xlabel('Density $\\rho$ [veh/km]')
    axs[1].set_ylabel('$v_e$ [m/s]')
    axs[1].set_xlim(left=0)
    axs[1].set_ylim(bottom=0)

    axs[0].legend(loc='upper left')
    axs[2].set_xlabel('Density $\\rho$ [veh/km]')
    axs[2].set_ylabel('Flow $q$ [veh/hr]')
    axs[2].set_xlim(left=0)
    axs[2].set_ylim(bottom=0)
    
    # axs[2].set_ylim([0, 3200])
    # plt.tight_layout()
    plt.show()
    return

def calc_ss_speed(rho, s0, tau, l):
    '''
    calculate steady state speed given rho (veh/km)
    '''
    s_e = 1000/rho # equilibrium spacing
    gap_e = s_e - s0 - l
    v_e = gap_e / tau
    print(f"steady state speed: {v_e}")
    return v_e


def get_detector_data(xml_file):
    '''
    plot the flow/density/speed relationship from xml_file (.out.xml)
    v, rho, q are background equilibrium macro quantities, derived from IDM parameters
    '''
    try:
        tree = ET.parse(xml_file)
    except:
        with open(xml_file, 'a') as file:
            file.write("</detector>" + '\n')

    tree = ET.parse(xml_file)
    root = tree.getroot()

    data = {}
    for interval in root.findall('interval'):
        id_value = interval.get('id')
        occupancy = float(interval.get('occupancy'))
        flow = float(interval.get('flow'))
        speed = float(interval.get('speed'))
        if speed == -1:
            continue
        interval_time = float(interval.get('end')) - float(interval.get('begin'))
        nVeh = float(interval.get('nVehContrib'))
        # vLen = float(interval.get('length'))
        # convert occupancy to density
        # occupancy_fraction = occupancy / 100
        effective_length = speed * interval_time
        density = nVeh / (effective_length / 1000) # vehcles per km

        if id_value not in data:
            data[id_value] = defaultdict(list)
        
        data[id_value]['occupancy'].append(occupancy)
        data[id_value]['flow'].append(flow)
        data[id_value]['speed'].append(speed)
        data[id_value]['density'].append(density)
    
    return data


def plot_detector_data(xml_file, idm_param, initial_val=None):
    '''
    Overlay FD of idm_param with loop detector data from simulation (det_data)
    '''
    l = idm_param[-1]
    v_arr, s_e_arr, rho_arr, q_arr = idm_fd(idm_param)
    det_data = get_detector_data(xml_file)
    

    fig, axs = plt.subplots(2,1, figsize=(4,7))

    # plot background FD
    axs[0].plot(rho_arr*1000, v_arr, label="IDM FD")
    axs[1].plot(rho_arr*1000, q_arr*3600, label="IDM FD" )
    

    axs[0].set_xlabel('Density $\\rho$ [veh/km]')
    axs[0].set_ylabel('$v_e$ [m/s]')
    
    axs[1].set_xlabel('Density $\\rho$ [veh/km]')
    axs[1].set_ylabel('Flow $q$ [veh/hr]')
    
    # plt.tight_layout()

    # plot detector data
    for i, (id_value, values) in enumerate(det_data.items()):
        print(i)
        axs[0].scatter(values['density'], values['speed'], label="loop detector" if i==0 else "")
        axs[1].scatter(values['density'], values['flow'], label="loop detector" if i==0 else "")

    # plot initial simulation values
    if initial_val is not None:
        axs[0].scatter(initial_val[0], initial_val[1], label="initial")
        axs[1].scatter(initial_val[0], initial_val[0]*initial_val[1]*3.6, label="initial")
    
    axs[0].set_xlim(left=0)
    axs[0].set_ylim(bottom=0)
    axs[1].set_xlim(left=0)
    axs[1].set_ylim(bottom=0)

    axs[0].legend()
    axs[1].legend()
    plt.tight_layout()
    plt.show()



    # plt.figure(figsize=(10, 6))
    # for id_value, values in data.items():
    #     plt.scatter(values['occupancy'], values['flow'], label=id_value)

    # plt.xlabel('Occupancy (% of the time a vehicle was at the detector)')
    # plt.ylabel('Flow (#vehicles/hour)')
    # plt.title('Detector Data')
    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    print("not implemented")