import numpy as np
import os
import sys
import pickle
from sys import platform
from math import floor, ceil
import numpy as np
import pandas as pd
import json

# ================
# initialization checed
# need to check get state
# ================


if platform == "linux" or platform == "linux2":
    # this is linux
    if os.path.exists('/headless/sumo'):
        os.environ['SUMO_HOME'] = '/headless/sumo'
    else:
        os.environ['SUMO_HOME'] = '/usr/share/sumo'
    try:
        import traci
        import traci.constants as tc
    except ImportError:
        if "SUMO_HOME" in os.environ:
            print(os.path.join(os.environ["SUMO_HOME"], "tools"))
            sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
            import traci
            import traci.constants as tc
        else:
            raise EnvironmentError("Please set SUMO_HOME environment variable or install traci as python module!")

elif platform == "win32":
    os.environ['SUMO_HOME'] = 'D:\\software\\sumo-0.32.0'

    try:
        import traci
        import traci.constants as tc
    except ImportError:
        if "SUMO_HOME" in os.environ:
            print(os.path.join(os.environ["SUMO_HOME"], "tools"))
            sys.path.append(
                os.path.join(os.environ["SUMO_HOME"], "tools")
            )
            import traci
            import traci.constants as tc
        else:
            raise EnvironmentError("Please set SUMO_HOME environment variable or install traci as python module!")

elif platform =='darwin':
    os.environ['SUMO_HOME'] = "/Users/{0}/sumo/".format(os.environ.get('USER'))
    print(os.environ['SUMO_HOME'])
    try:
        import traci
        import traci.constants as tc
    except ImportError:
        if "SUMO_HOME" in os.environ:
            print(os.path.join(os.environ["SUMO_HOME"], "tools"))
            sys.path.append(
                os.path.join(os.environ["SUMO_HOME"], "tools")
            )
            import traci
            import traci.constants as tc
        else:
            raise EnvironmentError("Please set SUMO_HOME environment variable or install traci as python module!")

else:
    sys.exit("platform error")




def get_traci_constant_mapping(constant_str):
    return getattr(tc, constant_str)

class Intersection:

    def __init__(self, light_id, list_vehicle_variables_to_sub, light_id_dict):

        '''
        still need to automate generation
        '''
        # TODO Auto generated

        self.node_light = "node{0}".format(light_id)
        self.list_vehicle_variables_to_sub = list_vehicle_variables_to_sub


        # grid settings
        self.length_lane = 300
        self.length_terminal = 50
        self.length_grid = 5
        self.num_grid = int(self.length_lane//self.length_grid)

        # generate all lanes
        self.list_entering_lanes = light_id_dict['entering_lanes']
        self.list_exiting_lanes = light_id_dict['leaving_lanes']

        self.list_lanes = self.list_entering_lanes + self.list_exiting_lanes
        self.adjacency_row = light_id_dict['adjacency_row']

        # generate signals # change
        self.list_phases = light_id_dict["phases"]

        self.all_yellow_phase_str = "y"*len(light_id_dict["phases"][0])
        self.all_red_phase_str = "r"*len(light_id_dict["phases"][0])
        self.all_yellow_phase_index = -1
        self.all_red_phase_index = -2

        # initialization

        # -1: all yellow, -2: all red, -3: none
        self.current_phase_index = 0
        self.previous_phase_index = 0
        self.next_phase_to_set_index = None
        self.current_phase_duration = -1
        self.all_red_flag = False
        self.all_yellow_flag = False
        self.flicker = 0

        self.dic_lane_sub_current_step = None
        self.dic_lane_sub_previous_step = None
        self.dic_vehicle_sub_current_step = None
        self.dic_vehicle_sub_previous_step = None
        self.list_vehicles_current_step = []
        self.list_vehicles_previous_step = []

        self.dic_vehicle_min_speed = {}  # this second
        self.dic_vehicle_arrive_leave_time = dict() # cumulative

        self.dic_feature = {} # this second

    def set_signal(self, action, action_pattern, yellow_time, all_red_time):

        if self.all_yellow_flag:
            # in yellow phase
            self.flicker = 0
            if self.current_phase_duration >= yellow_time: # yellow time reached
                self.current_phase_index = self.next_phase_to_set_index
                traci.trafficlights.setRedYellowGreenState(
                    self.node_light, self.list_phases[self.current_phase_index])
                self.all_yellow_flag = False
            else:
                pass
        else:
            # determine phase
            if action_pattern == "switch": # switch by order
                if action == 0: # keep the phase
                    self.next_phase_to_set_index = self.current_phase_index
                elif action == 1: # change to the next phase
                    self.next_phase_to_set_index = (self.current_phase_index + 1) % len(self.list_phases)
                else:
                    sys.exit("action not recognized\n action must be 0 or 1")

            elif action_pattern == "set": # set to certain phase
                self.next_phase_to_set_index = action

            # set phase
            if self.current_phase_index == self.next_phase_to_set_index: # the light phase keeps unchanged
                pass
            else: # the light phase needs to change
                # change to yellow first, and activate the counter and flag
                traci.trafficlights.setRedYellowGreenState(
                    self.node_light, self.all_yellow_phase_str)
                self.current_phase_index = self.all_yellow_phase_index
                self.all_yellow_flag = True
                self.flicker = 1

    def update_previous_measurements(self):

        self.previous_phase_index = self.current_phase_index
        self.dic_lane_sub_previous_step = self.dic_lane_sub_current_step
        self.dic_vehicle_sub_previous_step = self.dic_vehicle_sub_current_step
        self.list_vehicles_previous_step = self.list_vehicles_current_step

    def update_current_measurements(self):
        ## need change, debug in seeing format

        if self.current_phase_index == self.previous_phase_index:
            self.current_phase_duration += 1
        else:
            self.current_phase_duration = 1

        # ====== lane level observations =======

        self.dic_lane_sub_current_step = {lane: traci.lane.getSubscriptionResults(lane) for lane in self.list_lanes}

        # ====== vehicle level observations =======

        # get vehicle list
        # self.list_vehicles_current_step = traci.vehicle.getIDList() # TODO this should be get IDlist on lane, including inner lanes
        self.list_vehicles_current_step = self._update_current_intersection_vehicle_ids()

        list_vehicles_new_arrive = list(set(self.list_vehicles_current_step) - set(self.list_vehicles_previous_step))
        list_vehicles_new_left = list(set(self.list_vehicles_previous_step) - set(self.list_vehicles_current_step))
        list_vehicles_new_left_entering_lane_by_lane = self._update_leave_entering_approach_vehicle()
        list_vehicles_new_left_entering_lane = []
        for l in list_vehicles_new_left_entering_lane_by_lane:
            list_vehicles_new_left_entering_lane += l

        # update subscriptions
        for vehicle in list_vehicles_new_arrive:
            traci.vehicle.subscribe(vehicle, [getattr(tc, var) for var in self.list_vehicle_variables_to_sub])

        # vehicle level observations
        self.dic_vehicle_sub_current_step = {vehicle: traci.vehicle.getSubscriptionResults(vehicle) for vehicle in self.list_vehicles_current_step}

        # update vehicle arrive and left time
        self._update_arrive_time(list_vehicles_new_arrive)
        self._update_left_time(list_vehicles_new_left_entering_lane)

        # update vehicle minimum speed in history
        self._update_vehicle_min_speed()

        # update feature
        self._update_feature()

    # ================= update current step measurements ======================
    def _update_current_intersection_vehicle_ids(self):
        vehicle_ids = []
        list_all_lanes = set(np.array(traci.trafficlights.getControlledLinks(self.node_light)).flatten().tolist())
        for lane_id in list_all_lanes:
            vehicle_ids = vehicle_ids+traci.lane.getLastStepVehicleIDs(lane_id)
        return vehicle_ids

    def _update_leave_entering_approach_vehicle(self):

        list_entering_lane_vehicle_left = []

        # update vehicles leaving entering lane
        if self.dic_lane_sub_previous_step is None:
            for lane in self.list_entering_lanes:
                list_entering_lane_vehicle_left.append([])
        else:
            # TODO THIS IS a bug , when changing lanes, this would be wrong
            last_step_vehicle_id_list = []
            current_step_vehilce_id_list = []
            for lane in self.list_entering_lanes:
                last_step_vehicle_id_list.extend(self.dic_lane_sub_previous_step[lane][get_traci_constant_mapping("LAST_STEP_VEHICLE_ID_LIST")])
                current_step_vehilce_id_list.extend(self.dic_lane_sub_current_step[lane][get_traci_constant_mapping("LAST_STEP_VEHICLE_ID_LIST")])

            list_entering_lane_vehicle_left.append(
                    list(set(last_step_vehicle_id_list) - set(current_step_vehilce_id_list))
                )
        return list_entering_lane_vehicle_left

    def _update_arrive_time(self, list_vehicles_arrive):

        ts = self.get_current_time()
        # get dic vehicle enter leave time
        # print('list_vehicles_arrive:',list_vehicles_arrive)
        # print('self.dic_vehicle_arrive_leave_time:',self.dic_vehicle_arrive_leave_time)
        for vehicle in list_vehicles_arrive:
            if vehicle not in self.dic_vehicle_arrive_leave_time:
                self.dic_vehicle_arrive_leave_time[vehicle] = \
                    {"enter_time": ts, "leave_time": np.nan}
            # else:
            #     print("vehicle already exists!")
                # sys.exit(-1)

    def _update_left_time(self, list_vehicles_left):

        ts = self.get_current_time()
        # update the time for vehicle to leave entering lane
        for vehicle in list_vehicles_left:
            try:
                self.dic_vehicle_arrive_leave_time[vehicle]["leave_time"] = ts

            except KeyError:
                print("vehicle not recorded when entering")
                sys.exit(-1)

    def _update_vehicle_min_speed(self):
        '''
        record the minimum speed of one vehicle so far
        :return:
        '''
        dic_result = {}
        for vec_id, vec_var in self.dic_vehicle_sub_current_step.items():
            speed = vec_var[get_traci_constant_mapping("VAR_SPEED")]
            if vec_id in self.dic_vehicle_min_speed: # this vehicle appeared in previous time stamps:
                dic_result[vec_id] = min(speed, self.dic_vehicle_min_speed[vec_id])
            else:
                dic_result[vec_id] = speed
        self.dic_vehicle_min_speed = dic_result

    def _update_feature(self):

        dic_feature = dict()

        dic_feature["cur_phase"] = [self.current_phase_index]
        dic_feature["time_this_phase"] = [self.current_phase_duration]
        dic_feature["vehicle_position_img"] = None #self._get_lane_vehicle_position(self.list_entering_lanes)
        dic_feature["vehicle_speed_img"] = None #self._get_lane_vehicle_speed(self.list_entering_lanes)
        dic_feature["vehicle_acceleration_img"] = None
        dic_feature["vehicle_waiting_time_img"] = None #self._get_lane_vehicle_accumulated_waiting_time(self.list_entering_lanes)

        dic_feature["lane_num_vehicle"] = self._get_lane_num_vehicle(self.list_entering_lanes)

        dic_feature["coming_vehicle"] =self._get_coming_vehicles(self.list_entering_lanes)
        dic_feature["leaving_vehicle"] = self._get_leaving_vehicles(self.list_exiting_lanes)

        dic_feature["lane_num_vehicle_been_stopped_thres01"] = self._get_lane_num_vehicle_been_stopped(0.1, self.list_entering_lanes)
        dic_feature["lane_num_vehicle_been_stopped_thres1"] = self._get_lane_num_vehicle_been_stopped(1, self.list_entering_lanes)
        dic_feature["lane_queue_length"] = self._get_lane_queue_length(self.list_entering_lanes)
        dic_feature["lane_num_vehicle_left"] = None
        dic_feature["lane_sum_duration_vehicle_left"] = None
        dic_feature["lane_sum_waiting_time"] = self._get_lane_sum_waiting_time(self.list_entering_lanes)
        dic_feature["terminal"] = None

        dic_feature["pressure"] = self._get_pressure()

        dic_feature['adjacency_matrix'] = self._get_adjacency_row()


        self.dic_feature = dic_feature

    # ================= calculate features from current observations ======================

    def _get_pressure(self):
        pressure = 0
        all_enter_car_queue = 0
        for lane in self.list_entering_lanes:
            position_list = []
            for k in self.dic_lane_sub_current_step[lane][get_traci_constant_mapping('LAST_STEP_VEHICLE_ID_LIST')]:
                position_list.append(self.dic_vehicle_sub_current_step[k][get_traci_constant_mapping('VAR_LANEPOSITION')])
            if True in list(np.diff(position_list) < 10):
                all_enter_car_queue += np.sum(
                    len(list(np.diff(position_list) < 10)) - list(np.diff(position_list) < 10).index(True) + 1)
            elif True in list(np.array(position_list) > (traci.lane.getLength(lane) - 2)):
                all_enter_car_queue += 1

        all_leaving_car_queue = 0
        for lane in self.list_exiting_lanes:
            position_list = []
            for k in self.dic_lane_sub_current_step[lane][get_traci_constant_mapping('LAST_STEP_VEHICLE_ID_LIST')]:
                position_list.append(self.dic_vehicle_sub_current_step[k][get_traci_constant_mapping('VAR_LANEPOSITION')])
            if True in list(np.diff(position_list) < 10):
                all_leaving_car_queue += np.sum(
                    len(list(np.diff(position_list) < 10)) - list(np.diff(position_list) < 10).index(True) + 1)
            elif True in list(np.array(position_list) > (traci.lane.getLength(lane) - 2)):
                all_leaving_car_queue += 1

        p = all_enter_car_queue - all_leaving_car_queue

        if p < 0:
            p = -p

        return p

    def _get_lane_queue_length(self, list_lanes):
        '''
        queue length for each lane
        '''
        return [self.dic_lane_sub_current_step[lane][get_traci_constant_mapping("LAST_STEP_VEHICLE_HALTING_NUMBER")]
                for lane in list_lanes]

    def _get_lane_num_vehicle(self, list_lanes):
        '''
        vehicle number for each lane
        '''
        return [self.dic_lane_sub_current_step[lane][get_traci_constant_mapping("LAST_STEP_VEHICLE_NUMBER")]
                for lane in list_lanes]


    def lane_position_mapper(self, lane_pos, bins):
        lane_pos_np = np.array(lane_pos)
        digitized = np.digitize(lane_pos_np, bins)
        position_counter = [len(lane_pos_np[digitized == i]) for i in range(1, len(bins))]
        return position_counter


    def _get_coming_vehicles(self, list_lanes):
        '''
        vehicle number for each lane
        '''
        coming_vehicles_tracker = []

        for lane in list_lanes:
            position_list = []
            for k in self.dic_lane_sub_current_step[lane][get_traci_constant_mapping('LAST_STEP_VEHICLE_ID_LIST')]:
                position_list.append(self.dic_vehicle_sub_current_step[k][get_traci_constant_mapping('VAR_LANEPOSITION')])
            #TODO subscribe lane length
            lane_length = traci.lane.getLength(lane)
            bins = np.linspace(0, lane_length, 4).tolist()

            coming_distribution = self.lane_position_mapper(position_list, bins)
            coming_vehicles_tracker.extend(coming_distribution)

        coming_vehicles_tracker_aggregate = [0] * 12

        for i in range(3):
            coming_vehicles_tracker_aggregate[i] = coming_vehicles_tracker[i] + \
                                                   coming_vehicles_tracker[i + 3] + \
                                                   coming_vehicles_tracker[i + 6]
        for i in range(3, 6):
            coming_vehicles_tracker_aggregate[i] = coming_vehicles_tracker[i + 6] + \
                                                   coming_vehicles_tracker[i + 9] + \
                                                   coming_vehicles_tracker[i + 12]
        for i in range(6, 9):
            coming_vehicles_tracker_aggregate[i] = coming_vehicles_tracker[i + 12] + \
                                                   coming_vehicles_tracker[i + 15] + \
                                                   coming_vehicles_tracker[i + 18]
        for i in range(9, 12):
            coming_vehicles_tracker_aggregate[i] = coming_vehicles_tracker[i + 18] + \
                                                   coming_vehicles_tracker[i + 21] + \
                                                   coming_vehicles_tracker[i + 24]

        return coming_vehicles_tracker_aggregate


    def _get_leaving_vehicles(self, list_lanes):
        '''
        vehicle number for each lane
        '''
        leaving_vehicles_tracker = []
        for lane in list_lanes:
            position_list = []
            for k in self.dic_lane_sub_current_step[lane][get_traci_constant_mapping('LAST_STEP_VEHICLE_ID_LIST')]:
                position_list.append(self.dic_vehicle_sub_current_step[k][get_traci_constant_mapping('VAR_LANEPOSITION')])
            # vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
            # leaving_position = []
            # for vehicle_id in vehicle_ids:
            #     lane_pos = traci.vehicle.getLanePosition(vehicle_id)
            #     leaving_position.append(lane_pos)
            lane_length = traci.lane.getLength(lane)
            bins = np.linspace(0, lane_length, 4).tolist()
            leaving_distribution = self.lane_position_mapper(position_list, bins)
            leaving_vehicles_tracker.extend(leaving_distribution)
        leaving_vehicles_tracker_aggregate = [0] * 12
        for i in range(3):
            leaving_vehicles_tracker_aggregate[i] = leaving_vehicles_tracker[i] + \
                                                    leaving_vehicles_tracker[i + 3] + \
                                                    leaving_vehicles_tracker[i + 6]
        for i in range(3, 6):
            leaving_vehicles_tracker_aggregate[i] = leaving_vehicles_tracker[i + 6] + \
                                                    leaving_vehicles_tracker[i + 9] + \
                                                    leaving_vehicles_tracker[i + 12]
        for i in range(6, 9):
            leaving_vehicles_tracker_aggregate[i] = leaving_vehicles_tracker[i + 12] + \
                                                    leaving_vehicles_tracker[i + 15] + \
                                                    leaving_vehicles_tracker[i + 18]
        for i in range(9, 12):
            leaving_vehicles_tracker_aggregate[i] = leaving_vehicles_tracker[i + 18] + \
                                                    leaving_vehicles_tracker[i + 21] + \
                                                    leaving_vehicles_tracker[i + 24]

        return leaving_vehicles_tracker_aggregate


    def _get_lane_sum_waiting_time(self, list_lanes):
        '''
        waiting time for each lane
        '''
        return [self.dic_lane_sub_current_step[lane][get_traci_constant_mapping("VAR_WAITING_TIME")]
                for lane in list_lanes]

    def _get_lane_list_vehicle_left(self, list_lanes):
        '''
        get list of vehicles left at each lane
        ####### need to check
        '''

        return None

    def _get_lane_num_vehicle_left(self, list_lanes):

        list_lane_vehicle_left = self._get_lane_list_vehicle_left(list_lanes)
        list_lane_num_vehicle_left = [len(lane_vehicle_left) for lane_vehicle_left in list_lane_vehicle_left]
        return list_lane_num_vehicle_left

    def _get_lane_sum_duration_vehicle_left(self, list_lanes):

        ## not implemented error
        raise NotImplementedError

    def _get_lane_num_vehicle_been_stopped(self, thres, list_lanes):

        list_num_of_vec_ever_stopped = []
        for lane in list_lanes:
            cnt_vec = 0
            list_vec_id = self.dic_lane_sub_current_step[lane][get_traci_constant_mapping("LAST_STEP_VEHICLE_ID_LIST")]
            for vec in list_vec_id:
                if self.dic_vehicle_min_speed[vec] < thres:
                    cnt_vec += 1
            list_num_of_vec_ever_stopped.append(cnt_vec)

        return list_num_of_vec_ever_stopped

    def _get_position_grid_along_lane(self, vec):
        pos = int(self.dic_vehicle_sub_current_step[vec][get_traci_constant_mapping("VAR_LANEPOSITION")])
        return min(pos//self.length_grid, self.num_grid)


    def _get_lane_vehicle_position(self, list_lanes):

        list_lane_vector = []
        for lane in list_lanes:
            lane_vector = np.zeros(self.num_grid)
            list_vec_id = self.dic_lane_sub_current_step[lane][get_traci_constant_mapping("LAST_STEP_VEHICLE_ID_LIST")]
            for vec in list_vec_id:
                pos_grid = self._get_position_grid_along_lane(vec)
                lane_vector[pos_grid] = 1
            list_lane_vector.append(lane_vector)
        return np.array(list_lane_vector)

    def _get_lane_vehicle_speed(self, list_lanes):

        list_lane_vector = []
        for lane in list_lanes:
            lane_vector = np.full(self.num_grid, fill_value=np.nan)
            list_vec_id = self.dic_lane_sub_current_step[lane][get_traci_constant_mapping("LAST_STEP_VEHICLE_ID_LIST")]
            for vec in list_vec_id:
                pos_grid = self._get_position_grid_along_lane(vec)
                lane_vector[pos_grid] = self.dic_vehicle_sub_current_step[vec][get_traci_constant_mapping("VAR_SPEED")]
            list_lane_vector.append(lane_vector)
        return np.array(list_lane_vector)

    def _get_lane_vehicle_accumulated_waiting_time(self, list_lanes):

        list_lane_vector = []
        for lane in list_lanes:
            lane_vector = np.full(self.num_grid, fill_value=np.nan)
            list_vec_id = self.dic_lane_sub_current_step[lane][get_traci_constant_mapping("LAST_STEP_VEHICLE_ID_LIST")]
            for vec in list_vec_id:
                pos_grid = self._get_position_grid_along_lane(vec)
                lane_vector[pos_grid] = self.dic_vehicle_sub_current_step[vec][get_traci_constant_mapping("VAR_ACCUMULATED_WAITING_TIME")]
            list_lane_vector.append(lane_vector)
        return np.array(list_lane_vector)


    def _get_adjacency_row(self):
        return self.adjacency_row

    def _lane_position_mapper(self, lane_pos, bins):
        lane_pos_np = np.array(lane_pos)
        digitized = np.digitize(lane_pos_np, bins)
        position_counter = [len(lane_pos_np[digitized == i]) for i in range(1, len(bins))]
        return position_counter


    def _get_coming_vehicles(self, list_lanes):
        '''
        vehicle number for each lane
        '''
        coming_vehicles_tracker = []

        for lane in list_lanes:
            position_list = []
            for k in self.dic_lane_sub_current_step[lane][get_traci_constant_mapping('LAST_STEP_VEHICLE_ID_LIST')]:
                try:
                    position_list.append(self.dic_vehicle_sub_current_step[k][get_traci_constant_mapping('VAR_LANEPOSITION')])
                except:
                    pass
            #TODO subscribe lane length
            lane_length = self.length_lane
            # lane_length = traci.lane.getLength(lane)
            bins = np.linspace(0, lane_length, 4).tolist()

            coming_distribution = self._lane_position_mapper(position_list, bins)
            coming_vehicles_tracker.extend(coming_distribution)

        return coming_vehicles_tracker

    def _get_leaving_vehicles(self, list_lanes):
        '''
        vehicle number for each lane
        '''
        leaving_vehicles_tracker = []
        for lane in list_lanes:
            position_list = []
            for k in self.dic_lane_sub_current_step[lane][get_traci_constant_mapping('LAST_STEP_VEHICLE_ID_LIST')]:
                position_list.append(self.dic_vehicle_sub_current_step[k][get_traci_constant_mapping('VAR_LANEPOSITION')])
            lane_length = self.length_lane
            bins = np.linspace(0, lane_length, 4).tolist()
            leaving_distribution = self._lane_position_mapper(position_list, bins)
            leaving_vehicles_tracker.extend(leaving_distribution)

        return leaving_vehicles_tracker


    def _get_pressure(self):
        pressure = 0
        all_enter_car_queue = 0
        for lane in self.list_entering_lanes:
            position_list = []
            for k in self.dic_lane_sub_current_step[lane][get_traci_constant_mapping('LAST_STEP_VEHICLE_ID_LIST')]:
                position_list.append(self.dic_vehicle_sub_current_step[k][get_traci_constant_mapping('VAR_LANEPOSITION')])
            if True in list(np.diff(position_list) < 10):
                all_enter_car_queue += np.sum(
                    len(list(np.diff(position_list) < 10)) - list(np.diff(position_list) < 10).index(True) + 1)
            elif True in list(np.array(position_list) > (traci.lane.getLength(lane) - 2)):
                all_enter_car_queue += 1

        all_leaving_car_queue = 0
        for lane in self.list_exiting_lanes:
            position_list = []
            for k in self.dic_lane_sub_current_step[lane][get_traci_constant_mapping('LAST_STEP_VEHICLE_ID_LIST')]:
                position_list.append(self.dic_vehicle_sub_current_step[k][get_traci_constant_mapping('VAR_LANEPOSITION')])
            if True in list(np.diff(position_list) < 10):
                all_leaving_car_queue += np.sum(
                    len(list(np.diff(position_list) < 10)) - list(np.diff(position_list) < 10).index(True) + 1)
            elif True in list(np.array(position_list) > (traci.lane.getLength(lane) - 2)):
                all_leaving_car_queue += 1

        p = all_enter_car_queue - all_leaving_car_queue

        if p < 0:
            p = -p

        return p


    # ================= get functions from outside ======================

    def get_current_time(self):
        return traci.simulation.getCurrentTime() / 1000

    def get_dic_vehicle_arrive_leave_time(self):

        return self.dic_vehicle_arrive_leave_time

    def get_feature(self):

        return self.dic_feature

    def get_state(self, list_state_features):

        dic_state = {state_feature_name: self.dic_feature[state_feature_name] for state_feature_name in list_state_features}

        return dic_state

    def get_reward(self, dic_reward_info):

        dic_reward = dict()
        dic_reward["flickering"] = None
        dic_reward["sum_lane_queue_length"] = None
        dic_reward["sum_lane_wait_time"] = None
        dic_reward["sum_lane_num_vehicle_left"] = None
        dic_reward["sum_duration_vehicle_left"] = None
        dic_reward["sum_num_vehicle_been_stopped_thres01"] = None
        dic_reward["sum_num_vehicle_been_stopped_thres1"] = np.sum(self.dic_feature["lane_num_vehicle_been_stopped_thres1"])

        dic_reward['pressure'] = self.dic_feature["pressure"]

        reward = 0
        for r in dic_reward_info:
            if dic_reward_info[r] != 0:
                reward += dic_reward_info[r] * dic_reward[r]
        return reward

    def _get_vehicle_info(self, veh_id):
        try:
            pos = self.dic_vehicle_sub_current_step[veh_id][get_traci_constant_mapping("VAR_LANEPOSITION")]
            speed = self.dic_vehicle_sub_current_step[veh_id][get_traci_constant_mapping("VAR_SPEED")]
            return pos, speed
        except:
            return None, None



class SumoEnv:

    # add more variables here if you need more measurements
    LIST_LANE_VARIABLES_TO_SUB = [
        "LAST_STEP_VEHICLE_NUMBER",
        "LAST_STEP_VEHICLE_ID_LIST",
        "LAST_STEP_VEHICLE_HALTING_NUMBER",
        "VAR_WAITING_TIME",

    ]

    # add more variables here if you need more measurements
    LIST_VEHICLE_VARIABLES_TO_SUB = [
        "VAR_POSITION",
        "VAR_SPEED",
        # "VAR_ACCELERATION",
        # "POSITION_LON_LAT",
        "VAR_WAITING_TIME",
        "VAR_ACCUMULATED_WAITING_TIME",
        # "VAR_LANEPOSITION_LAT",
        "VAR_LANEPOSITION",
    ]

    def _get_sumo_cmd(self): 

        if platform == "linux" or platform == "linux2":
            if os.path.exists('/headless/sumo/bin/'):
                sumo_binary = r"/headless/sumo/bin/sumo-gui"
                sumo_binary_nogui = r"/headless/sumo/bin/sumo"
            else:
                sumo_binary = r"/usr/bin/sumo-gui"
                sumo_binary_nogui = r"/usr/bin/sumo"
        elif platform == "darwin":
            sumo_binary = r"/opt/local/bin/sumo-gui"
            sumo_binary_nogui = r"/opt/local/bin/sumo"
        elif platform == "win32":
            sumo_binary = r'D:\\software\\sumo-0.32.0\\bin\\sumo-gui.exe'
            sumo_binary_nogui = r'D:\\software\\sumo-0.32.0\\bin\\sumo.exe'
        else:
            sys.exit("platform error")

        real_path_to_sumo_files = os.path.join(os.path.split(os.path.realpath(__file__))[0], self.path_to_work_directory, "cross.sumocfg")

        # path = self.dic_traffic_env_conf[""]

        sumo_cmd = [sumo_binary,
                   '-c',
                   r'{0}'.format(real_path_to_sumo_files),
                   "--step-length",
                   str(self.dic_traffic_env_conf["INTERVAL"]),
                   "--full-output",
                   os.path.join(self.path_to_work_directory,'sumo_replay.xml')
                    ]

        sumo_cmd_nogui = [sumo_binary_nogui,
                         '-c',
                         r'{0}'.format(real_path_to_sumo_files),
                         "--step-length",
                         str(self.dic_traffic_env_conf["INTERVAL"])
                          ]

        if self.dic_traffic_env_conf["IF_GUI"]:
            return sumo_cmd
        else:
            return sumo_cmd_nogui

    @staticmethod
    def _coordinate_sequence(list_coord_str):
        import re
        list_coordinate =[ re.split(r'[ ,]',lane_str) for lane_str in list_coord_str]
        # x coordinate
        x_all = np.array(list_coordinate, dtype=float)[:,[0,2]]
        west = np.int(np.argmin(x_all)/2)

        y_all = np.array(list_coordinate, dtype=float)[:,[1,3]]

        south = np.int(np.argmin(y_all)/2)

        east = np.int(np.argmax(x_all)/2)
        north = np.int(np.argmax(y_all)/2)

        list_coord_sort=[west,north,east,south]
        return list_coord_sort

    @staticmethod
    def _sort_lane_id_by_sequence(ids,sequence=[2,3,0,1]):
        result = []
        for i in sequence:
            result.extend(ids[i*3: i*3+3])
        return result

    @staticmethod
    def get_actual_lane_id(lane_id_list):
        actual_lane_id_list = []
        for lane_id in lane_id_list:
            if not lane_id.startswith(":"):
                actual_lane_id_list.append(lane_id)
        return actual_lane_id_list



    def _infastructure_extraction(self, sumocfg_file):
        import xml.etree.ElementTree
        e = xml.etree.ElementTree.parse(sumocfg_file).getroot()
        network_file_name = e.find('input/net-file').attrib['value']
        network_file = os.path.join(os.path.split(sumocfg_file)[0],network_file_name)
        net = xml.etree.ElementTree.parse(network_file).getroot()

        additional_file_name = e.find('input/additional-files').attrib['value']
        additional_file = os.path.join(os.path.split(sumocfg_file)[0],additional_file_name)
        phase_xml = xml.etree.ElementTree.parse(additional_file).getroot()

        traffic_light_node_dict = {}
        for tl in net.findall("tlLogic"):
            if tl.attrib['id'] not in traffic_light_node_dict.keys():
                node_id = tl.attrib['id']
                traffic_light_node_dict[node_id] = {'leaving_lanes': [], 'entering_lanes': [],
                                                    'leaving_lanes_pos': [], 'entering_lanes_pos': [],
                                                    "total_inter_num": None,
                                                    'adjacency_row':None}
        total_inter_num = len(traffic_light_node_dict)
        for index, item in enumerate(traffic_light_node_dict):
            traffic_light_node_dict[item]['total_inter_num'] = total_inter_num

        # read the phases
        for tl in phase_xml.findall("tlLogic"):
            if tl.attrib['id'] in traffic_light_node_dict.keys():
                traffic_light_node_dict[tl.attrib['id']]["phases"] = [child.attrib["state"] for child in tl]

        for edge in net.findall("edge"):
            if not edge.attrib['id'].startswith(":"):
                if edge.attrib['from'] in traffic_light_node_dict.keys():
                    for child in edge:
                        traffic_light_node_dict[edge.attrib['from']]['leaving_lanes'].append(child.attrib['id'])
                        if child.attrib['index'] == "0":
                            traffic_light_node_dict[edge.attrib['from']]['leaving_lanes_pos'].append(child.attrib['shape'])
                if edge.attrib['to'] in traffic_light_node_dict.keys():
                    for child in edge:
                        traffic_light_node_dict[edge.attrib['to']]['entering_lanes'].append(child.attrib['id'])
                        if child.attrib['index'] == "0":
                            traffic_light_node_dict[edge.attrib['to']]['entering_lanes_pos'].append(child.attrib['shape'])

        for junction in net.findall("junction"):
            if junction.attrib['id'] in traffic_light_node_dict.keys():
                traffic_light_node_dict[junction.attrib['id']]['location'] = {'x': float(junction.attrib['x']),
                                                                              'y': float(junction.attrib['y'])}

        for k,v in traffic_light_node_dict.items():
            entering_sequence = SumoEnv._coordinate_sequence(v["entering_lanes_pos"])
            v["entering_lanes"] = SumoEnv._sort_lane_id_by_sequence(v["entering_lanes"] ,sequence=entering_sequence)
            leaving_sequence = SumoEnv._coordinate_sequence(v["leaving_lanes_pos"])
            v["leaving_lanes"] = SumoEnv._sort_lane_id_by_sequence(v["leaving_lanes"] ,sequence=leaving_sequence)

        #get the adjacency row info

        top_k = self.dic_traffic_env_conf["TOP_K_ADJACENCY"]

        for i in range(total_inter_num):
            location_1 = traffic_light_node_dict["node{0}".format(i)]['location']
            # TODO return with Top K results
            row = np.array([0]*total_inter_num)
            for j in range(total_inter_num):
                location_2 = traffic_light_node_dict["node{0}".format(j)]['location']
                dist = SumoEnv._cal_distance(location_1,location_2)
                row[j] = dist
            if len(row) == top_k:
                adjacency_row_unsorted = np.argpartition(row, -1)[:top_k].tolist()
            elif len(row) > top_k:
                adjacency_row_unsorted = np.argpartition(row, top_k)[:top_k].tolist()
            else:
                adjacency_row_unsorted = list(range(total_inter_num))

            adjacency_row_unsorted.remove(i)
            traffic_light_node_dict["node{0}".format(i)]['adjacency_row'] = [i]+adjacency_row_unsorted

        # get lane length

        return traffic_light_node_dict

    @staticmethod
    def _cal_distance(loc_dict1, loc_dict2):
        a = np.array((loc_dict1['x'], loc_dict1['y']))
        b = np.array((loc_dict2['x'], loc_dict2['y']))
        return np.sqrt(np.sum((a-b)**2))


    def __init__(self, path_to_log, path_to_work_directory, dic_traffic_env_conf):

        self.path_to_log = path_to_log
        self.path_to_work_directory = path_to_work_directory
        self.dic_traffic_env_conf = dic_traffic_env_conf

        self.sumo_cmd_str = self._get_sumo_cmd()

        self.list_intersection = None
        self.list_inter_log = None
        self.list_lanes = None

        # check min action time
        if self.dic_traffic_env_conf["MIN_ACTION_TIME"] <= self.dic_traffic_env_conf["YELLOW_TIME"]:
            print ("MIN_ACTION_TIME should include YELLOW_TIME")
            pass
            #raise ValueError

        # touch new inter_{}.pkl (if exists, remove)
        for inter_ind in range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"]):
            path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
            f = open(path_to_log_file, "wb")
            f.close()

    def reset(self):

        # initialize intersections

        self.traffic_light_node_dict = self._infastructure_extraction(self.sumo_cmd_str[2])

        self.list_intersection = [Intersection(i, self.LIST_VEHICLE_VARIABLES_TO_SUB, self.traffic_light_node_dict['node{0}'.format(i)]) for i in range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"])]

        self.list_inter_log = [[] for i in range(len(self.list_intersection))]
        # get lanes list
        self.list_lanes = []
        for inter in self.list_intersection:
            self.list_lanes += inter.list_lanes
        self.list_lanes = np.unique(self.list_lanes).tolist()

        print ("start sumo")
        while True:
            try:
                traci.start(self.sumo_cmd_str)
                break
            except:
                continue
        print("succeed in start sumo")

        # start subscription
        for lane in self.list_lanes:
            traci.lane.subscribe(lane, [getattr(tc, var) for var in self.LIST_LANE_VARIABLES_TO_SUB])

        # get new measurements
        for inter in self.list_intersection:
            inter.update_current_measurements()

        state, done = self.get_state()

        return state

    @staticmethod
    def convert_dic_to_df(dic):
        list_df = []
        for key in dic:
            df = pd.Series(dic[key], name=key)
            list_df.append(df)
        return pd.DataFrame(list_df)

    def bulk_log(self):

        valid_flag = {}
        for inter_ind in range(len(self.list_intersection)):
            path_to_log_file = os.path.join(self.path_to_log, "vehicle_inter_{0}.csv".format(inter_ind))
            dic_vehicle = self.list_intersection[inter_ind].get_dic_vehicle_arrive_leave_time()
            df = self.convert_dic_to_df(dic_vehicle)
            if df.empty:
                df = pd.DataFrame({'vehicle_id':[],'enter_time':[],'leave_time':[]})
            # else:
            #     df['duration'] = df['leave_time'] - df['enter_time']
            df.to_csv(path_to_log_file, na_rep="nan")

            inter = self.list_intersection[inter_ind]
            feature = inter.get_feature()
            print(feature['lane_num_vehicle'])
            if max(feature['lane_num_vehicle']) > 30:
                valid_flag[inter_ind] = 0
            else:
                valid_flag[inter_ind] = 1
        json.dump(valid_flag, open(os.path.join(self.path_to_log, "valid_flag.json"), "w"))

        for inter_ind in range(len(self.list_inter_log)):
            path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
            f = open(path_to_log_file, "wb")
            pickle.dump(self.list_inter_log[inter_ind], f)
            f.close()

    def log_attention(self, attention_dict):
        path_to_log_file = os.path.join(self.path_to_log, "attention.pkl")
        f = open(path_to_log_file, "wb")
        pickle.dump(attention_dict, f)
        f.close()


    def end_sumo(self):
        traci.close()
        print("sumo process end")

    def get_current_time(self):
        return traci.simulation.getCurrentTime() / 1000

    def get_feature(self):

        list_feature = [inter.get_feature() for inter in self.list_intersection]
        return list_feature

    def get_state(self):

        list_state = [inter.get_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"]) for inter in self.list_intersection]
        done = self._check_episode_done(list_state)

        return list_state, done

    def get_reward(self):

        list_reward = [inter.get_reward(self.dic_traffic_env_conf["DIC_REWARD_INFO"]) for inter in self.list_intersection]

        return list_reward

    # def log(self, cur_time, before_action_feature, action):
    #
    #     for inter_ind in range(len(self.list_intersection)):
    #         path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
    #         f = open(path_to_log_file, "ab+")
    #         pickle.dump(
    #             {"time": cur_time,
    #              "state": before_action_feature[inter_ind],
    #              "action": action[inter_ind]}, f)
    #         f.close()

    def log(self, cur_time, before_action_feature, action):

        for inter_ind in range(len(self.list_intersection)):
            self.list_inter_log[inter_ind].append({"time": cur_time,
                                                    "state": before_action_feature[inter_ind],
                                                    "action": action[inter_ind]})

    def step(self, action,test_flag):

        list_action_in_sec = [action]
        list_action_in_sec_display = [action]
        for i in range(self.dic_traffic_env_conf["MIN_ACTION_TIME"]-1):
            if self.dic_traffic_env_conf["ACTION_PATTERN"] == "switch":
                list_action_in_sec.append(np.zeros_like(action).tolist())
            elif self.dic_traffic_env_conf["ACTION_PATTERN"] == "set":
                list_action_in_sec.append(np.copy(action).tolist())
            list_action_in_sec_display.append(np.full_like(action, fill_value=-1).tolist())


        average_reward_action_list = [0]*len(action)
        for i in range(self.dic_traffic_env_conf["MIN_ACTION_TIME"]):

            action_in_sec = list_action_in_sec[i]
            action_in_sec_display = list_action_in_sec_display[i]

            instant_time = self.get_current_time()

            before_action_feature = self.get_feature()
            state = self.get_state()

            if self.dic_traffic_env_conf["DEBUG"]:
                print("time: {0}, phase: {1}, time this phase: {2}, action: {3}".format(instant_time, before_action_feature[0]["cur_phase"], before_action_feature[0]["time_this_phase"], action_in_sec_display[0]))
            else:
                if i == 0:

                    print("time: {0}, phase: {1}, time this phase: {2}, action: {3}"
                                      .format(instant_time,
                                              str([before_action_feature[j]["cur_phase"] for j in range(len(before_action_feature))]),
                                              str([before_action_feature[j]["time_this_phase"] for j in range(len(before_action_feature))]),
                                              str(list_action_in_sec))
                                     )

            # _step
            self._inner_step(action_in_sec)

            # get reward
            reward = self.get_reward()

            for j in range(len(reward)):
                average_reward_action_list[j] = (average_reward_action_list[j]*i + reward[j])/(i+1)
            # log
            self.log(cur_time=instant_time, before_action_feature=before_action_feature, action=action_in_sec_display)

            next_state, done = self.get_state()

        return next_state, reward, done, average_reward_action_list

    def _inner_step(self, action):

        # copy current measurements to previous measurements
        for inter in self.list_intersection:
            inter.update_previous_measurements()

        # set signals
        for inter_ind, inter in enumerate(self.list_intersection):
            inter.set_signal(
                action=action[inter_ind],
                action_pattern=self.dic_traffic_env_conf["ACTION_PATTERN"],
                yellow_time=self.dic_traffic_env_conf["YELLOW_TIME"],
                all_red_time=self.dic_traffic_env_conf["ALL_RED_TIME"]
            )

        # run one step

        for i in range(int(1/self.dic_traffic_env_conf["INTERVAL"])):
            traci.simulationStep()

        # get new measurements
        for inter in self.list_intersection:
            inter.update_current_measurements()

        #self.log_lane_vehicle_position()
        # self.log_first_vehicle()
        #self.log_phase()

    def _check_episode_done(self, list_state):

        # ======== to implement ========

        return False

    def log_lane_vehicle_position(self):
        def list_to_str(alist):
            new_str = ""
            for s in alist:
                new_str = new_str + str(s) + " "
            return new_str

        dic_lane_map = {
            "edge1-0_0": "w",
            "edge2-0_0": "e",
            "edge3-0_0": "s",
            "edge4-0_0": "n"
        }
        for inter in self.list_intersection:
            for lane in inter.list_entering_lanes:
                print(str(self.get_current_time()) + ", " + lane + ", " + list_to_str(inter._get_lane_vehicle_position([lane])[0]),
                      file=open(os.path.join(self.path_to_log, "lane_vehicle_position_%s.txt" % dic_lane_map[lane]),
                                "a"))

    def log_first_vehicle(self):
        _veh_id = "1."
        _veh_id_2 = "3."
        _veh_id_3 = "4."
        _veh_id_4 = "6."
        for inter in self.list_intersection:
            for i in range(100):
                veh_id = _veh_id + str(i)
                veh_id_2 = _veh_id_2 + str(i)
                pos, speed = inter._get_vehicle_info(veh_id)
                pos_2, speed_2 = inter._get_vehicle_info(veh_id_2)
                #print(i, veh_id, pos, veh_id_2, speed, pos_2, speed_2)

                if not os.path.exists(os.path.join(self.path_to_log, "first_vehicle_info_a")):
                    os.makedirs(os.path.join(self.path_to_log, "first_vehicle_info_a"))

                if not os.path.exists(os.path.join(self.path_to_log, "first_vehicle_info_b")):
                    os.makedirs(os.path.join(self.path_to_log, "first_vehicle_info_b"))

                if pos and speed:
                    print("%f, %f, %f" % (self.get_current_time(), pos, speed),
                          file=open(os.path.join(self.path_to_log, "first_vehicle_info_a", "first_vehicle_info_a_%d.txt"%i), "a"))
                if pos_2 and speed_2:
                    print("%f, %f, %f" % (self.get_current_time(), pos_2, speed_2),
                          file=open(os.path.join(self.path_to_log, "first_vehicle_info_b", "first_vehicle_info_b_%d.txt"%i), "a"))

                veh_id_3 = _veh_id_3 + str(i)
                veh_id_4 = _veh_id_4 + str(i)
                pos_3, speed_3 = inter._get_vehicle_info(veh_id_3)
                pos_4, speed_4 = inter._get_vehicle_info(veh_id_4)
                # print(i, veh_id, pos, veh_id_2, speed, pos_2, speed_2)
                if not os.path.exists(os.path.join(self.path_to_log, "first_vehicle_info_c")):
                    os.makedirs(os.path.join(self.path_to_log, "first_vehicle_info_c"))

                if not os.path.exists(os.path.join(self.path_to_log, "first_vehicle_info_d")):
                    os.makedirs(os.path.join(self.path_to_log, "first_vehicle_info_d"))

                if pos_3 and speed_3:
                    print("%f, %f, %f" % (self.get_current_time(), pos_3, speed_3),
                          file=open(
                              os.path.join(self.path_to_log, "first_vehicle_info_c", "first_vehicle_info_a_%d.txt" % i),
                              "a"))
                if pos_4 and speed_4:
                    print("%f, %f, %f" % (self.get_current_time(), pos_4, speed_4),
                          file=open(
                              os.path.join(self.path_to_log, "first_vehicle_info_d", "first_vehicle_info_b_%d.txt" % i),
                              "a"))

    def log_phase(self):
        for inter in self.list_intersection:
            print("%f, %f" % (self.get_current_time(), inter.current_phase_index),
                  file=open(os.path.join(self.path_to_log, "log_phase.txt"), "a"))


if __name__ == '__main__':
    pass
    file = "/Users/Wingslet/PycharmProjects/MI_trafficLightRL-develope/data/2_test/DeeplightDeeplightDeeplight_['3_intersections_500_0.3_uni.xml']_['3_intersections_500_0.3_uni.xml']_11_27_15_24_09_IRL/cross.sumocfg"

    traffic_light_node_dict = SumoEnv._infastructure_extraction(file)
    print(traffic_light_node_dict)