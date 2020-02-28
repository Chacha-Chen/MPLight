import sys

sys.path.append("..")
from agent import Agent
import random
import numpy as np
import os


class FixedtimeOffsetAgent(Agent):

    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, cnt_round, intersection_id):

        super(FixedtimeOffsetAgent, self).__init__(dic_agent_conf, dic_traffic_env_conf, dic_path, intersection_id)

        self.current_phase_time = 0
        self.IF_MULTI = False
        ### Arterial
        ## TODO grid offset
        if dic_traffic_env_conf["NUM_ROW"]>1:
            self.offset = (int(self.intersection_id) * 28)
        else:
            self.offset = (int(self.intersection_id) * 28)
        self.phase_length = len(self.dic_traffic_env_conf["PHASE"][self.dic_traffic_env_conf["SIMULATOR_TYPE"]])

        traffic_demand = self.dic_traffic_env_conf['TRAFFIC_FILE'].split('_')[3]
        if self.dic_traffic_env_conf['SIMULATOR_TYPE'] == 'sumo':
            ratio = self.dic_traffic_env_conf['TRAFFIC_FILE'].split('_')[4].split('.xml')[0]
        else:
            ratio = self.dic_traffic_env_conf['TRAFFIC_FILE'].split('_')[4].split('.json')[0]



        ###TODO real world pattern no phase split needed

        if len(self.dic_agent_conf["FIXED_TIME"]) == 4:
            self.IF_MULTI = True
            self.dic_agent_conf["FIXED_TIME"] = [15, 15, 15, 15]

        else:
            traffic_demand = int(traffic_demand)

            ratio = float(ratio)

        # if ratio%1 != 0:
            self.dic_agent_conf["FIXED_TIME"] = self.get_phase_split(traffic_demand, ratio)
        # else:
        #     self.dic_agent_conf["FIXED_TIME"] = [15,15,15,15]
        #


        cycle = np.sum(self.dic_agent_conf["FIXED_TIME"])+ 5 *len(self.dic_agent_conf["FIXED_TIME"])

        self.offset = self.offset % cycle

        file = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"],'GW_SETTING_{0}.txt'.format(intersection_id))
        with open(file,'w') as f:
        # f.write("123")
            f.write("phase_split_{0}_{1}_offset{2}_{3}".format(self.dic_agent_conf["FIXED_TIME"][0],\
                                                           self.dic_agent_conf["FIXED_TIME"][1],self.offset,cycle))

            # f.write("Fail to read csv of inter {0} in early stopping of round {1}\n".format(inter_id, cnt_round))

        if self.dic_traffic_env_conf["SIMULATOR_TYPE"] == "anon":
            self.DIC_PHASE_MAP = {
                1: 0,
                2: 1,
                3: 2,
                4: 3,
                0: 0
            }
        else:
            self.DIC_PHASE_MAP = {
                0: 0,
                1: 1,
                2: 2,
                3: 3,
                -1: -1
            }

    def choose_action(self, count, state):
        ''' choose the best action for current state '''

        if state["cur_phase"][0] == -1:
            return self.action
        cur_phase = self.DIC_PHASE_MAP[state["cur_phase"][0]]

        if self.dic_traffic_env_conf["ACTION_PATTERN"] == "set":
            if count < self.offset + self.dic_agent_conf["FIXED_TIME"][cur_phase]:
                self.action = cur_phase
                self.current_phase_time += 1
                return cur_phase

            elif state["time_this_phase"][0] <= self.dic_agent_conf["FIXED_TIME"][cur_phase]:
                self.action = cur_phase
                self.current_phase_time += 1
                return cur_phase

            else:
                self.current_phase_time = 0
                self.action = (cur_phase + 1) % self.phase_length
                return (cur_phase + 1) % self.phase_length

        else:
            if state["time_this_phase"][0] >= self.dic_agent_conf["FIXED_TIME"][cur_phase] and cur_phase != -1:
                self.current_phase_time = 0
                self.action = 1
                return 1
            else:
                self.current_phase_time += 1
                self.action = 0
                return 0

    def round_up(self, x, b=5):
        round_x = (b * np.ceil(x.astype(float) / b)).astype(int)
        round_x[np.where(round_x < self.dic_agent_conf["MIN_PHASE_TIME"])] = self.dic_agent_conf["MIN_PHASE_TIME"]
        return round_x

    def get_phase_split(self, traffic_demand, ratio):

        h = 2.45
        tL_set = 5
        tL = 7
        PHF = 1
        vc = 1
        N = 2
        vehicles_count_for_critical_lane_phase = traffic_demand * (1 + ratio)
        max_allowed_vol = 3600 / h * PHF * vc
        total_vol = np.sum(vehicles_count_for_critical_lane_phase)
        if total_vol / max_allowed_vol > 0.95:
            cycle_length = N * tL / (1 - 0.95)
        else:
            cycle_length = N * tL / (1 - total_vol / max_allowed_vol)

        if cycle_length < 0:
            sys.exit("cycle length calculation error")

        effect_cycle_length = cycle_length - tL_set * N
        if np.sum(vehicles_count_for_critical_lane_phase) != 0:
            phase_split = np.copy(vehicles_count_for_critical_lane_phase) / np.sum(
                vehicles_count_for_critical_lane_phase) * effect_cycle_length
        else:
            phase_split = np.full(shape=(len(vehicles_count_for_critical_lane_phase),),
                                  fill_value=1 / len(vehicles_count_for_critical_lane_phase)) * effect_cycle_length

        phase_split = int(phase_split) + 1
        green = int(phase_split / (1 + ratio)) + 1
        red = int(phase_split / (1 + ratio) * ratio) + 1

        phase_split = np.array([green, red])

        # b = self.dic_agent_conf["MIN_PHASE_TIME"]

        phase_split = self.round_up(phase_split, b=self.dic_agent_conf["MIN_PHASE_TIME"])

        # while green % b != 0:
        #     green += 1
        #
        # while red % b != 0:
        #     red += 1


        if self.IF_MULTI:
            green1 = green / 7
            green2 = green / 7 * 6
            red1 = red / 7
            red2 = red / 7 * 6
            # while green1 % b != 0:
            #     green1 += 1
            # while green2 % b != 0:
            #     green2 += 1
            # while red1 % b != 0:
            #     red1 += 1
            # while red2 % b != 0:
            #     red2 += 1
            phase_split = np.array([green2,green1,red2,red1])

            phase_split = self.round_up(phase_split, b=self.dic_agent_conf["MIN_PHASE_TIME"])


            return phase_split

            ## split red green

        # while green % b != 0:
        #     green += 1
        #
        # while red % b != 0:
        #     red += 1

        return phase_split
