import config
import copy
from baseline.oneline import OneLine
import os
import time
from multiprocessing import Process
import sys
from script import get_traffic_volume
import argparse



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memo", type=str, default='FT_benchmark_0418')
    parser.add_argument("-all", action="store_true", default=False)
    parser.add_argument("--env", type=int, default=0)
    parser.add_argument("-g", action="store_true", default=True)
    parser.add_argument("--road_net", type=str, default='1_3')
    parser.add_argument("--volume", type=int, default=700)
    parser.add_argument("--ratio", type=float, default=0.3)
    parser.add_argument("--model", type=str, default="Fixedtime")
    parser.add_argument("--count",type=int, default=3600)
    parser.add_argument("--lane", type=int, default=1)
    # parser.add_argument("-uniform", action="store_true",default=False)
    # parser.add_argument("-s", action="store_true",default=False)
    parser.add_argument("-syn", action="store_true",default=True)
    parser.add_argument("-uniform", action="store_true",default=False)
    parser.add_argument("-real", action="store_true",default=False)
    parser.add_argument("-beaver", action="store_true",default=False)
    parser.add_argument("-hangzhou", action="store_true",default=False)

    return parser.parse_args()

def merge(dic_tmp, dic_to_change):
    dic_result = copy.deepcopy(dic_tmp)
    dic_result.update(dic_to_change)

    return dic_result

def check_all_workers_working(list_cur_p):
    for i in range(len(list_cur_p)):
        if not list_cur_p[i].is_alive():
            return i
    return -1

def oneline_wrapper(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path):
    oneline = OneLine(dic_exp_conf=merge(config.DIC_EXP_CONF, dic_exp_conf),
                      dic_agent_conf=merge(getattr(config, "DIC_{0}_AGENT_CONF".format(dic_exp_conf["MODEL_NAME"].upper())),
                                           dic_agent_conf),
                      dic_traffic_env_conf=merge(config.dic_traffic_env_conf, dic_traffic_env_conf),
                      dic_path=merge(config.DIC_PATH, dic_path)
                      )
    oneline.train()
    return

def main(memo, rall, road_net, env, gui, volume, ratio, model,count,lane,synthetic,uniform,jinan_real,beaver,hangzhou):
    ENVIRONMENT = ["sumo", "anon"][env]

    separate_test = False
    sparse_test = False
    if uniform:
        traffic_file_list = []
        lane = 1
        cnt = 3600
        road_net = "1_6"
        data_path = os.path.join("data", "template_s", "1_6")
        traffic_file_list = [i for i in os.listdir(os.path.join(os.getcwd(),data_path)) if 'roadnet' not in i]
    elif jinan_real:
        lane = 3
        cnt = 3600
        road_net = "1_3"
        traffic_file_list = ["anon_1_3_700_0.6_real.json"]
    elif beaver:
        lane = 3
        cnt = 3600
        road_net = "1_5"
        traffic_file_list = ["anon_1_5_700_0.6_beaver.json"]
    elif hangzhou:
        lane = 3
        cnt = 3600
        road_net = "4_4"
        traffic_file_list = ["anon_4_4_hangzhou_real.json"]
        # num_rounds = 200
    # elif r_all:
    #     traffic_file_list = [ENVIRONMENT+"_"+road_net+"_%d_%s.json" %(v,r) for v in [300,700] for r in [0.3,0.6]]
    elif synthetic:
        # separate_test = True
        # sparse_test = True
        # traffic_file_list = [ENVIRONMENT + "_" + road_net + "_%d_%s_%d" % (v, r, j) for v in [500, 700] for r in
        #                      [0.3, 0.6] for j in [1, 3]]
        cnt = 3600
        # road_net = "4_4"
        lane = 3
        traffic_file_list = [ENVIRONMENT + "_" + road_net + "_%d_%s_synthetic.json" % (v, r) for v in [500] for r in
                             [0.5]]
    elif road_net == '2_2':
        traffic_file_list = [ENVIRONMENT + "_" + road_net + "_%d_%s_synthetic.json" % (v, r) for v in [300,500,700] for r in
                             [1.0]]
        lane = 1
        num_rounds = 600
    else:
        traffic_file_list = ["{0}_{1}_{2}_{3}.xml".format(ENVIRONMENT, road_net, volume, ratio)]

    NUM_COL = int(road_net.split('_')[0])
    NUM_ROW = int(road_net.split('_')[1])
    num_intersections = NUM_ROW * NUM_COL
    print('num_intersections:', num_intersections)

    # NUM_COL = int(road_net.split('_')[0])
    # NUM_ROW = int(road_net.split('_')[1])
    # num_intersections = NUM_ROW * NUM_COL
    # print('num_intersections:', num_intersections)
    #
    # NUM_COL = int(road_net.split('_')[0])
    # NUM_ROW = int(road_net.split('_')[1])
    # num_intersections = NUM_ROW * NUM_COL
    #
    ENVIRONMENT = ["sumo", "anon"][env]
    #
    #
    # if rall:
    #     traffic_file_list = [ENVIRONMENT+"_"+road_net+"_%d_%s" %(v,r) for v in [300,500,700] for r in [0.3,0.6]]
    # else:
    #     traffic_file_list=["{0}_{1}_{2}_{3}".format(ENVIRONMENT, road_net, volume, ratio)]
    # separate_test = False
    #
    # if synthetic:
    #     separate_test = True
    #     sparse_test = True
    #     traffic_file_list = [ENVIRONMENT + "_" + road_net + "_%d_%s_%d" % (v, r, j) for v in [500, 700] for r in
    #                          [0.3, 0.6] for j in [1, 3]]
    #     cnt = 10800
    #
    #
    # if env:
    #     traffic_file_list = [i+ ".json" for i in traffic_file_list ]
    # else:
    #     traffic_file_list = [i+ ".xml" for i in traffic_file_list ]

    #
    # if uniform:
    #     traffic_file_list = []
    #     lane = 1
    #     count = 3600
    #     road_net = "1_6"
    #     data_path = os.path.join("data", "template_s", "1_6")
    #     traffic_file_list = [i for i in os.listdir(os.path.join(os.getcwd(),data_path)) if 'roadnet' not in i]
    #


    process_list = []
    multi_process = True
    n_workers = 60

    # !! important para !!
    # best cycle length sets the min_action_time 1
    # normal formula 1
    min_action_time = 1

    for traffic_file in traffic_file_list:
      #for cl in range(1, 60):
        cl = 30
        model_name = model

        # automatically choose the n_segments
        if "cross" in traffic_file:
            n_segments = 1
        else:
            n_segments = 12
        dic_exp_conf_extra = {
            "RUN_COUNTS": count,
            "MODEL_NAME": model_name,
            "TRAFFIC_FILE": [traffic_file],
            "ROADNET_FILE": "roadnet_{0}.json".format(road_net)
        }
        #
        # if 'anon' not in traffic_file:
        #     volume = int(traffic_file.split('_')[3])
        # else:
        #     volume = get_traffic_volume(traffic_file)

        '''
        ## deeplight agent conf
        dic_agent_conf_extra = {
            "EPOCHS": 100,
            "SAMPLE_SIZE": 300,
            "MAX_MEMORY_LEN": 1000,
            "PHASE_SELECTOR": False,
            "SEPARATE_MEMORY": False,
            "EPSILON": 0

        }
        '''


        dic_traffic_env_conf_extra = {

            "NUM_AGENTS": num_intersections,
            "NUM_INTERSECTIONS": num_intersections,
            "ACTION_PATTERN": "set",
            "MIN_ACTION_TIME": min_action_time,

            "MEASURE_TIME": 10,
            "IF_GUI": gui,
            "DEBUG": False,
            "TOP_K_ADJACENCY": 4,
            "SIMULATOR_TYPE": ENVIRONMENT,
            "BINARY_PHASE_EXPANSION": True,
            "FAST_COMPUTE": False,
            "SEPARATE_TEST": separate_test,
            "NEIGHBOR": False,
            "MODEL_NAME": model,
            "RUN_COUNTS": count,


            "SAVEREPLAY": True,
            "NUM_ROW": NUM_ROW,
            "NUM_COL": NUM_COL,

            "TRAFFIC_FILE": traffic_file,
            "ROADNET_FILE": "roadnet_{0}.json".format(road_net),

            "LIST_STATE_FEATURE": [
                "cur_phase",
                "time_this_phase",
                # "vehicle_position_img",
                # "vehicle_speed_img",
                # "vehicle_acceleration_img",
                # "vehicle_waiting_time_img",
                # "lane_num_vehicle",
                # "lane_num_vehicle_been_stopped_thres01",
                # "lane_num_vehicle_been_stopped_thres1",
                # "lane_queue_length",
                # "lane_num_vehicle_left",
                # "lane_sum_duration_vehicle_left",
                # "lane_sum_waiting_time",
                # "terminal",
                "coming_vehicle",
                "leaving_vehicle",
                "pressure"

                # "adjacency_matrix",
                # "lane_queue_length"
            ],

                "DIC_FEATURE_DIM": dict(
                    D_LANE_QUEUE_LENGTH=(4,),
                    D_LANE_NUM_VEHICLE=(4,),

                    D_COMING_VEHICLE = (4,),
                    D_LEAVING_VEHICLE = (4,),

                    D_LANE_NUM_VEHICLE_BEEN_STOPPED_THRES1=(4,),
                    D_CUR_PHASE=(1,),
                    D_NEXT_PHASE=(1,),
                    D_TIME_THIS_PHASE=(1,),
                    D_TERMINAL=(1,),
                    D_LANE_SUM_WAITING_TIME=(4,),
                    D_VEHICLE_POSITION_IMG=(4, 60,),
                    D_VEHICLE_SPEED_IMG=(4, 60,),
                    D_VEHICLE_WAITING_TIME_IMG=(4, 60,),

                    D_PRESSURE=(1,),

                    D_ADJACENCY_MATRIX=(2,)
                ),

            "DIC_REWARD_INFO": {
                "flickering": 0,
                "sum_lane_queue_length": 0,
                "sum_lane_wait_time": 0,
                "sum_lane_num_vehicle_left": 0,
                "sum_duration_vehicle_left": 0,
                "sum_num_vehicle_been_stopped_thres01": 0,
                "sum_num_vehicle_been_stopped_thres1": -0.25,
                "pressure": 0 # -0.25
            },

            "LANE_NUM": {
                "LEFT": 1,
                "RIGHT": 1,
                "STRAIGHT": 1
            },

            "PHASE": {
                "sumo": {
                    0: [0, 1, 0, 1, 0, 0, 0, 0],# 'WSES',
                    1: [0, 0, 0, 0, 0, 1, 0, 1],# 'NSSS',
                    2: [1, 0, 1, 0, 0, 0, 0, 0],# 'WLEL',
                    3: [0, 0, 0, 0, 1, 0, 1, 0]# 'NLSL',
                },
                "anon": {
                    # 0: [0, 0, 0, 0, 0, 0, 0, 0],
                    1: [0, 1, 0, 1, 0, 0, 0, 0],# 'WSES',
                    3: [1, 0, 1, 0, 0, 0, 0, 0], # 'WLEL',
                    2: [0, 0, 0, 0, 0, 1, 0, 1],# 'NSSS',
                    4: [0, 0, 0, 0, 1, 0, 1, 0]# 'NLSL',
                    # 'WSWL',
                    # 'ESEL',
                    # 'WSES',
                    # 'NSSS',
                    # 'NSNL',
                    # 'SSSL',
                },
            }
        }



        ## ==================== multi_phase ====================

        # if dic_traffic_env_conf_extra["LANE_NUM"] == config._LS:
        if lane == 2:
            template = "template_ls"
            fixed_time = [cl, cl, cl, cl]
            fixed_time_dic = config.DIC_FIXEDTIME_MULTI_PHASE
        elif lane == 1:
            template = "template_s"
            fixed_time_dic = [15,15]
            dic_traffic_env_conf_extra["PHASE"] = {
                "sumo": {
                    0: [0, 1, 0, 1, 0, 0, 0, 0], # 'WSES',
                    1: [0, 0, 0, 0, 0, 1, 0, 1], # 'NSSS',
                },
                "anon": {
                    1: [0, 1, 0, 1, 0, 0, 0, 0], # 'WSES',
                    2: [0, 0, 0, 0, 0, 1, 0, 1], # 'NSSS',
                },
            }
            dic_traffic_env_conf_extra["LANE_NUM"] = {
                "LEFT": 0,
                "RIGHT": 0,
                "STRAIGHT": 1
            }
        elif lane == 3:
            if jinan_real:
                template = "template_sss"
                fixed_time_dic = [15, 15, 15, 15]
                dic_traffic_env_conf_extra["LANE_NUM"] = {
                    "LEFT": 0,
                    "RIGHT": 0,
                    "STRAIGHT": 3
                }
            else:
                template = "template_lsr"
                fixed_time_dic = [15,15,15,15]

        else:
            raise ValueError
        # lit_template = "lit_" + template

        ## fixedtime agent conf

        dic_agent_conf_extra = {
            "DAY_TIME": 3600,
            "UPDATE_PERIOD": 3600 / n_segments,  # if synthetic, update_period: 3600/12
            "FIXED_TIME": fixed_time_dic,
            "ROUND_UP": 1,
            "PHASE_TO_LANE": [[0, 1], [2, 3]],
            "MIN_PHASE_TIME": 5,
            "TRAFFIC_FILE": [traffic_file],
        }
        dic_traffic_env_conf_extra["NUM_AGENTS"] = dic_traffic_env_conf_extra["NUM_INTERSECTIONS"]

        prefix_intersections = str(road_net)
        dic_path_extra = {
            "PATH_TO_MODEL": os.path.join("model", memo, traffic_file+"_"+time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))+"_%d"%cl),
            "PATH_TO_WORK_DIRECTORY": os.path.join("records", memo, traffic_file+"_"+time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))+"_%d"%cl),
            "PATH_TO_DATA": os.path.join("data", template, prefix_intersections)
        }

        if multi_process:
            process_list.append(Process(target=oneline_wrapper,
                                        args=(dic_exp_conf_extra, dic_agent_conf_extra,
                                              dic_traffic_env_conf_extra, dic_path_extra))
                                )
        else:
            oneline_wrapper(dic_exp_conf_extra, dic_agent_conf_extra, dic_traffic_env_conf_extra, dic_path_extra)

    if multi_process:
        i = 0
        list_cur_p = []
        for p in process_list:
            if len(list_cur_p) < n_workers:
                print(i)
                p.start()
                list_cur_p.append(p)
                i += 1
            if len(list_cur_p) < n_workers:
                continue

            idle = check_all_workers_working(list_cur_p)

            while idle == -1:
                time.sleep(1)
                idle = check_all_workers_working(
                    list_cur_p)
            del list_cur_p[idle]

        for p in list_cur_p:
            p.join()

if __name__ == "__main__":
    args = parse_args()

    # memo = "test"
    # args.g = True
# <<<<<<< HEAD
    main(args.memo, args.all, args.road_net, args.env, args.g, args.volume, args.ratio,args.model,args.count,args.lane,args.syn,args.uniform,args.real,args.beaver,args.hangzhou)
# =======
#     main(args.memo, args.all, args.road_net, args.env, args.g, args.volume, args.ratio,args.model,args.count,args.lane)
# >>>>>>> ana_simulator


    # def main(memo=None, baseline, road_net, env, gui, volume, ratio, model):
