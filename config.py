# parameters and paths
from baseline.deeplight_agent import DeeplightAgent
from baseline.fixedtime_agent import FixedtimeAgent
from baseline.fixedtimeoffset_agent import FixedtimeOffsetAgent
from baseline.maxpressure_agent import MaxPressureAgent
from baseline.formula_agent import FormulaAgent
from simple_dqn_agent import SimpleDQNAgent
from transfer_dqn_agent import TransferDQNAgent
from transfer_dqn_agent_pressure import TransferDQNPressAgent
from transfer_dqn_agent_pressure_one import TransferDQNPressOneAgent
from baseline.random_agent import RandomAgent
from dgn_agent import DGNAgent
from stgat_agent import STGATAgent
from gcn_agent import GCNAgent
from simple_dqn_one_agent import SimpleDQNOneAgent
from transfer_dqn_one_agent import TransferDQNOneAgent
from sumo_env import SumoEnv
from anon_env import AnonEnv

DIC_EXP_CONF = {
    "RUN_COUNTS": 3600,
    "TRAFFIC_FILE": [
        "cross.2phases_rou01_equal_450.xml"
    ],
    "MODEL_NAME": "SimpleDQN",
    "NUM_ROUNDS": 200,
    "NUM_GENERATORS": 3,
    "LIST_MODEL":
        ["Fixedtime", "Deeplight", "SimpleDQN", "TransferDQN","TransferDQNPress","TransferDQNPressOne","TransferDQNOne"],
    "LIST_MODEL_NEED_TO_UPDATE":
        ["Deeplight", "SimpleDQN", "DGN","GCN", "SimpleDQNOne","STGAT", "TransferDQN","TransferDQNPress","TransferDQNOne","TransferDQNPressOne"],
    "MODEL_POOL": False,
    "NUM_BEST_MODEL": 3,
    "PRETRAIN": True,
    "PRETRAIN_MODEL_NAME": "Random",
    "PRETRAIN_NUM_ROUNDS": 10,
    "PRETRAIN_NUM_GENERATORS": 10,
    "AGGREGATE": False,
    "DEBUG": False,
    "EARLY_STOP": False,

    "MULTI_TRAFFIC": False,
    "MULTI_RANDOM": False,
}


DIC_FIXEDTIME_AGENT_CONF = {
    "FIXED_TIME": [
        30,
        30
    ],
}

DIC_RANDOM_AGENT_CONF = {

}

DIC_DGN_AGENT_CONF = {

}


DIC_FORMULA_AGENT_CONF = {
    "DAY_TIME": 3600,
    "UPDATE_PERIOD": 3600,
    "FIXED_TIME": [30, 30],
    "ROUND_UP": 5,
    "PHASE_TO_LANE": [[0, 1], [2, 3]],
    "MIN_PHASE_TIME": 5,
    "TRAFFIC_FILE": [
        "cross.2phases_rou01_equal_450.xml"
    ],
}

dic_traffic_env_conf = {
    "ACTION_PATTERN": "set",
    "NUM_INTERSECTIONS": 1,
    "MIN_ACTION_TIME": 10,
    "YELLOW_TIME": 5,
    "ALL_RED_TIME": 0,
    "NUM_PHASES": 2,
    "NUM_LANES": 1,
    "ACTION_DIM": 2,
    "MEASURE_TIME": 10,
    "IF_GUI": True,
    "DEBUG": False,

    "INTERVAL": 1,
    "THREADNUM": 8,
    "SAVEREPLAY": True,
    "RLTRAFFICLIGHT": True,

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

    "LIST_STATE_FEATURE": [
        "cur_phase",
        "time_this_phase",
        "vehicle_position_img",
        "vehicle_speed_img",
        "vehicle_acceleration_img",
        "vehicle_waiting_time_img",
        "lane_num_vehicle",
        "lane_num_vehicle_been_stopped_thres01",
        "lane_num_vehicle_been_stopped_thres1",
        "lane_queue_length",
        "lane_num_vehicle_left",
        "lane_sum_duration_vehicle_left",
        "lane_sum_waiting_time",
        "terminal",

        "coming_vehicle",
        "leaving_vehicle",
        "pressure",

        "adjacency_matrix"

    ],

    "DIC_REWARD_INFO": {
        "flickering": 0,
        "sum_lane_queue_length": 0,
        "sum_lane_wait_time": 0,
        "sum_lane_num_vehicle_left": 0,
        "sum_lane_num_vehicle":0,
        "sum_delta_lane_num_vehicle":0,
        "sum_duration_vehicle_left": 0,
        "sum_num_vehicle_been_stopped_thres01": 0,
        "sum_num_vehicle_been_stopped_thres1": -0.25,
        "pressure": 0,
    },

    "LANE_NUM": {
        "LEFT": 1,
        "RIGHT": 1,
        "STRAIGHT": 1
    },

    "PHASE": [
        'WSES',
        'NSSS',
        'WLEL',
        'NLSL',
        # 'WSWL',
        # 'ESEL',
        # 'NSNL',
        # 'SSSL',
    ],

}

_LS = {"LEFT": 1,
       "RIGHT": 0,
       "STRAIGHT": 1
       }
_S = {
    "LEFT": 0,
    "RIGHT": 0,
    "STRAIGHT": 1
}
_LSR = {
    "LEFT": 1,
    "RIGHT": 1,
    "STRAIGHT": 1
}

DIC_DEEPLIGHT_AGENT_CONF = {
    "LEARNING_RATE": 0.001,
    "UPDATE_PERIOD": 300,
    "SAMPLE_SIZE": 300,
    "SAMPLE_SIZE_PRETRAIN": 3000,
    "BATCH_SIZE": 20,
    "EPOCHS": 100,
    "EPOCHS_PRETRAIN": 500,
    "SEPARATE_MEMORY": True,
    "PRIORITY_SAMPLING": False,
    "UPDATE_Q_BAR_FREQ": 5,
    "GAMMA": 0.8,
    "GAMMA_PRETRAIN": 0,
    "MAX_MEMORY_LEN": 1000,
    "PATIENCE": 10,
    "PHASE_SELECTOR": True,
    "KEEP_OLD_MEMORY": 0,
    "DDQN": False,
    "D_DENSE": 20,
    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.95,
    "MIN_EPSILON": 0.2,
    "LOSS_FUNCTION": "mean_squared_error",
    "NORMAL_FACTOR": 20,

}

DIC_SIMPLEDQN_AGENT_CONF = {
    "LEARNING_RATE": 0.001,
    "SAMPLE_SIZE": 1000,
    "BATCH_SIZE": 20,
    "EPOCHS": 100,
    "UPDATE_Q_BAR_FREQ": 5,
    "UPDATE_Q_BAR_EVERY_C_ROUND": False,
    "GAMMA": 0.8,
    "MAX_MEMORY_LEN": 10000,
    "PATIENCE": 10,
    "D_DENSE": 20,
    "N_LAYER": 2,
    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.95,
    "MIN_EPSILON": 0.2,
    "LOSS_FUNCTION": "mean_squared_error",
    "SEPARATE_MEMORY": False,
    "NORMAL_FACTOR": 20,
    "TRAFFIC_FILE": "cross.2phases_rou01_equal_450.xml",
}

DIC_TRANSFERDQN_AGENT_CONF = {
    "LEARNING_RATE": 0.001,
    "SAMPLE_SIZE": 1000,
    "BATCH_SIZE": 20,
    "EPOCHS": 100,
    "UPDATE_Q_BAR_FREQ": 5,
    "UPDATE_Q_BAR_EVERY_C_ROUND": False,
    "GAMMA": 0.8,
    "MAX_MEMORY_LEN": 10000,
    "PATIENCE": 10,
    "D_DENSE": 20,
    "N_LAYER": 2,
    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.95,
    "MIN_EPSILON": 0.2,
    "LOSS_FUNCTION": "mean_squared_error",
    "SEPARATE_MEMORY": False,
    "NORMAL_FACTOR": 20,
    "TRAFFIC_FILE": "cross.2phases_rou01_equal_450.xml",
}

DIC_TRANSFERDQNONE_AGENT_CONF = {
    "LEARNING_RATE": 0.001,
    "SAMPLE_SIZE": 1000,
    "BATCH_SIZE": 20,
    "EPOCHS": 100,
    "UPDATE_Q_BAR_FREQ": 5,
    "UPDATE_Q_BAR_EVERY_C_ROUND": False,
    "GAMMA": 0.8,
    "MAX_MEMORY_LEN": 10000,
    "PATIENCE": 10,
    "D_DENSE": 20,
    "N_LAYER": 2,
    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.95,
    "MIN_EPSILON": 0.2,
    "LOSS_FUNCTION": "mean_squared_error",
    "SEPARATE_MEMORY": False,
    "NORMAL_FACTOR": 20,
    "TRAFFIC_FILE": "cross.2phases_rou01_equal_450.xml",
}
DIC_TRANSFERDQNPRESS_AGENT_CONF = {
    "LEARNING_RATE": 0.001,
    "SAMPLE_SIZE": 1000,
    "BATCH_SIZE": 20,
    "EPOCHS": 100,
    "UPDATE_Q_BAR_FREQ": 5,
    "UPDATE_Q_BAR_EVERY_C_ROUND": False,
    "GAMMA": 0.8,
    "MAX_MEMORY_LEN": 10000,
    "PATIENCE": 10,
    "D_DENSE": 20,
    "N_LAYER": 2,
    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.95,
    "MIN_EPSILON": 0.2,
    "LOSS_FUNCTION": "mean_squared_error",
    "SEPARATE_MEMORY": False,
    "NORMAL_FACTOR": 20,
    "TRAFFIC_FILE": "cross.2phases_rou01_equal_450.xml",
}

DIC_TRANSFERDQNPRESSONE_AGENT_CONF = {
    "LEARNING_RATE": 0.001,
    "SAMPLE_SIZE": 1000,
    "BATCH_SIZE": 20,
    "EPOCHS": 100,
    "UPDATE_Q_BAR_FREQ": 5,
    "UPDATE_Q_BAR_EVERY_C_ROUND": False,
    "GAMMA": 0.8,
    "MAX_MEMORY_LEN": 10000,
    "PATIENCE": 10,
    "D_DENSE": 20,
    "N_LAYER": 2,
    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.95,
    "MIN_EPSILON": 0.2,
    "LOSS_FUNCTION": "mean_squared_error",
    "SEPARATE_MEMORY": False,
    "NORMAL_FACTOR": 20,
    "TRAFFIC_FILE": "cross.2phases_rou01_equal_450.xml",
}

DIC_SIMPLEDQNONE_AGENT_CONF = {
    "LEARNING_RATE": 0.001,
    "SAMPLE_SIZE": 1000,
    "BATCH_SIZE": 20,
    "EPOCHS": 100,
    "UPDATE_Q_BAR_FREQ": 5,
    "UPDATE_Q_BAR_EVERY_C_ROUND": False,
    "GAMMA": 0.8,
    "MAX_MEMORY_LEN": 10000,
    "PATIENCE": 10,
    "D_DENSE": 20,
    "N_LAYER": 2,
    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.95,
    "MIN_EPSILON": 0.2,
    "LOSS_FUNCTION": "mean_squared_error",
    "SEPARATE_MEMORY": False,
    "NORMAL_FACTOR": 20,
    "TRAFFIC_FILE": "cross.2phases_rou01_equal_450.xml",
}

DIC_DGN_AGENT_CONF = {
    "nan_code":True,
    "att_regularization":False,
    "rularization_rate":0.03,
    "LEARNING_RATE": 0.001,
    "SAMPLE_SIZE": 1000,
    "BATCH_SIZE": 20,
    "EPOCHS": 100,
    "UPDATE_Q_BAR_FREQ": 5,
    "UPDATE_Q_BAR_EVERY_C_ROUND": False,
    "GAMMA": 0.8,
    "MAX_MEMORY_LEN": 10000,
    "PATIENCE": 10,
    "D_DENSE": 20,
    "N_LAYER": 2,
    #special care for pretrain
    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.95,
    "MIN_EPSILON": 0.2,

    "LOSS_FUNCTION": "mean_squared_error",
    "SEPARATE_MEMORY": False,
    "NORMAL_FACTOR": 20,
    "TRAFFIC_FILE": "cross.2phases_rou01_equal_450.xml",
}

DIC_STGAT_AGENT_CONF = {
    "PRIORITY": False,
    "nan_code":True,
    "att_regularization":False,
    "rularization_rate":0.03,
    "LEARNING_RATE": 0.001,
    "SAMPLE_SIZE": 1000,
    "BATCH_SIZE": 20,
    "EPOCHS": 100,
    "UPDATE_Q_BAR_FREQ": 5,
    "UPDATE_Q_BAR_EVERY_C_ROUND": False,
    "GAMMA": 0.8,
    "MAX_MEMORY_LEN": 10000,
    "PATIENCE": 10,
    "D_DENSE": 20,
    "N_LAYER": 2,
    #special care for pretrain
    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.95,
    "MIN_EPSILON": 0.2,

    "LOSS_FUNCTION": "mean_squared_error",
    "SEPARATE_MEMORY": False,
    "NORMAL_FACTOR": 20,
    "TRAFFIC_FILE": "cross.2phases_rou01_equal_450.xml",
}

DIC_GCN_AGENT_CONF = {
    "LEARNING_RATE": 0.001,
    "SAMPLE_SIZE": 1000,
    "BATCH_SIZE": 20,
    "EPOCHS": 100,
    "UPDATE_Q_BAR_FREQ": 5,
    "UPDATE_Q_BAR_EVERY_C_ROUND": False,
    "GAMMA": 0.8,
    "MAX_MEMORY_LEN": 10000,
    "PATIENCE": 10,
    "D_DENSE": [20,20],
    "N_LAYER": 2,
    "DROPOUT":0,
    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.95,
    "MIN_EPSILON": 0.2,
    "LOSS_FUNCTION": "mean_squared_error",
    "SEPARATE_MEMORY": False,
    "NORMAL_FACTOR": 20,
    "TRAFFIC_FILE": "cross.2phases_rou01_equal_450.xml",
}

DIC_PATH = {
    "PATH_TO_MODEL": "model/default",
    "PATH_TO_WORK_DIRECTORY": "records/default",
    "PATH_TO_DATA": "data/template",
    "PATH_TO_PRETRAIN_MODEL": "model/default",
    "PATH_TO_PRETRAIN_WORK_DIRECTORY": "records/default",
    "PATH_TO_PRETRAIN_DATA": "data/template",
    "PATH_TO_AGGREGATE_SAMPLES": "records/initial",
    "PATH_TO_ERROR": "errors/default"
}

DIC_AGENTS = {
    "Deeplight": DeeplightAgent,
    "Fixedtime": FixedtimeAgent,
    "FixedtimeOffset": FixedtimeOffsetAgent,
    "MaxPressure": MaxPressureAgent,
    "SimpleDQN": SimpleDQNAgent,
    "Formula": FormulaAgent,
    "Random": RandomAgent,
    'DGN': DGNAgent,
    'GCN': GCNAgent,
    'SimpleDQNOne': SimpleDQNOneAgent,
    'STGAT':STGATAgent,
    "TransferDQN": TransferDQNAgent,
    "TransferDQNOne":TransferDQNOneAgent,
    "TransferDQNPress": TransferDQNPressAgent,
    "TransferDQNPressOne": TransferDQNPressOneAgent,
}

DIC_ENVS = {
    "sumo": SumoEnv,
    "anon": AnonEnv
}

DIC_FIXEDTIMEOFFSET_AGENT_CONF = {
    100: [3, 3],
    200: [3, 3],
    300: [7, 7],
    400: [12, 12],
    500: [15, 15],
    600: [28, 28],
    700: [53, 53]
}

DIC_FIXEDTIME = {
    100: [3, 3],
    200: [3, 3],
    300: [7, 7],
    400: [12, 12],
    500: [15, 15],
    600: [28, 28],
    700: [53, 53]
    }

# min_action_time: 1
DIC_FIXEDTIME_NEW_SUMO = {
        100: [4, 4],
        200: [4, 4],
        300: [5, 5],
        400: [13, 13],
        500: [9, 9],
        600: [15, 15],
        700: [22, 22]
}

DIC_FIXEDTIME_MULTI_PHASE = {
    100: [4 for _ in range(4)],
    200: [4 for _ in range(4)],
    300: [5 for _ in range(4)],
    400: [8 for _ in range(4)],
    500: [9 for _ in range(4)],
    600: [13 for _ in range(4)],
    700: [23 for _ in range(4)]
}

DIC_MAXPRESSURE_AGENT_CONF = {
    100: [3, 3],
    200: [3, 3],
    300: [7, 7],
    400: [12, 12],
    500: [15, 15],
    600: [28, 28],
    700: [53, 53]
}
#DIC_FIXEDTIME_MULTI_PHASE = {
#    100: [2 for _ in range(4)],
#    200: [2 for _ in range(4)],
#    300: [1 for _ in range(4)],
#    400: [8 for _ in range(4)],
#    500: [16 for _ in range(4)],
#    600: [28 for _ in range(4)],
#    700: [55 for _ in range(4)]
#}

#DIC_FIXEDTIME_MULTI_PHASE = {
#    100: [5, 5, 5, 5],
#    200: [5, 5, 5, 5],
#    300: [5, 5, 5, 5],
#    400: [15, 15, 15, 15],
#    500: [20, 20, 20, 20],
#    600: [35, 35, 35, 35],
#    700: [50, 50, 50, 50]
#}