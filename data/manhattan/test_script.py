import cityflow
import json

config = {
    "interval": 1.0,
    "seed": 0,
    "dir": "./",
    "roadnetFile": "manhattan.json",
    "flowFile": "manhattan_7846.json",
    "rlTrafficLight": False,
    "saveReplay": True,
    "roadnetLogFile": "roadnet_manhattan.json",
    "replayLogFile": "log.txt"
}

with open('config.json', 'w') as fp:
    json.dump(config, fp)

config_path = 'config.json'
eng = cityflow.Engine(config_path, thread_num=1)


for i in range(100):
    # print(i)
    print("Vehicle count: ",eng.get_vehicle_count())
    eng.next_step()
    print("Average travel time:",eng.get_average_travel_time())

print(eng.get_average_travel_time())
