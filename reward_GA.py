import numpy as np
import copy

def getReward(x, ncar, data, maxspeed, capacity, policy, flag):
    s_points, s_window, s_need, s_service = data['s_points'], data['s_window'], data['s_need'], data['s_service']
    maxspeed = maxspeed # locker speed (km/min)
    capacity = capacity #locker capacity

    x1 = np.array(x[0], copy=True) #task allocation
    x2 = np.array(x[1], copy=True) #task sorting

    total_dispatch = 0 #number of locker deployed
    total_distance = 0 #travel distance
    total_late = 0 #total delay
    task = {} #task list of each locker
    for i in range(ncar):
        task[i] = []
        tk = np.argwhere(x1 == i).reshape(-1)
        if len(tk) != 0:#not null
            index = np.array([np.argwhere(x2 == tx).reshape(-1) for tx in tk]).reshape(-1)
            sorted_index = np.argsort(index)
            tk = tk[sorted_index] #task list after sorting 
            task[i] = tk
            total_dispatch += 1

    path = {} #Route（ID）
    route = {} #Route（Location）
    distance = {} #distance
    load = {} #loading when a locker conducts a task
    arrive = {} #arriving time
    leave = {} #leaving time
    late = {} #delay
    stay = {} #duration

    startpoint = np.array([0, 0]) #location of depot
    startid = len(s_points) #ID of depot
    for i in task:
        tk = task[i] #task list of current locker
        path[i] = []
        route[i] = []
        distance[i] = []
        load[i] = []
        arrive[i] = []
        leave[i] = []
        late[i] = []
        stay[i] = []
        if len(tk) != 0: #not null
            traveltime = np.linalg.norm(np.array(s_points[tk[0]]) - startpoint) / maxspeed
            starttime = s_window[tk[0]][0] - traveltime #depart time
            path[i].append(startid)
            route[i].append(startpoint)
            load[i].append(0)
            arrive[i].append(7*60) #working hours for depot
            leave[i].append(starttime)
            stay[i].append(0)
            late[i].append(0)
            for next in tk:
                nextpoint = np.array(s_points[next]) 
                et = s_window[next][0]
                lt = s_window[next][1]
                need = s_need[next]
                service = s_service[next]

                cur_leave = leave[i][-1] #leaving time
                cur_point = route[i][-1] #current location
                cur_load = load[i][-1] #current loading
                cur_path = path[i][-1] #current node
                if cur_path == startid:
                    cur_lt = et
                else:
                    cur_lt = s_window[cur_path][1]
                traveltime = np.linalg.norm(cur_point - nextpoint) / maxspeed #travel time
                est_stay = 0
                if policy == 1 and (cur_leave + traveltime <= et and cur_lt + traveltime >= et):
                    est_stay = et - traveltime - cur_leave #duration
                est_arrive = cur_leave + traveltime + est_stay #estimate arriving time if conducting next task
                est_load = cur_load + need #estimate loading when conducting next task
                stay[i].append(est_stay)
                if est_arrive < et or est_load > capacity: # mobile locker return to the warehouse and come back if arrives early or overweight
                    traveltime = np.linalg.norm(cur_point - startpoint) / maxspeed # travel time
                    path[i].append(startid)
                    route[i].append(startpoint)
                    load[i].append(0)
                    arrive[i].append(cur_leave + traveltime)
                    late[i].append(0)
                    traveltime = np.linalg.norm(nextpoint -startpoint) / maxspeed  # travel time
                    leave[i].append(et - traveltime)

                    path[i].append(next)
                    route[i].append(nextpoint)
                    load[i].append(need)
                    arrive[i].append(et)
                    late[i].append(0)
                    leave[i].append(et + service)
                else: #arrive on time or delay
                    path[i].append(next)
                    route[i].append(nextpoint)
                    load[i].append(est_load)
                    arrive[i].append(est_arrive)
                    late[i].append(np.max([0, est_arrive - lt]))
                    leave[i].append(est_arrive + service)
            cur_point = route[i][-1]
            cur_leave = leave[i][-1]
            path[i].append(startid)
            route[i].append(startpoint)
            load[i].append(0)
            traveltime = np.linalg.norm(cur_point - startpoint) / maxspeed  # travel time
            arrive[i].append(cur_leave+traveltime)
            late[i].append(0)
            leave[i].append(0)
            stay[i].append(0)

        for k in range(len(route[i]) - 1):
            point1 = route[i][k]
            point2 = route[i][k+1]
            dis = np.linalg.norm(point1 -point2)
            total_distance += dis
            distance[i].append(dis)
        for la in late[i]:
            total_late += la

    fit = total_distance + 5*total_dispatch + 5*total_late #objective function
    reward = fit #total reward
    if flag:
        return reward, total_dispatch, total_distance, task, path, route, distance, load, arrive, leave, late, stay
    else:
        return reward