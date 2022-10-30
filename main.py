from HQM import Qlearning
from reward_GA import getReward
from reward_HQM import getReward1
from generate import getData
from genetic_algorithm import *
from result import drawconv, drawallcon, drawallrwd, drawallrwd2, drawroute, saveresult
import numpy as np

if __name__ == "__main__":
    '''Initialise parameters'''
    radius_limit = [0.1, 1.0]  # customer walking range，km
    wait_limit = [60 - 30, 60 + 30, 1]  # time span of customer，min
    stop_limit = [50 - 20, 50 + 20]  # time span of parking space
    maxspeed = 0.7
    capacity = 20  # locker capacity
    max_car = 35  # maximum number of locker
    gene = 1000 #iteration number
    popsize = 100 #popsize

    ###HQM_BTD
    policy = 0 #policy 0-locker back to depot if time windows not satisfy， 1-wait in the current parking space umtil time windows are met
    filepath0 = './result_0/'
    filepath1 = './result_0_ga/'
    filepath_0_all = './result_0_all/'
    data, s_points, customer_points, max_radius, mean_window, mean_stop = getData(radius_limit, wait_limit,
                                                                                  stop_limit, capacity,
                                                                                  filepath0, saveflag=1)
    bestfit_0_rl, bestx_0_rl, Qnorm_0, initial_result_q0 = Qlearning(ncar=max_car, npoints=len(s_points), maxspeed = maxspeed, capacity = capacity, gene=gene, popsize=popsize, policy=policy, data=data)
    reward, total_dispatch, total_distance, task, path, route, distance, load, arrive, leave, late, stay = getReward1(
        x=bestx_0_rl, ncar=max_car, data=data, maxspeed = maxspeed, capacity = capacity, policy=policy, flag=1)
    print('Initial result:' + str(initial_result_q0))
    print('Final result:' + str(reward))
    print('Improvement rate:' + str((reward - initial_result_q0) / initial_result_q0))
    drawroute(filepath0, customer_points, route)
    saveresult(filepath0, arrive, leave, late, total_dispatch, total_distance, task, path, load, stay)
    drawconv(filepath0, bestfit_0_rl, Qnorm_0)
    
    ###GA_BTD
    bestfit_0_ga, bestx_0_ga, = ga(ncar=max_car, npoints=len(s_points),  gene=gene, popsize=popsize, maxspeed = maxspeed, capacity = capacity, policy=policy, data=data)
    reward1, total_dispatch1, total_distance1, task1, path1, route1, distance1, load1, arrive1, leave1, late1, stay1 = getReward(
        x=bestx_0_ga, ncar=max_car, data=data, maxspeed = maxspeed, capacity = capacity, policy=policy, flag=1)
    print('Initial result:' + str(1 / bestfit_0_ga[0]))
    print('Final result' + str(1 / bestfit_0_ga[-1]))
    initial_reward0 = 1 / bestfit_0_ga[0]
    final_reward0 = 1 / bestfit_0_ga[-1]
    print('Improvement rate:' + str((final_reward0 - initial_reward0) / initial_reward0))
    drawroute(filepath1, customer_points, route1)
    saveresult(filepath1, arrive1, leave1, late1, total_dispatch1, total_distance1, task1, path1, load1, stay1)


    ###HQM_HCPS
    policy = 1 #policy 0-locker back to depot if time windows not satisfy， 1-wait in the current parking space umtil time windows are met
    filepath2 = './result_1/'
    filepath3 = './result_1_ga/'
    filepath_all = './result_all/'
    
    bestfit_1_rl, bestx_1_rl, Qnorm_1, initial_result_q = Qlearning(ncar=max_car, npoints=len(s_points), maxspeed = maxspeed, capacity = capacity, gene=gene, popsize=popsize, policy=policy, data=data)
    reward2, total_dispatch2, total_distance2, task2, path2, route2, distance2, load2, arrive2, leave2, late2, stay2 = getReward1(
        x=bestx_1_rl, ncar=max_car, data=data, maxspeed = maxspeed, capacity = capacity, policy=policy, flag=1)
    print('Initial result:' + str(initial_result_q))
    print('Final result:' + str(reward2))
    print('Improvement rate:' + str((reward2 - initial_result_q) / initial_result_q))
    drawroute(filepath2, customer_points, route2)
    saveresult(filepath2, arrive2, leave2, late2, total_dispatch2, total_distance2, task2, path2, load2, stay2)
    drawconv(filepath2, bestfit_1_rl, Qnorm_1)

    ###GA_HCPS
    bestfit_1_ga, bestx_1_ga  = ga(ncar=max_car, npoints=len(s_points),  gene=gene, popsize=popsize, maxspeed = maxspeed, capacity = capacity, policy=policy, data=data)
    reward3, total_dispatch3, total_distance3, task3, path3, route3, distance3, load3, arrive3, leave3, late3, stay3 = getReward(
        x=bestx_1_ga, ncar=max_car, data=data, maxspeed = maxspeed, capacity = capacity, policy=policy, flag=1)
    print('Initial result:' + str(1 / bestfit_1_ga[0]))
    print('Final result:' + str(1 / bestfit_1_ga[-1]))
    initial_reward1 = 1 / bestfit_1_ga[0]
    final_reward1 = 1 / bestfit_1_ga[-1]
    print('Improvement rate:' + str((final_reward1 - initial_reward1) / initial_reward1))
    drawroute(filepath3, customer_points, route3)
    saveresult(filepath3, arrive3, leave3, late3, total_dispatch3, total_distance3, task3, path3, load3, stay3)

    drawallcon(filepath_all, gene, bestfit_0_ga, bestfit_1_ga, Qnorm_0, Qnorm_1)
    drawallrwd(filepath_all, gene, bestfit_0_ga, bestfit_1_ga, bestfit_0_rl, bestfit_1_rl)
    drawallrwd2(filepath_all, gene, bestfit_0_ga, bestfit_1_ga, bestfit_0_rl, bestfit_1_rl)
    
    
    

    
    
