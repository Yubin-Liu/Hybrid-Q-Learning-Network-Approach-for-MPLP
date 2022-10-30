import colorsys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + np.random.random() * 10
        l = 50 + np.random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step

    return hls_colors

def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])

    return rgb_colors

def color(value):
    digit = list(map(str, range(10))) + list("ABCDEF")
    if isinstance(value, tuple):
        string = '#'
        for i in value:
            a1 = i // 16
            a2 = i % 16
            string += digit[a1] + digit[a2]
        return string
    elif isinstance(value, str):
        a1 = digit.index(value[1]) * 16 + digit.index(value[2])
        a2 = digit.index(value[3]) * 16 + digit.index(value[4])
        a3 = digit.index(value[5]) * 16 + digit.index(value[6])
        return (a1, a2, a3)

def getColor(number):
    return list(map(lambda x: color(tuple(x)), ncolors(number)))

# 字典转Dataframe
def dict2df(dic, col):
    k = list(dic.keys())
    v = list(dic.values())
    df = pd.DataFrame(list(zip(k, v)), columns=col)
    return df

#固定小数点个数
def fixDecimal(dic, n):
    new_dict = {}
    for i in dic:
        new_dict[i] = []
        lis = dic[i]
        new_list = []
        for x in lis:
            new_list.append(round(x, n))
        new_dict[i] = new_list
    return new_dict

def drawconv(filepath, bestfit, Qnorm):
    t = np.arange(0, len(bestfit))
    plt.rcParams['font.sans-serif'] = ['SimHei']  
    plt_fit = np.array(bestfit)
    plt_norm = np.array(Qnorm)
    plt.figure()
    plt.plot(t, plt_fit)
    plt.xlabel('Number of iteration')
    plt.ylabel('Reward')
    plt.savefig(filepath+'Convergence_of_reward.png')
    plt.figure()
    plt.plot(t, plt_norm)
    plt.xlabel('Number of iteration')
    plt.ylabel('Error of Q-Value')
    plt.savefig(filepath + 'Convergence_of_Q_value.png')
    
def drawallcon(filepath, gene, bestfit_0_ga, bestfit_1_ga, Qnorm_0, Qnorm_1):
    t = np.arange(0, gene)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    bestfit_0_ga = list(bestfit_0_ga)
    bestfit_1_ga = list(bestfit_1_ga)
    
        
    plt_fit_0_ga = np.array(bestfit_0_ga)
    plt_fit_1_ga = np.array(bestfit_1_ga)
    plt_norm0 = np.array(Qnorm_0)
    plt_norm1 = np.array(Qnorm_1)
    
    plt_norm0 = np.delete(plt_norm0, 0)
    plt_norm1 = np.delete(plt_norm1, 0)

    fig, ax = plt.subplots(2, 1)
    ax[0].set_ylabel('Error of Q-Value', color = 'tab:blue')
    ax[0].plot(t, plt_norm0, linewidth = '1.5',label = 'HQM-Policy1', color = '#0343df', linestyle = '-')
    ax2 = ax[0].twinx()
    ax2.plot(t, plt_fit_0_ga, linewidth='1.5', label='GA-Policy1', color='#e50000', linestyle='-')
    ax2.set_ylabel('Value of f(X,Y,Z)', color='tab:red')
    ax[0].legend(loc = 'upper right')
    ax2.legend(loc = 'center right')
    ax[0].tick_params(axis = 'y', labelcolor = 'tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:red')
 
 
    ax[1].set_xlabel('Number of iteration')
    ax[1].set_ylabel('Error of Q-Value', color = 'tab:blue')
    ax[1].plot(t, plt_norm1, linewidth='1.5', label='HQM-Policy2', color='#0343df', linestyle='-')
    ax3 = ax[1].twinx()
    ax3.plot(t, plt_fit_1_ga, linewidth='1.5', label='GA-Policy2', color='#e50000', linestyle='-')
    ax3.set_ylabel('Value of f(X,Y,Z)', color='tab:red')
    ax[1].legend(loc = 'upper right')
    ax3.legend(loc = 'center right')
    ax[1].tick_params(axis = 'y', labelcolor = 'tab:blue')
    ax3.tick_params(axis='y', labelcolor='tab:red')  
    plt.tight_layout()

    fig.savefig(filepath+'Convergence_comparision.png')

def drawallrwd(filepath, gene, bestfit_0_ga, bestfit_1_ga, bestfit_0_rl, bestfit_1_rl):
    
    t= np.arange(0, gene)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt_fit_0_ga = np.array(bestfit_0_ga)
    plt_fit_1_ga = np.array(bestfit_1_ga)
    plt_fit_0_ga = 1 / plt_fit_0_ga
    plt_fit_1_ga = 1 / plt_fit_1_ga


    plt_fit_0_rl = np.array(bestfit_0_rl)
    plt_fit_1_rl = np.array(bestfit_1_rl)
    plt_fit_0_rl = np.delete(plt_fit_0_rl, 0)
    plt_fit_1_rl = np.delete(plt_fit_1_rl, 0)
    

    fig, ax = plt.subplots(2,1)
    ax[0].set_ylabel('Reward', color = 'tab:blue')
    ax[0].plot(t, plt_fit_0_rl, linewidth = '1.5',label = 'HQM-Policy1', color = '#0343df', linestyle = '-')
    ax2 = ax[0].twinx()
    ax2.plot(t, plt_fit_0_ga, linewidth='1.5', label='GA-Policy1', color='#e50000', linestyle='-')
    ax2.set_ylabel('Reward(1/f(X,Y,Z)', color='tab:red')
    ax[0].legend(loc = 'lower right')
    ax2.legend(loc = 'center right')
    ax[0].tick_params(axis = 'y', labelcolor = 'tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:red')
 
 
    ax[1].set_xlabel('Number of iteration')
    ax[1].set_ylabel('Reward', color = 'tab:blue')
    ax[1].plot(t, plt_fit_1_rl, linewidth='1.5', label='HQM-Policy2', color='#0343df', linestyle='-')
    ax3 = ax[1].twinx()
    ax3.plot(t, plt_fit_1_ga, linewidth='1.5', label='GA-Policy2', color='#e50000', linestyle='-')
    ax3.set_ylabel('Reward(1/f(X,Y,Z)', color='tab:red')
    ax[1].legend(loc = 'lower right')
    ax3.legend(loc = 'center right')
    ax[1].tick_params(axis = 'y', labelcolor = 'tab:blue')
    ax3.tick_params(axis='y', labelcolor='tab:red')  
    plt.tight_layout()
     
    fig.savefig(filepath+'Reward_comparision.png')
    
    
def drawallrwd2(filepath, gene, bestfit_0_ga, bestfit_1_ga, bestfit_0_rl, bestfit_1_rl):
    
    t= np.arange(0, gene)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt_fit_0_ga = np.array(bestfit_0_ga)
    plt_fit_1_ga = np.array(bestfit_1_ga)
    plt_fit_0_ga = 1 / plt_fit_0_ga
    plt_fit_1_ga = 1 / plt_fit_1_ga


    plt_fit_0_rl = np.array(bestfit_0_rl)
    plt_fit_1_rl = np.array(bestfit_1_rl)
    plt_fit_0_rl = np.delete(plt_fit_0_rl, 0)
    plt_fit_1_rl = np.delete(plt_fit_1_rl, 0)

    

    fig, ax = plt.subplots(2,1)
    ax[0].set_ylabel('Reward')
    ax[0].plot(t, plt_fit_0_rl, linewidth = '1.5',label = 'HQM-Policy1', color = '#0343df', linestyle = '-')
    ax[0].plot(t, plt_fit_0_ga, linewidth='1.5', label='GA-Policy1', color='#e50000', linestyle='-')
    ax[0].legend(loc = 'lower right')

 
 
    ax[1].set_xlabel('Number of iteration')
    ax[1].set_ylabel('Reward')
    ax[1].plot(t, plt_fit_1_rl, linewidth='1.5', label='HQM-Policy2', color='#0343df', linestyle='-')
    ax[1].plot(t, plt_fit_1_ga, linewidth='1.5', label='GA-Policy2', color='#e50000', linestyle='-')
    ax[1].legend(loc = 'lower right')
    plt.tight_layout()
     
    fig.savefig(filepath+'Reward_comparision_1.png')



def drawroute(filepath, customer_points, route):
    startpoint = [0, 0]
    num = len(route)
    colorlist = getColor(num+2)
    plt.rcParams['font.sans-serif'] = ['SimHei']  
    plt.figure()
    customer_points = np.array(customer_points)

    for i in route:
        pointlist = route[i]
        if len(pointlist) == 0:
            continue
        cur_color = colorlist[i]
        x = []
        y = []
        for point in pointlist:
            x.append(point[0])
            y.append(point[1])
        x = np.array(x)
        y = np.array(y)
        plt.plot(x, y, '*', color=cur_color)
        plt.plot(x, y, '--', color=cur_color)
    plt.plot(startpoint[0], startpoint[1], '*', color=colorlist[-1], label='Depot')
    plt.plot(customer_points[:, 0], customer_points[:, 1], '.', color=colorlist[-2], label='Customer')
    plt.legend(loc='upper left')
    plt.xlabel('x/km')
    plt.ylabel('y/km')
    plt.savefig(filepath + 'Route.png')

    for i in route:
        pointlist = route[i]
        if len(pointlist) == 0:
            continue
        cur_color = colorlist[i]
        x = []
        y = []
        for point in pointlist:
            x.append(point[0])
            y.append(point[1])
        x = np.array(x)
        y = np.array(y)
        plt.figure()
        customer_points = np.array(customer_points)
        plt.plot(customer_points[:, 0], customer_points[:, 1], '.', color=colorlist[-2], label='Customer')
        plt.plot(x, y, '*', color=cur_color, label='Parking Space')
        plt.plot(x, y, '--', color=cur_color, label='Route')
        plt.plot(startpoint[0], startpoint[1], '*', color=colorlist[-1], label='Depot')
        plt.legend(loc='upper left')
        plt.xlabel('x/km')
        plt.ylabel('y/km')
        plt.title('Mobile Locker'+str(i))
        plt.savefig(filepath + 'Mobile Locker'+str(i)+'Route.png')

def saveresult(filepath, arrive, leave, late, total_dispatch, total_distance, task, path, load, stay):
    filename = filepath + 'Result.xlsx'
    arrive = fixDecimal(arrive, 1)
    leave = fixDecimal(leave, 1)
    late = fixDecimal(late, 1)
    stay = fixDecimal(stay, 1)

    df_dispatch = pd.DataFrame([total_dispatch], columns=['Quantity of Locker'])
    df_distance = pd.DataFrame([total_distance], columns=['Driving Distance'])
    df_task = dict2df(task, col=['Locker_ID', 'Task'])
    df_path = dict2df(path, col=['Locker_ID', 'Route_ID'])
    df_arrive = dict2df(arrive, col=['Locker_ID', 'Arriving_Time'])
    df_leave = dict2df(leave, col=['Locker_ID', 'Leaving_Time'])
    df_late = dict2df(late, col=['Locker_ID', 'Delay'])
    df_load = dict2df(load, col=['Locker_ID', 'Loading'])
    df_stay = dict2df(stay, col=['Locker_ID', 'Service_Time'])
    writer = pd.ExcelWriter(filename)
    df_dispatch.to_excel(writer, sheet_name='Quantity of Locker')
    df_distance.to_excel(writer, sheet_name='Driving Distance')
    df_task.to_excel(writer, sheet_name='Schedule')
    df_path.to_excel(writer, sheet_name='Route')
    df_arrive.to_excel(writer, sheet_name='Arriving_Time')
    df_leave.to_excel(writer, sheet_name='Leaving_Time')
    df_late.to_excel(writer, sheet_name='Delay')
    df_load.to_excel(writer, sheet_name='Loading')
    df_stay.to_excel(writer, sheet_name='Service_Time')
    writer.save()