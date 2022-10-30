import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
# Generate node randomly within the radius of the center
def getRandomPointInCircle(num, radius, centerx, centery):
    samplePoint = []
    for i in range(num):
        while True:
            x = np.random.uniform(-radius, radius)
            y = np.random.uniform(-radius, radius)
            if (x ** 2) + (y ** 2) <= (radius ** 2):
                samplePoint.append([x + centerx, y + centery])
                break
    return samplePoint

# Generate nodes for each cluster
# cluster_num--number of clusters, area--cluster area, size_limit--maximum number of nodes in a single cluster, radius_limit--distribution of nodes within a cluster 
def getRandomPoint(cluster_num, area, size_limit, radius_limit):
    rx = area[0]
    ry = area[1]
    center_points = []
    for i in range(cluster_num):
        center_points.append([rx*np.random.random(), ry*np.random.random()])
    cluster_points = []
    for i in range(cluster_num):
        num = np.random.randint(size_limit[0], size_limit[1])
        ra = np.random.uniform(radius_limit[0], radius_limit[1])
        center = center_points[i]
        samples = getRandomPointInCircle(num, ra, center[0], center[1])
        cluster_points.extend(samples)
    return cluster_points

# Generate time windows and demand for each parking space, and assign customer for each parking space
# cluster_points--parking space, size_limit--maximum number of customer for a parking space, window_limit--time span of customer（min), wait_limit--the tolerance of customer waiting service(min)
def getPointInfo(cluster_points, size_limit, window_limit, wait_limit, need_limit):
    points_id = [] #The customer ID corresponding to the whereabouts
    points_windows = [] #Time-windows of the whereabouts
    points_need = [] #Demand of the whereabouts
    count = 0
    idn = 0
    cluster_size = len(cluster_points) #Total number of customer whereabouts
    wait_seq = np.arange(wait_limit[0], wait_limit[1], wait_limit[2])
    while True:
        size = np.random.randint(size_limit[0], size_limit[1]+1) #number of whereabouts for each customer
        if count+size > cluster_size:
            size = cluster_size - count
        window_seq = np.arange(window_limit[0], window_limit[1], window_limit[2])
        et = np.random.choice(window_seq, replace=False) #starting time
        lt = et + np.random.choice(wait_seq, replace=False) #ending time
        for i in range(size):
            points_windows.append([et, lt])
            points_id.append(idn)
            points_need.append(np.random.randint(need_limit[0], need_limit[1]+1))
            if lt + 60 > window_limit[1]:#Prevent time Windows from overlapping for each customer
                window_seq = np.arange(window_limit[0], et - 60, window_limit[2])
            else:
                window_seq = np.arange(lt, window_limit[1], window_limit[2])
            et = np.random.choice(window_seq, replace=False)
            lt = et + np.random.choice(wait_seq, replace=False)
        idn = idn + 1
        count = count + size
        if count >= cluster_size:
            break
    return points_windows, points_need, points_id

#Calculate the time spend and travel distance from each node to the parking space
def getWalkTime(points, centerid, centers, walkspeed):
    walktime = []
    distance = []
    for i in range(len(points)):
        point = points[i]
        center = centers[centerid[i]]
        dis = np.linalg.norm(point - center)
        walktime.append(dis / walkspeed)
        distance.append(dis)

    return walktime, distance

#Generate time windows and location for tasks
def getVirtualPoint(parkingspace, stoptime, capacity, points, windows, needs, spacenumber, walktime):
    s_points = []  # the location of each task
    s_window = []  # the time windows of each task
    s_need = []  # the demand of each task
    s_service = []  # the service time of each task
    s_clusters = {} # the cluster ID corresponding to the task
    windows = np.array(windows)
    needs = np.array(needs)
    walktime = np.array(walktime)
    count = -1
    for i in range(len(parkingspace)):
        space = parkingspace[i] #parking space
        stime = stoptime[i] #available parking time
        clus = np.argwhere(spacenumber == i).reshape(-1) #Customer ID of each parking space
        window = windows[clus, :] #time windows of whereabouts
        index = np.argsort(window[:, 0]) #sorting time windows
        clus = clus[index] #Customer ID of each parking space
        load = 0
        count = count + 1
        s_clusters[count] = []
        et = windows[clus[0]][0]
        lt = et + stime
        s_points.append(space)
        s_window.append([et, lt])
        for clu in clus:
            if windows[clu][0] >= et and walktime[clu] + windows[clu][0] <= lt and needs[clu] + load < capacity/2:
                load = load + needs[clu]
                s_clusters[count].append(clu)
            else:
                count = count + 1
                et = windows[clu][0]
                lt = et + stime
                s_points.append(space)
                s_window.append([et, lt])
                s_clusters[count] = []
                s_clusters[count].append(clu)
                load = 0

    for i in s_clusters:
        clusters = np.array(s_clusters[i])
        if len(clusters) == 0:
            continue
        need = np.sum(needs[clusters])
        service = round(np.max(walktime[clusters]) + 10, 3)
        s_need.append(need)
        s_service.append(service)

    return s_points, s_window, s_need, s_service, s_clusters

def getData(radius_limit, wait_limit, stop_limit, capacity, filepath, saveflag):
    np.random.seed(10)
    area = [5, 5]  # service range of mobile lockers，km
    parking_num = 10  # quantity of parking space
    #stop_limit = [60, 120] #available time span of parking space,min
    size_limit1 = [20, 21] # customer size of each cluster
    #radius_limit = [0.1, 0.8] #service radius of each parking space,km
    size_limit2 = [1, 4] #range of whereabouts for each customer
    window_limit = [9*60, 18*60, 10] #time span of each customer,min
    #wait_limit = [30, 120, 10] #range of waiting time for each customer,min
    need_limit = [1, 4] #range of demand for each customer
    walkspeed = 0.08 #walking speed，km/min
    #capacity = 30 #capacity of lockers
    customer_points = getRandomPoint(parking_num, area, size_limit1, radius_limit)  # generate location for customer whereabouts
    customer_windows, customer_needs, customer_id = getPointInfo(customer_points, size_limit2, window_limit, wait_limit,
                                                                 need_limit)  # generate time windows and demand for customers
    kmeans = KMeans(n_clusters=parking_num)  # kmeans 
    kmeans.fit(customer_points)  # kmeans
    parkingspace = kmeans.cluster_centers_  # parking space
    parking_stoptime = np.random.randint(stop_limit[0], stop_limit[1] + 1, len(parkingspace))  # available parking time for parking space
    spacenumber = kmeans.labels_  # Parking space ID
    customer_walktime, customer_distance = getWalkTime(customer_points, spacenumber, parkingspace, walkspeed)  # time spend and travel distance from whereabouts to parking space
    s_points, s_window, s_need, s_service, s_clusters = getVirtualPoint(parkingspace, parking_stoptime, capacity,
                                                                        customer_points, customer_windows,
                                                                        customer_needs, spacenumber, customer_walktime)

    data1 = np.column_stack((np.array(parkingspace), parking_stoptime))
    data2 = np.column_stack((np.array(customer_points), np.array(customer_windows), np.array(customer_needs),
                             np.array(customer_id), np.array(spacenumber)))
    data3 = np.column_stack((np.array(s_points), np.array(s_window), np.array(s_need), np.array(s_service)))
    df1 = pd.DataFrame(data1, columns=['x', 'y', 'stoptime'])
    df2 = pd.DataFrame(data2, columns=['x', 'y', 'et', 'lt', 'need', 'id', 'spacenumber'])
    df3 = pd.DataFrame(data3, columns=['x', 'y', 'et', 'lt', 'need', 'servicetime'])

    parking_points = np.array(parkingspace)
    customer_points = np.array(customer_points)
    customer_windows = np.array(customer_windows)
    parking_stoptime = np.array(parking_stoptime)
    max_radius = radius_limit[1]
    mean_window = np.mean([(customer_windows[i][1] - customer_windows[i][0]) for i in range(len(customer_windows))])
    mean_stop = np.mean(parking_stoptime)
    data = {'s_points': s_points, 's_window': s_window, 's_need': s_need, 's_service': s_service}
    if saveflag:
        writer = pd.ExcelWriter(filepath+'Data.xlsx')
        df1.to_excel(writer, sheet_name='parking')
        df2.to_excel(writer, sheet_name='customer')
        df3.to_excel(writer, sheet_name='virtual')
        writer.save()

        plt.rcParams['font.sans-serif'] = ['SimHei']  # 
        plt.figure()
        plt.plot(parking_points[:, 0], parkingspace[:, 1], '*', color='r')
        plt.plot(customer_points[:, 0], customer_points[:, 1], '.')
        plt.legend(['Parking Space', 'Customer Node'])
        plt.xlabel('x/km')
        plt.ylabel('y/km')
        plt.savefig(filepath+'Network_map.png')

    return data, s_points, customer_points, max_radius, mean_window, mean_stop
