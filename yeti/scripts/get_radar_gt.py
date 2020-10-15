import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
from evaluate_odometry import get_inverse_tf
from plot_scan_matching_accuracy import enforce_orthogonality

# Given a ROS time (in seconds), ROS/GPS time synch parameters, and a ground truth file,
# This function returns ground truth [rostime, x, y, theta, v, w] interpolated at the exact rostime.
def get_groundtruth(rostime, synch_parameters, gtlines, gt_times):
    gpstime = rostime - (synch_parameters[1] + synch_parameters[0] * rostime)
    print(gpstime)

    # lines = open(gtfile, 'r').read().splitlines()

    diff = 0.1
    closest_time = -1
    closest_index = -1
    for i in range(0, len(gt_times)):
        # line = gtlines[i]
        # parts = line.split(',')
        # gt_time = float(parts[0])
        gt_time = gt_times[i]
        delta = abs(gpstime - gt_time)
        if delta < diff:
            diff = delta
            closest_time = gt_time
            closest_index = i
    assert(closest_index >= 0)

    # Interpolate to get more accurate ground truth
    indices = []
    if closest_time < gpstime:
        indices = [closest_index, closest_index + 1]
    else:
        indices = [closest_index - 1, closest_index]

    m1 = gtlines[indices[0]].split(',')
    m2 = gtlines[indices[1]].split(',')

    t1 = float(m1[0])
    t2 = float(m2[0])
    ratio = (gpstime - t1) / (t2 - t1)

    x1 = float(m1[1])
    y1 = float(m1[2])
    v1 = np.sqrt(float(m1[4])**2 + float(m1[5])**2)
    theta1 = float(m1[9])
    w1 = float(m1[10])

    x2 = float(m2[1])
    y2 = float(m2[2])
    v2 = np.sqrt(float(m2[4])**2 + float(m2[5])**2)
    theta2 = float(m2[9])
    w2 = float(m2[10])

    x = x1 + ratio * (x2 - x1)
    y = y1 + ratio * (y2 - y1)
    v = v1 + ratio * (v2 - v1)
    theta = theta1 + ratio * (theta2 - theta1)
    w = w1 + ratio * (w2 - w1)

    return [rostime, x, y, theta, v, w]

def get_transform(x, y, theta):
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float64)
    # if np.linalg.det(R) != 1.0:
        # enforce_orthogonality(R)
    T = np.identity(3, dtype=np.float64)
    T[0:2, 0:2] = R
    T[0, 2] = x
    T[1, 2] = y
    return T


# Goal: csv file with fname1, fname2, x, y, theta, v1, w1, v2, w2
# v1 and v2 >= 14

T_navtech = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

if __name__ == '__main__':

    project = '/home/keenan/Documents/data/boreas/2020_10_06'
    radar_files = os.listdir(project + '/radar/')
    radar_files.sort()

    timefile = project + '/ros_and_gps_time.csv'
    gtfile = project + '/applanix_post_process_gps_icra.csv'

    # Synchronize ROS time and GPS time
    deltas = []
    rostimes = []
    gpstimes = []
    timelines = open(timefile, 'r').read().splitlines()
    for line in timelines:
        parts = line.split(',')
        rostime = int(parts[0])
        rostime /= 1.0e9
        gpstime = float(parts[1])
        rostimes.append(rostime)
        gpstimes.append(gpstime)
        deltas.append(rostime - gpstime)

    deltas = np.array(deltas)
    rostimes = np.array(rostimes)
    p = np.polyfit(rostimes, deltas, deg=1)
    print('offset: {} skew: {}'.format(p[1], p[0]))


    gtlines = open(gtfile, 'r').read().splitlines()
    gt_times = []
    for line in gtlines:
        gt_times.append(float(line.split(',')[0]))

    groundtruth1 = []
    groundtruth2 = []
    for file in radar_files:
        timestamp = int(file.split('.')[0])
        timestamp /= 1.0e9
        print(timestamp)
        if timestamp < 1602033998:
            gt = get_groundtruth(timestamp, p, gtlines, gt_times)
            gt[0] = int(file.split('.')[0])
            groundtruth1.append(gt)
        elif timestamp > 1602034049:
            gt = get_groundtruth(timestamp, p, gtlines, gt_times)
            gt[0] = int(file.split('.')[0])
            groundtruth2.append(gt)

    g1 = open(project + '/gt1', 'r')
    g2 = open(project + '/gt2', 'r')
    # pickle.dump(groundtruth1, g1)
    # pickle.dump(groundtruth2, g2)
    groundtruth1 = pickle.load(g1)
    groundtruth2 = pickle.load(g2)

    outfile = project + '/radar_groundtruth.csv'
    f = open(outfile, 'w')

    minvel = 10.0
    for gt in groundtruth1:
        v = gt[4]
        if v < minvel:
            continue

        x = gt[1]
        y = gt[2]
        theta = gt[3]
        w = gt[5]

        mind = 10.0
        closest = -1
        for i in range(0, len(groundtruth2)):
            gt2 = groundtruth2[i]
            v2 = gt2[4]
            if v2 < minvel:
                continue
            x2 = gt2[1]
            y2 = gt2[2]
            d = np.sqrt((x - x2)**2 + (y - y2)**2)
            if d < mind:
                mind = d
                closest = i

        if closest >= 0:
            T_i_r1 = get_transform(x, y, theta)
            T_i_r1 = np.matmul(T_i_r1, T_navtech)

            time2 = groundtruth2[closest][0]
            x2 = groundtruth2[closest][1]
            y2 = groundtruth2[closest][2]
            theta2 = groundtruth2[closest][3]
            v2 = groundtruth2[closest][4]
            w2 = groundtruth2[closest][5]

            T_i_r2 = get_transform(x2, y2, theta2)
            T_i_r2 = np.matmul(T_i_r2, T_navtech)

            T_r1_r2 = np.matmul(get_inverse_tf(T_i_r1), T_i_r2)

            # yaw = -1 * np.arcsin(T_r1_r2[0, 1])
            yaw = theta2 - theta

            f.write('{},{},{},{},{},{},{},{},{},\n'.format(gt[0], time2, T_r1_r2[0, 2], T_r1_r2[1, 2], yaw, v, w, v2, w2))
