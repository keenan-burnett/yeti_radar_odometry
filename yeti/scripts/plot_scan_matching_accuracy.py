import csv
import matplotlib.pyplot as plt
import numpy as np

def enforce_orthogonality(R):
    epsilon = 0.001
    if abs(R[0, 0] - R[1, 1]) > epsilon or abs(R[1, 0] + R[0, 1]) > epsilon:
        print("ERROR: this is not a proper rigid transformation!")
    a = (R[0, 0] + R[1, 1]) / 2;
    b = (-R[1, 0] + R[0, 1]) / 2;
    sum = np.sqrt(a**2 + b**2);
    a /= sum;
    b /= sum;
    R[0, 0] = a; R[0, 1] = b;
    R[1, 0] = -b; R[1, 1] = a;

def get_transform(x, y, theta):
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    if np.linalg.det(R) != 1.0:
        enforce_orthogonality(R)
    xbar = np.array([x, y])
    xbar = np.reshape(xbar, (2, 1))
    T = np.identity(3)
    T[0:2, 0:2] = R
    xbar = np.matmul(-R, xbar)
    T[0, 2] = xbar[0]
    T[1, 2] = xbar[1]
    return T

def get_ins_transformation(time1, time2, gpsfile):

    assert(time2 > time1)
    time1vars = []
    time2vars = []

    def get_vars(row, x1, y1, t1, theta1, time):
        t2 = int(row[0])
        x2 = float(row[5])
        y2 = float(row[6])
        theta2 = float(row[14])
        delta = float(time1 - t1) / float(t2 - t1)
        x = x1 + (x2 - x1) * delta
        y = y1 + (y2 - y1) * delta
        theta = theta1 + (theta2 - theta1) * delta
        return [x, y, theta]

    with open(gpsfile) as f:
        reader = csv.reader(f, delimiter=',')
        # Find timestamps in GPS file that bound time1
        x1 = 0
        y1 = 0
        t1 = 0
        theta1 = 0
        i = 0
        for row in reader:
            if i == 0:
                i = 1
                continue
            if (int(row[0]) - time1) > 0 and len(time1vars) == 0:
                time1vars = [x1, y1, theta1]
                # time1vars = get_vars(row, x1, y1, t1, theta1, time1)
            if (int(row[0]) - time2) > 0 and len(time2vars) == 0:
                # time2vars = get_vars(row, x1, y1, t1, theta1, time2)
                time2vars = [x1, y1, theta1]
                break

            t1 = int(row[0])
            x1 = float(row[5])
            y1 = float(row[6])
            theta1 = float(row[14])

    xbar = np.array([time2vars[0] - time1vars[0], time2vars[1] - time1vars[1]])
    xbar = np.reshape(xbar, (2, 1))
    theta = time1vars[2]
    dtheta = time2vars[2] - time1vars[2]
    R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    xbar = np.matmul(-R, xbar)
    T = np.identity(3)
    R = np.array([[np.cos(dtheta), -np.sin(dtheta)], [np.sin(dtheta), np.cos(dtheta)]])
    T[0:2, 0:2] = R
    T[0, 2] = xbar[0]
    T[1, 2] = xbar[1]

    return T

def extract_translation(T):
    R = T[0:2, 0:2]
    t = T[0:2, 2]
    t = np.reshape(t, (2, 1))
    t = np.matmul(-R.transpose(), t)
    return t[0], t[1]

if __name__ == '__main__':

    dx = []
    dy = []
    dyaw = []

    gpsfile = '/home/keenan/radar_ws/data/2019-01-16-14-15-33-radar-oxford-10k/gps/ins.csv'

    with open('accuracy.csv') as f:
        reader = csv.reader(f, delimiter=',')
        i = 0
        for row in reader:
            if i == 0:
                i = 1
                continue
            dx.append(float(row[3]) - float(row[8]))
            dy.append(float(row[4]) - float(row[9]))
            dyaw.append(180 * (float(row[5]) - float(row[10])) / np.pi)

    dx = np.array(dx)
    dy = np.array(dy)
    dyaw = np.array(dyaw)

    dr = np.sqrt(dx**2 + dy**2)
    print('translation error:')
    print(np.mean(dr))
    print(np.median(dr))
    print(np.sqrt(np.mean((dr - np.median(dr))**2)))
    print('rotation error:')
    print(np.mean(dyaw))
    print(np.median(dyaw))
    print(np.sqrt(np.mean((dyaw - np.median(dyaw))**2)))

    fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True, figsize=(16, 5))
    axs[0].hist(dx, bins=20)
    axs[0].set_title('Translation Error X (m)')
    axs[1].hist(dy, bins=20)
    axs[1].set_title('Translation Error Y (m)')
    axs[2].hist(dyaw, bins=20)
    axs[2].set_title('Rotation Error Yaw (deg)')

    plt.savefig('accuracy_hist.png')

    # Create plot of the trajectories
    T_gt = np.identity(3)
    T_rigid = np.identity(3)
    T_md = np.identity(3)
    T_gps = np.identity(3)

    xgt = []
    ygt = []
    xrigid = []
    yrigid = []
    xmd = []
    ymd = []
    xgps = []
    ygps = []

    with open('accuracy.csv') as f:
        reader = csv.reader(f, delimiter=',')
        i = 0
        for row in reader:
            if i == 0:
                i = 1
                continue
            # Create transformation matrices
            T_gt_ = get_transform(float(row[3]), float(row[4]), float(row[5]))
            T_rigid_ = get_transform(float(row[0]), float(row[1]), float(row[2]))
            T_md_ = get_transform(float(row[8]), float(row[9]), float(row[10]))
            T_gt = np.matmul(T_gt, T_gt_)
            T_rigid = np.matmul(T_rigid, T_rigid_)
            T_md = np.matmul(T_md, T_md_)

            R_gt = T_gt[0:2,0:2]
            R_rigid = T_rigid[0:2,0:2]
            R_md = T_md[0:2,0:2]
            if np.linalg.det(R_gt) != 1.0:
                enforce_orthogonality(R_gt)
                T_gt[0:2,0:2] = R_gt
            if np.linalg.det(R_rigid) != 1.0:
                enforce_orthogonality(R_rigid)
                T_rigid[0:2,0:2] = R_rigid
            if np.linalg.det(R_md) != 1.0:
                enforce_orthogonality(R_md)
                T_md[0:2,0:2] = R_md

            # Get GPS ground truth between the frames
            time1 = int(row[6])
            time2 = int(row[7])
            T_gps_ = get_ins_transformation(time1, time2, gpsfile)
            T_gps = np.matmul(T_gps, T_gps_)
            R_gps = T_gps[0:2,0:2]
            if np.linalg.det(R_gps) != 1.0:
                enforce_orthogonality(R_gps)
                T_gps[0:2,0:2] = R_gps

            xgt.append(T_gt[0, 2])
            ygt.append(T_gt[1, 2])
            xrigid.append(T_rigid[0, 2])
            yrigid.append(T_rigid[1, 2])
            xmd.append(T_md[0, 2])
            ymd.append(T_md[1, 2])
            xgps.append(T_gps[0, 2])
            ygps.append(T_gps[1, 2])

    xgt = np.array(xgt)
    ygt = np.array(ygt)
    xrigid = np.array(xrigid)
    yrigid = np.array(yrigid)
    xmd = np.array(xmd)
    ymd = np.array(ymd)
    xgps = np.array(xgps)
    ygps = np.array(ygps)

    fig, ax = plt.subplots(tight_layout=True)
    ax.set_aspect('equal')
    ax.plot(xgt, ygt, 'k', linewidth=2, label='Ground Truth')
    ax.plot(xrigid, yrigid, 'r', linewidth=2, label='RIGID')
    ax.plot(xmd, ymd, 'b', linewidth=2, label='MDRANSAC')
    ax.plot(xgps, ygps, 'g', linewidth=2, label='GPS')
    ax.set_title('Ground Truth vs. Radar Odometry (Demo Sequence)')
    ax.legend(loc="upper left", fontsize='xx-small')
    plt.savefig('trajectory.png')
    # plt.show()
