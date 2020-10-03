import numpy as np
import csv
import sys
import os
import matplotlib.pyplot as plt
from plot_scan_matching_accuracy import *

lengths = [100, 200, 300, 400, 500, 600, 700, 800]

# Calculates path length along the trajectory
def trajectoryDistances(poses):
    dist = [0]
    for i in range(1, len(poses)):
        P1 = poses[i - 1]
        P2 = poses[i]
        dx = P1[0, 2] - P2[0, 2]
        dy = P1[1, 2] - P2[1, 2]
        dist.append(dist[i-1] + np.sqrt(dx**2 + dy**2))
    return dist

def lastFrameFromSegmentLength(dist, first_frame, length):
    for i in range(first_frame, len(dist)):
        if dist[i] > dist[first_frame] + length:
            return i
    return -1

def rotationError(pose_error):
    return abs(np.arcsin(pose_error[0, 1]))

def translationError(pose_error):
    return np.sqrt(pose_error[0, 2]**2 + pose_error[1, 2]**2)

def get_inverse_tf(T):
    T2 = np.identity(3)
    R = T[0:2, 0:2]
    t = T[0:2, 2]
    t = np.reshape(t, (2, 1))
    T2[0:2, 0:2] = R.transpose()
    t = np.matmul(-1 * R.transpose(), t)
    T2[0, 2] = t[0]
    T2[1, 2] = t[1]
    return T2

def calcSequenceErrors(poses_gt, poses_res):
    err = []
    step_size = 4 # Every second
    # Pre-compute distances from ground truth as reference
    dist = trajectoryDistances(poses_gt)

    for first_frame in range(0, len(poses_gt), step_size):
        for i in range(0, len(lengths)):
            length = lengths[i]
            last_frame = lastFrameFromSegmentLength(dist, first_frame, length)
            if last_frame == -1:
                continue
            # Compute rotational and translation errors
            pose_delta_gt = np.matmul(get_inverse_tf(poses_gt[first_frame]), poses_gt[last_frame])
            pose_delta_res = np.matmul(get_inverse_tf(poses_res[first_frame]), poses_res[last_frame])
            pose_error = np.matmul(get_inverse_tf(pose_delta_res), pose_delta_gt)
            r_err = rotationError(pose_error)
            t_err = translationError(pose_error)
            # Approx speed
            num_frames = float(last_frame - first_frame + 1)
            speed = float(length) / (0.25 * num_frames)
            err.append([first_frame, r_err/float(length), t_err/float(length), length, speed])
    return err

def saveSequenceErrors(err, file_name):
    with open(file_name, "w") as f:
        for e in err:
            f.write("{},{},{},{},{}\n".format(e[0], e[1], e[2], e[3], e[4]))

def saveErrorPlots(errlist, filename):
    fig, axs = plt.subplots(2, 2, tight_layout=True)
    axs[0, 0].set_title('Translation Error vs. Path Length')
    axs[0, 1].set_title('Rotation Error vs. Path Length')
    axs[1, 0].set_title('Translation Error vs. Speed')
    axs[1, 1].set_title('Rotation Error vs. Speed')

    for j in range(0, len(errlist)):
        err = errlist[j]
        t_len_err = []
        r_len_err = []
        t_vel_err = []
        r_vel_err = []
        for i in range(0, len(lengths)):
            length = lengths[i]
            num = 0
            t_err = 0
            r_err = 0
            for e in err:
                if e[3] - length < 1.0:
                    t_err += e[2]
                    r_err += e[1]
                    num += 1
            if num == 0:
                break
            t_len_err.append(t_err / float(num))
            r_len_err.append(r_err / float(num))

        for v in range(2, 26):
            num = 0
            t_err = 0
            r_err = 0
            for e in err:
                if v - e[4] < 2.0:
                    t_err += e[2]
                    r_err += e[1]
                    num += 1
            if num == 0:
                break
            t_vel_err.append(t_err / float(num))
            r_vel_err.append(r_err / float(num))

        vx = np.arange(2, 26, 1)
        l = len(t_len_err)
        m = len(t_vel_err)
        if j == 0:
            axs[0, 0].plot(lengths[:l], t_len_err, 'ob-', label='RIGID')
            axs[0, 1].plot(lengths[:l], r_len_err, 'ob-', label='RIGID')
            axs[1, 0].plot(vx[:m], t_vel_err, 'ob-', label='RIGID')
            axs[1, 1].plot(vx[:m], r_vel_err, 'ob-', label='RIGID')
        if j == 1:
            axs[0, 0].plot(lengths[:l], t_len_err, 'Dr-', label='MDRANSAC')
            axs[0, 1].plot(lengths[:l], r_len_err, 'Dr-', label='MDRANSAC')
            axs[1, 0].plot(vx[:m], t_vel_err, 'Dr-', label='MDRANSAC')
            axs[1, 1].plot(vx[:m], r_vel_err, 'Dr-', label='MDRANSAC')
        if j == 2:
            axs[0, 0].plot(lengths[:l], t_len_err, 'og-', label='DOPPLER')
            axs[0, 1].plot(lengths[:l], r_len_err, 'og-', label='DOPPLER')
            axs[1, 0].plot(vx[:m], t_vel_err, 'og-', label='DOPPLER')
            axs[1, 1].plot(vx[:m], r_vel_err, 'og-', label='DOPPLER')

    axLine, axLabel = axs[0, 0].get_legend_handles_labels()

    fig.legend(axLine, axLabel, loc = 'best', fontsize='xx-small')
    plt.savefig(filename)


def getStats(err):
    t_err = 0
    r_err = 0
    for e in err:
        t_err += e[2]
        r_err += e[1]
    t_err /= float(len(err))
    r_err /= float(len(err))
    return t_err, r_err

if __name__ == '__main__':
    # afile = 'accuracy.csv'
    # if len(sys.argv) > 1:
    #     afile = sys.argv[1]
    # print(afile)

    ff = os.listdir('.')
    files = []
    for f in ff:
        if 'accuracy' in f:
            files.append(f)

    err_rigid = []
    err_md = []
    err_dopp = []

    for files in files:
        T_gt = np.identity(3)
        T_res = np.identity(3)
        T_md = np.identity(3)
        T_dopp = np.identity(3)
        poses_gt = []
        poses_res = []
        poses_md = []
        poses_dopp = []
        with open(afile) as f:
            reader = csv.reader(f, delimiter=',')
            i = 0
            for row in reader:
                if i == 0:
                    i = 1
                    continue
                # Create transformation matrices
                T_gt_ = get_transform(float(row[3]), float(row[4]), float(row[5]))
                T_res_ = get_transform(float(row[0]), float(row[1]), float(row[2]))
                T_md_ = get_transform(float(row[8]), float(row[9]), float(row[10]))
                T_dopp_ = get_transform(float(row[11]), float(row[12]), float(row[13]))
                T_gt = np.matmul(T_gt, T_gt_)
                T_res = np.matmul(T_res, T_res_)
                T_md = np.matmul(T_md, T_md_)
                T_dopp = np.matmul(T_dopp, T_dopp_)

                R_gt = T_gt[0:2,0:2]
                R_res = T_res[0:2,0:2]
                R_md = T_md[0:2,0:2]
                R_dopp = T_dopp[0:2,0:2]
                if np.linalg.det(R_gt) != 1.0:
                    enforce_orthogonality(R_gt)
                    T_gt[0:2,0:2] = R_gt
                if np.linalg.det(R_res) != 1.0:
                    enforce_orthogonality(R_res)
                    T_res[0:2,0:2] = R_res
                if np.linalg.det(R_md) != 1.0:
                    enforce_orthogonality(R_md)
                    T_md[0:2,0:2] = R_md
                if np.linalg.det(R_dopp) != 1.0:
                    enforce_orthogonality(R_dopp)
                    T_dopp[0:2,0:2] = R_dopp

                poses_gt.append(T_gt)
                poses_res.append(T_res)
                poses_md.append(T_md)
                poses_dopp.append(T_dopp)

        err_rigid.extend(calcSequenceErrors(poses_gt, poses_res))
        err_md.extend(calcSequenceErrors(poses_gt, poses_md))
        err_dopp.extend(calcSequenceErrors(poses_gt, poses_dopp))

    saveSequenceErrors(err_rigid, 'pose_error_rigid.csv')
    saveSequenceErrors(err_md, 'pose_error_mdransac.csv')
    saveSequenceErrors(err_dopp, 'pose_error_dopp.csv')

    saveErrorPlots([err_rigid, err_md, err_dopp], 'pose_error.png')
    t_err, r_err = getStats(err)
    print('RIGID:')
    print('t_err: {} %'.format(t_err * 100))
    print('r_err: {} deg/m'.format(r_err * 180 / np.pi))
    t_err, r_err = getStats(err2)
    print('MDRANSAC:')
    print('t_err: {} %'.format(t_err * 100))
    print('r_err: {} deg/m'.format(r_err * 180 / np.pi))
    t_err, r_err = getStats(err3)
    print('DOPPLER:')
    print('t_err: {} %'.format(t_err * 100))
    print('r_err: {} deg/m'.format(r_err * 180 / np.pi))
