import numpy as np
import csv
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
    afile = 'accuracy_large.csv'
    T_gt = np.identity(3)
    T_res = np.identity(3)
    poses_gt = []
    poses_res = []
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
            T_gt = np.matmul(T_gt, T_gt_)
            T_res = np.matmul(T_res, T_res_)

            R_gt = T_gt[0:2,0:2]
            R_res = T_res[0:2,0:2]
            if np.linalg.det(R_gt) != 1.0:
                enforce_orthogonality(R_gt)
                T_gt[0:2,0:2] = R_gt
            if np.linalg.det(R_res) != 1.0:
                enforce_orthogonality(R_res)
                T_res[0:2,0:2] = R_res

            poses_gt.append(T_gt)
            poses_res.append(T_res)

    err = calcSequenceErrors(poses_gt, poses_res)
    saveSequenceErrors(err, 'pose_error.csv')
    t_err, r_err = getStats(err)
    print('t_err: {} %'.format(t_err * 100))
    print('r_err: {} deg/m'.format(r_err * 180 / np.pi))