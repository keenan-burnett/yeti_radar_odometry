import csv
import matplotlib.pyplot as plt
import numpy as np

dx = []
dy = []
dyaw = []

with open('accuracy.csv') as f:
    reader = csv.reader(f, delimiter=',')
    i = 0
    for row in reader:
        if i == 0:
            i = 1
            continue
        dx.append(float(row[3]) - float(row[0]))
        dy.append(float(row[4]) - float(row[1]))
        dyaw.append(180 * (float(row[5]) - float(row[2])) / np.pi)

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
T_res = np.identity(3)

xgt = []
ygt = []
xres = []
yres = []

with open('accuracy.csv') as f:
    reader = csv.reader(f, delimiter=',')
    i = 0
    for row in reader:
        if i == 0:
            i = 1
            continue

        # Create transformation matrices
        T_gt_ = np.array([[np.cos(float(row[2])), -np.sin(float(row[2])), float(row[0])],
                          [np.sin(float(row[2])), np.cos(float(row[2])), float(row[1])],
                          [0, 0, 1]])
        T_res_ = np.array([[np.cos(float(row[5])), -np.sin(float(row[5])), float(row[3])],
                          [np.sin(float(row[5])), np.cos(float(row[5])), float(row[4])],
                          [0, 0, 1]])

        T_gt = T_gt_.dot(T_gt)
        T_res = T_res_.dot(T_res)
        xgt.append(T_gt[0, 2])
        ygt.append(T_gt[1, 2])
        xres.append(T_res[0, 2])
        yres.append(T_res[1, 2])

xgt = np.array(xgt)
ygt = np.array(ygt)
xres = np.array(xres)
yres = np.array(yres)

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.plot(xgt, ygt, 'b', linewidth=2, label='Ground Truth')
ax.plot(xres, yres, 'r', linewidth=2, label='Radar Odometry')
ax.set_title('Ground Truth vs. Radar Odometry (Demo Sequence)')
ax.legend(loc='upper left')
plt.savefig('trajectory.png')

# plt.show()
