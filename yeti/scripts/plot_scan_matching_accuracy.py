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


fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
axs[0].hist(dx, bins=20)
axs[0].set_title('Translation Error X (m)')
axs[1].hist(dy, bins=20)
axs[1].set_title('Translation Error Y (m)')
axs[2].hist(dyaw, bins=20)
axs[2].set_title('Rotation Error Yaw (deg)')



plt.show()
