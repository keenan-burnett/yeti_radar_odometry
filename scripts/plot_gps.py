import csv
import matplotlib.pyplot as plt
import numpy as np
import geodesy.utm

gpsfile = '/home/keenan/radar_ws/data/2019-01-10-14-36-48-radar-oxford-10k-partial/gps/ins.csv'

x = []
y = []

with open(gpsfile) as f:
    reader = csv.reader(f, delimiter=',')
    i = 0
    for row in reader:
        if i == 0:
            i = 1
            continue
        x.append(row[6])
        y.append(row[5])

x = np.array(x)
y = np.array(y)
plt.plot(x, y)
plt.show()
