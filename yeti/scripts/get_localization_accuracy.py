import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def getRotDiff(r1, r2):
    if r1 < 0:
        r1 = r1 + 2 * np.pi
    if r2 < 0:
        r2 = r2 + 2 * np.pi
    return abs(r1 - r2)

if __name__ == "__main__":
    file = "localization_accuracy_icra.csv"

    dt1 = []
    dt2 = []
    dt3 = []
    dt4 = []
    dt5 = []
    dr1 = []
    dr2 = []
    dr3 = []
    dr4 = []
    dr5 = []

    threshold = 10
    rot_hold = 10

    i = 0
    with open(file, 'r') as f:
        f.readline()
        for line in f:
            i += 1
            if i > 136:
                break
            row = line.split(',')
            gtx = float(row[15])
            gtx *= -1
            gty = float(row[16])
            gtr = np.sqrt(gtx**2 + gty**2)
            gtyaw = float(row[17])
            # if gtyaw < 0:
                # gtyaw = gtyaw + 2 * np.pi
            # if gtr > 10:
                # continue

            dt = np.sqrt((gtx - float(row[0]))**2 + (gty - float(row[1]))**2)
            if dt < threshold:
                dt1.append(dt)
                dr = 180 * getRotDiff(gtyaw, float(row[2])) / np.pi
                if dr < rot_hold:
                    dr1.append(dr)

            dt = np.sqrt((gtx - float(row[3]))**2 + (gty - float(row[4]))**2)
            if dt < threshold:
                dt2.append(dt)
                dr = 180 * getRotDiff(gtyaw, float(row[5])) / np.pi
                if dr < rot_hold:
                    dr2.append(dr)

            dt = np.sqrt((gtx - float(row[6]))**2 + (gty - float(row[7]))**2)
            if dt < threshold:
                dt3.append(dt)
                dr = 180 * getRotDiff(gtyaw, float(row[8])) / np.pi
                if dr < rot_hold:
                    dr3.append(dr)

            dt = np.sqrt((gtx - float(row[9]))**2 + (gty - float(row[10]))**2)
            if dt < threshold:
                dt4.append(dt)
                dr = 180 * getRotDiff(gtyaw, float(row[11])) / np.pi
                if dr < rot_hold:
                    dr4.append(dr)

            dt = np.sqrt((gtx - float(row[12]))**2 + (gty - float(row[13]))**2)
            if dt < threshold:
                dt5.append(dt)
                dr = 180 * getRotDiff(gtyaw, float(row[14])) / np.pi
                if dr < rot_hold:
                    dr5.append(dr)

    dt1 = np.array(dt1)
    dt2 = np.array(dt2)
    dt3 = np.array(dt3)
    dt4 = np.array(dt4)
    dt5 = np.array(dt5)
    dr1 = np.array(dr1)
    dr2 = np.array(dr2)
    dr3 = np.array(dr3)
    dr4 = np.array(dr4)
    dr5 = np.array(dr5)

    print('RIGID: dt: {} sigma_dt: {} dr: {} sigma_dr: {}'.format(np.median(dt1), np.mean((dt1 - np.median(dt1))**2), np.median(dr1), np.mean((dr1 - np.median(dr1))**2)))
    print('DOPP ONLY: {} sigma_dt: {} dr: {} sigma_dr: {}'.format(np.median(dt2), np.mean((dt2 - np.median(dt2))**2), np.median(dr2), np.mean((dr2 - np.median(dr2))**2)))
    print('DOPP + MD: {} sigma_dt: {} dr: {} sigma_dr: {}'.format(np.median(dt3), np.mean((dt3 - np.median(dt3))**2), np.median(dr3), np.mean((dr3 - np.median(dr3))**2)))
    print('MD ONLY: {} sigma_dt: {} dr: {} sigma_dr: {}'.format(np.median(dt4), np.mean((dt4 - np.median(dt4))**2), np.median(dr4), np.mean((dr4 - np.median(dr4))**2)))
    print('MD + DOPP: {} sigma_dt: {} dr: {} sigma_dr: {}'.format(np.median(dt5), np.mean((dt5 - np.median(dt5))**2), np.median(dr5), np.mean((dr5 - np.median(dr5))**2)))

    matplotlib.rcParams.update({'font.size': 13, 'xtick.labelsize' : 14, 'ytick.labelsize' : 14})
    plt.figure(figsize=(10, 5))
    bins = np.arange(0, 9.0, 0.5)
    plt.hist([dt1, dt4, dt5], bins=bins, label=['RIGID', 'MC', 'MC+Dopp'], color=['b', 'r', 'g'], normed=True)
    plt.xlabel('Translation Error (m)', fontsize=15)
    plt.ylabel('Probability', fontsize=15)
    plt.legend(loc='best')
    plt.savefig('localization_accuracy.pdf')
    plt.show()
