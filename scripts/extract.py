#!/usr/bin/python2
from __future__ import print_function

import json
import os
import rospy
import time
import numpy as np
import message_filters
import cv2

import threading
import Queue
from pyquaternion import Quaternion

from nav_msgs.msg import Odometry
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, Image, CompressedImage, Imu

from tf import TransformListener
from cv_bridge import CvBridge, CvBridgeError

import logging
import boto3
from botocore.exceptions import ClientError

import rosbag

project = '2020_09_01'
sub_folders = ['radar', 'lidar', 'camera']
cam_freq = 1   # Limit camera images to this many frames per second
seq_length = 10

bagtimes = [1598985897.857571, 1598988864.5207274, 1598988920.2734032, 1598988968.4253597, 1598989018.9533634, 1598989066.9333131, 1598989111.769795,
    1598989155.8576038, 1598989199.8493268, 1598989243.745398, 1598989286.9094148, 1598989331.8933647, 1598989378.1332874, 1598989425.1493843,
    1598989471.813618, 1598989517.625287, 1598989569.4056506, 1598989619.689403, 1598989669.6573105, 1598989728.1653106, 1598989785.6653063,
    1598989835.7654295, 1598989886.2333837, 1598989937.6533794, 1598989988.9732802, 1598990037.0573351, 1598990085.637276, 1598990137.8214273,
    1598990191.03333, 1598990208.4969854, 1598990259.7613246, 1598990312.0492828, 1598990367.5493488, 1598990415.9053643, 1598990471.609272,
    1598990520.6773267, 1598990566.8413854, 1598990620.1893027, 1598990675.7692657, 1598990732.2332795, 1598990790.4052997, 1598990839.0532715,
    1598990898.7612636, 1598990961.497345, 1598991020.165395, 1598991077.0456839, 1598991134.105351, 1598991187.5053365, 1598991241.0134072,
    1598991293.1293848, 1598991356.3013449, 1598991420.3894076, 1598991475.6732764, 1598991529.2452998, 1598991583.2773438, 1598991634.6174212,
    1598991690.8052511, 1598991749.3772895, 1598991800.1852984, 1598991868.8613098, 1598991925.9932773, 1598991976.7013175, 1598992025.5452874,
    1598992084.7412977, 1598992145.209352, 1598992206.9972925, 1598992271.4747393, 1598992338.2892962, 1598992404.5413747, 1598992470.8333194,
    1598992538.541387, 1598992600.9973068, 1598992663.3692565, 1598992724.0093498, 1598992784.0573237, 1598992844.7733536, 1598992905.5132747,
    1598992965.0813851, 1598993026.6697662]

if __name__ == '__main__':
    # root = '/media/keenan/autorontossd1/BUICK_2020_09_01/'
    project = '/home/keenan/Documents/data/boreas/2020_09_01'
    root = project + "/"
    files = os.listdir(root)
    bagfiles = []
    for file in files:
        if file.split('.')[-1] == 'bag':
            bagfiles.append(file)
    bagfiles.sort()

    # bagtimes = []
    # # Get a time stamp for each rosbag
    # for bagfile in bagfiles:
    #     bag = rosbag.Bag(root + bagfile)
    #     for topic, msg, t in bag.read_messages(topics=['/navsat/odom']):
    #         bagtimes.append(t.to_sec())
    #         break

    seqtimes = [1598986495]

    # seqtimes = []
    # f = open('sequences.csv', 'r')
    # for line in f:
    #     t = line.split(',')[0]
    #     seqtimes.append(int(t))
    # seqtimes.sort()

    # Initialize folder structure if not done already
    if not os.path.isdir(project):
        os.mkdir(project)
    for seqtime in seqtimes:
        s = project + "/" + str(seqtime)
        if not os.path.isdir(s):
            os.mkdir(s)
        for sub in sub_folders:
            ss = s + "/" + sub
            if not os.path.isdir(ss):
                os.mkdir(ss)

    # for i in range(0, len(seqtimes)):
        # seqtime = seqtimes[i]
    seqtime = 1598986495
        # Get the corresponding rosbag for this sequence time (lower bound):
        # lower_bound = 1000000
        # bag_index = -1
        # best_bag_time = 0
        # for j in range(0, len(bagtimes)):
        #     bagtime = bagtimes[j]
        #     diff = seqtime - bagtime
        #     if diff < lower_bound and diff >= 0:
        #         lower_bound = diff
        #         bag_index = j
        #         best_bag_time = bagtime
        #
        # assert(bag_index >= 0)
        #
        # print('Sequence: {} Bag Time: {} Bag: {}'.format(seqtime, best_bag_time, bagfiles[bag_index]))

    start_time = seqtime
    end_time = seqtime + seq_length

    bag = rosbag.Bag(root + bagfiles[0])

    folder = project + "/" + str(seqtime)

    gpsfile = open(folder + "/gps.csv", "w")
    imufile = open(folder + "/imu.csv", "w")

    last_cam = 0

    for topic, msg, t in bag.read_messages(topics=['/navsat/odom_zero', '/imu/data',
        '/blackfly/image_color/compressed', '/talker1/Navtech/Polar/compressed', '/velodyne_points']):
        if t.to_sec() < start_time:
            continue
        if t.to_sec() > end_time:
            break

        timestamp = t.to_nsec()

        if topic == '/blackfly/image_color/compressed':
            if t.to_sec() - last_cam < float(1) / float(cam_freq):
                continue
            last_cam = t.to_sec()
            np_arr = np.fromstring(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            cv2.imwrite(folder + "/camera/" + str(timestamp) + ".png", img)

        if topic == '/talker1/Navtech/Polar/compressed':
            np_arr = np.fromstring(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(folder + "/radar/" + str(timestamp) + ".png", img)

        if topic == '/navsat/odom_zero':
            odom = msg
            pos = odom.pose.pose.position
            orient = odom.pose.pose.orientation
            v = odom.twist.twist.linear
            w = odom.twist.twist.angular
            gpsfile.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(timestamp,
                pos.x, pos.y, pos.z, orient.x, orient.y, orient.z, orient.w,
                v.x, v.y, v.z, w.x, w.y, w.z))

        if topic == '/imu/data':
            imu = msg
            imufile.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(timestamp, imu.orientation.x,
                imu.orientation.y, imu.orientation.z, imu.orientation.w,
                imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z,
                imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z))

        if topic == '/velodyne_points':
            cloud_points = list(point_cloud2.read_points(msg, skip_nans=True))
            points = np.array(cloud_points, dtype=np.float32)
            s = msg.header.stamp.secs + 6.36
            ns = msg.header.stamp.nsecs
            timestamp = int(s * 1e9 + ns)
            np.savetxt(folder + "/lidar/" + str(timestamp) + '.txt', points, delimiter=',')

    gpsfile.close()
    imufile.close()
