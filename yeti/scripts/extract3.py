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

root = '/media/keenan/autorontossd1/2020_10_06/'
project = '/home/keenan/Documents/data/boreas/2020_10_06'

lidar_times = np.zeros(1, dtype=np.int64)

already_downloaded = os.listdir(project + "/lidar/")

def get_closest(times, ts, min_delta_t=0.15):
    closest = -1
    for i in range(0, len(times)):
        time = times[i]
        if abs(time - ts) < min_delta_t:
            min_delta_t = abs(time - ts)
            closest = i
    return closest

def callback(msg):
    timestamp = msg.header.stamp.to_nsec()
    closest = get_closest(lidar_times, timestamp, 1.5e8)
    if closest < 0:
        return
    if '{}.txt'.format(timestamp) in already_downloaded:
        return
    print(timestamp)
    cloud_points = list(point_cloud2.read_points(msg, skip_nans=True))
    points = np.array(cloud_points, dtype=np.float32)
    np.savetxt(project + "/lidar/" + str(timestamp) + '.txt', points, delimiter=',')

if __name__ == '__main__':
    files = os.listdir(root)
    bagfiles = []
    for file in files:
        if file.split('.')[-1] == 'bag':
            bagfiles.append(file)
    bagfiles.sort()

    gtfile = project + '/radar_groundtruth.csv'
    g = open(gtfile, 'r')
    glines = g.readlines()
    radar_times = []
    for line in glines:
        line = line.split(',')
        t = int(line[0])
        radar_times.append(t / 1.0e9)
        t = int(line[1])
        radar_times.append(t / 1.0e9)

    # lidar_times = np.zeros(len(radar_times), dtype=np.int64)
    #
    # for bagfile in bagfiles:
    #     bag = rosbag.Bag(root + bagfile)
    #     print(bagfile)
    #     for topic, msg, t in bag.read_messages(topics=['/velodyne_packets']):
    #         timestamp = t.to_nsec()
    #         ts = timestamp / 1.0e9
    #         timestamp /= 1000
    #         timestamp *= 1000
    #         closest = get_closest(radar_times, ts)
    #         if closest < 0:
    #             continue
    #         lidar_times[closest] = timestamp
    #
    # for i in range(0, len(lidar_times)):
    #     if lidar_times[i] == 0:
    #         rtime = radar_times[i]
    #         closest = get_closest(lidar_times, rtime * 1e9, 1.5e8)
    #         if closest > 0:
    #             lidar_times[i] = lidar_times[closest]
    #
    # lfile = project + '/lidar_files.csv'
    # l = open(lfile, 'w')
    # for i in range(0, len(lidar_times) / 2):
    #     l.write('{},{},\n'.format(lidar_times[2 * i], lidar_times[2 * i + 1]))

    lfile = project + '/lidar_files.csv'
    l = open(lfile, 'r')
    lidar_times = []
    for line in l:
        line = line.split(',')
        lidar_times.append(int(line[0]))
        lidar_times.append(int(line[1]))

    rospy.init_node('lidar_listener')
    rospy.Subscriber("/velodyne_points", PointCloud2, callback, queue_size = 1000)
    rospy.spin()
