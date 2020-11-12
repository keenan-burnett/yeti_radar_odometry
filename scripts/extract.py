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

if __name__ == '__main__':
    project = '/home/keenan/Documents/data/boreas/2020_09_01'
    root = project + "/"
    files = os.listdir(root)
    bagfiles = []
    for file in files:
        if file.split('.')[-1] == 'bag':
            bagfiles.append(file)
    bagfiles.sort()

    seqtimes = [1598986495]

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

    seqtime = 1598986495

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
