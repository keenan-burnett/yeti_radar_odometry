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

if __name__ == '__main__':
    root = '/media/keenan/autorontossd1/2020_10_06/'
    project = '/home/keenan/Documents/data/boreas/2020_10_06'
    files = os.listdir(root)
    bagfiles = []
    for file in files:
        if file.split('.')[-1] == 'bag':
            bagfiles.append(file)
    bagfiles.sort()

    folder = project

    gpsfile = open(folder + "/gps.csv", "w")
    imufile = open(folder + "/imu.csv", "w")
    gpstimefile = open(folder + "/ros_and_gps_time.csv", "w")

    for bagfile in bagfiles:
        bag = rosbag.Bag(root + bagfile)
        print(bagfile)
        for topic, msg, t in bag.read_messages(topics=['/navsat/group/1', '/navsat/odom', '/imu/data']): # ,'/talker1/Navtech/Polar/compressed']):

            timestamp = t.to_nsec()

            if topic == '/navsat/group/1':
                gpstimefile.write('{},{},\n'.format(timestamp, msg.time_distance.time1))

            # if topic == '/talker1/Navtech/Polar/compressed':
            #     np_arr = np.fromstring(msg.data, np.uint8)
            #     img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
            #     cv2.imwrite(folder + "/radar/" + str(timestamp) + ".png", img)
            
            if topic == '/navsat/odom':
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



    gpsfile.close()
    imufile.close()
