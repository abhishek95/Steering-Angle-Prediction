# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 20:42:57 2018

@author: abhis
"""

import rosbag
bag = rosbag.Bag('HMB_4.bag')
for topic, msg, t in bag.read_messages(topics=['chatter', 'numbers']):
    print msg
bag.close()