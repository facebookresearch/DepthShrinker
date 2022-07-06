# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import shutil
import sys

os.system('cd ~/fbsource/fbcode')

os.system('mkdir ondevice_ai_tools')
os.system('manifoldfs manifold.blobstore ondevice_ai_tools ./ondevice_ai_tools')

root_fyg = 'ondevice_ai_tools/workflows/yongganfu/'
date_list=os.listdir(root_fyg)

log_list = []

for date in date_list:
    date_path = os.path.join(root_fyg, date)

    if os.path.isdir(date_path):
        for work_id in os.listdir(date_path):
            work_path = os.path.join(date_path, work_id)

            for root, dirs, files in os.walk(work_path):
                for fname in files:
                    if fname == 'stdout.log':
                        log_path = os.path.join(root, fname)
                        log_list.append(log_path)


os.system('mkdir depthshrinker_logs')
for log in log_list:
    new_name = log.split('/')[-4]
    shutil.copy(log, 'depthshrinker_logs/' + new_name + '.log')



