#!/usr/bin/env python
# coding: utf-8


import subprocess
import os
import time

def format_time(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes}\'{seconds}\""

video_folder = './video'

for filename in os.listdir(video_folder):
    if filename.endswith('.mov'):  
        filepath = os.path.join(video_folder, filename)
        

        start_time = time.time()

        
        try:
            subprocess.run(["python", "count_flies.py", "--video_path", filepath], check=True, stderr=subprocess.PIPE, text=True)
        except subprocess.CalledProcessError as e:
            print(f"\n JOB {filename} ERROR \n{e.stderr}")
            continue

        
        elapsed_time = time.time() - start_time
        formatted_time = format_time(elapsed_time)
        print(f"Job \[{filename}\]complited. Time spent {formatted_time}")
