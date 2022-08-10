#!/bin/bash

# Download the zipped Projects directory from shared onedrive link
wget --no-check-certificate "https://onedrive.live.com/download?cid=C773214E388750AA&resid=C773214E388750AA%2163931&authkey=AIMRa2r9nqDti_A" -O Projects.zip

# unzip the Projects directory
unzip Projects.zip

# Remove downloaded file
rm Projects.zip
