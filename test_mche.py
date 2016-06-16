#!/usr/bin/env python
import logging
import mche
import os
import re

if __name__ == "__main__":
    logging.basicConfig(filename="test_mche.log", level=logging.DEBUG)
    path = "/home/pi/mc/juco/region"
    for e in os.listdir(path):
        f = os.path.join(path, e)
        if re.match("r.-?\d+\.-?\d+\.mca$", e):
            rf = mche.RegionFile(f)
            rf.read()
    

