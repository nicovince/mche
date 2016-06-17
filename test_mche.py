#!/usr/bin/env python
import logging
import filecmp
import mche
import os
import re

def test_read_write():
    path = "/home/pi/mc/juco/region"
    diff = 0
    for e in os.listdir(path):
        f = os.path.join(path, e)
        if re.match("r.-?\d+\.-?\d+\.mca$", e):
            rf = mche.RegionFile(f)
            rf.read()
            mche_file = f + ".mche"
            rf.write(mche_file)
            if rf.has_gap:
                # check size
                if os.path.getsize(f) != os.path.getsize(mche_file):
                    print "%s and %s size differs" % (f, mche_file)
                    diff = diff + 1
            else:
                if not filecmp.cmp(f, mche_file) and not rf.has_gap:
                    print "%s and %s content differs" % (os.path.basename(f),
                                                         os.path.basename(mche_file))
                    diff = diff + 1

    return diff == 0

if __name__ == "__main__":
    logging.basicConfig(filename="test_mche.log", filemode='w', level=logging.ERROR)
    failed = 0
    if test_read_write():
        print "Test PASS : Read Write"
    else:
        print "Test FAIL : Read Write"

    

