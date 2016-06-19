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

def test_chunk_eq():
    """Chunk Equality Test"""
    filename = "/home/pi/mc/juco/region/r.0.0.mca"
    rf = mche.RegionFile(filename)
    rf_bis = mche.RegionFile(filename)
    rf_bis.read()
    rf.read()
    c1 = rf.chunks[0]
    c2 = rf.chunks[1]
    c1_bis = rf_bis.chunks[0]
    errors = 0

    TP = "__eq__ on different chunks"
    if not c1 == c2:
        print "TP OK : %s" % TP
    else:
        print "TP KO : %s" % TP
        print c1
        print c2
        errors = errors + 1

    TP = "__eq__ on identical chunks"
    if c1 == c1_bis:
        print "TP OK: %s" % TP
    else:
        print "TP KO : %s" % TP
        print c1
        print c1_bis
        errors = errors + 1

    return errors == 0

def test_region_eq():
    """Region Equality Test"""
    filename = "/home/pi/mc/juco/region/r.0.0.mca"
    filename2 = "/home/pi/mc/juco/region/r.0.1.mca"
    rf

if __name__ == "__main__":
    logging.basicConfig(filename="test_mche.log", filemode='w', level=logging.ERROR)
    failed = 0

    if test_chunk_eq():
        print "PASS : %s" % test_chunk_eq.__doc__
    else:
        print "FAIL : %s" % test_chunk_eq.__doc__

    #if test_read_write():
    #    print "Test PASS : Read Write"
    #else:
    #    print "Test FAIL : Read Write"

