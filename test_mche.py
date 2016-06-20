#!/usr/bin/env python
import logging
import filecmp
import mche
import os
import re
import sys

def log_tp(test, name):
    """
    Log Test point

    test : true when test is successful
    name : Name of the test to log
    """
    if test:
        print "TP OK : %s" % name
    else:
        print "TP KO : %s" % name
    return test

def log_test(test_func):
    print "------------------"
    print test_func.__doc__
    if test_func():
        print "PASS : %s" % test_func.__doc__
    else:
        print "FAIL : %s" % test_func.__doc__

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

    if not log_tp(c1 != c2, "__eq__ on different chunks"):
        print c1
        print c2
        errors = errors + 1

    if not log_tp(c1 == c1_bis, "__eq__ on identical chunks"):
        print c1
        print c1_bis
        errors = errors + 1
    return errors == 0

def test_region_eq():
    """Region Equality Test"""
    errors = 0
    filename = "/home/pi/mc/juco/region/r.0.0.mca"
    filename2 = "/home/pi/mc/juco/region/r.0.1.mca"
    rf = mche.RegionFile(filename)
    rf.read()
    rf2 = mche.RegionFile(filename2)
    rf2.read()
    rf_bis = mche.RegionFile(filename)
    rf_bis.read()

    if not log_tp(rf != rf2, "__eq__ on different region files"):
        errors += 1
    if not log_tp(rf == rf_bis, "__eq__ on identical region files"):
        print "rf.x : %d" % rf.x
        print "rf.z : %d" % rf.z
        print "rf_bis.x : %d" % rf_bis.x
        print "rf_bis.z : %d" % rf_bis.z
        for i in range(1024):
            if rf.chunks[i] != rf_bis.chunks[i]:
                print "%d differents :" % i
                print "  - %s" % rf.chunks[i]
                print repr(rf.chunks[i])
                print "  - %s" % rf_bis.chunks[i]
                print repr(rf_bis.chunks[i])
        errors += 1

    # Remove chunk and check equality test fails
    rf_bis.delete_chunk(0, 0)
    log_tp(rf != rf_bis, "__eq__ on identical region file modulo one chunk")
    return errors == 0

if __name__ == "__main__":
    logging.basicConfig(filename="test_mche.log", filemode='w', level=logging.ERROR)
    failed = 0

    log_test(test_chunk_eq)
    log_test(test_region_eq)

    #if test_read_write():
    #    print "Test PASS : Read Write"
    #else:
    #    print "Test FAIL : Read Write"

