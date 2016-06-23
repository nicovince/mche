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
    """Read and Write Region files"""
    path = "/home/pi/mc/juco/region"
    diff = 0
    for e in os.listdir(path):
        f = os.path.join(path, e)
        if re.match("r.-?\d+\.-?\d+\.mca$", e):
            rf = mche.RegionFile(f)
            rf.read()
            mche_file = f + ".mche"
            rf.write(mche_file)
            rf_written = mche.RegionFile(mche_file)
            rf_written.read()
            if rf != rf_written:
                diff += 1
            os.remove(mche_file)

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


def test_delete_chunk():
    """Chunk Deletion Test"""
    # TODO: automate testing
    rf_name = "/home/pi/mc/juco/region/r.3.3.mca"
    mche_ext = ".mche"
    rf_mche_name = rf_name + mche_ext
    rf = mche.RegionFile(rf_name)
    rf.read()
    # 3, 11 is the last chunk stored in the file
    # remove it
    rf.delete_chunk(3, 11)
    rf.write(rf_mche_name)
    # Read Region file with chunk removed
    rf_mche = mche.RegionFile(rf_mche_name)
    rf_mche.read()

    # Re-read original region file
    rf = mche.RegionFile(rf_name)
    rf.read()

    # Compare
    chunks_preserved = True
    chunk_deleted = True
    errors = 0
    for z in range(32):
        for x in range(32):
            i = 32*z + x
            c1 = rf.chunks[i]
            c2 = rf_mche.chunks[i]
            # Check preserved chunks are identicals, and removed chunk is not
            # present anymore
            if x != 3 or z != 11:
                if c1 != c2:
                    chunks_preserved = False
                    errors += 1
                    log_tp(chunks_preserved,
                           "Chunk (%d, %d) shall be preserved" % (x, z))
            else:
                if not c2.is_generated():
                    chunk_deleted = False
                    errors += 1
                log_tp(chunk_deleted, "Chunk (%d, %d) shall be cleared"
                       % (x, z))
    log_tp(chunks_preserved, "Chunks others than (3, 11) are untouched")
    return (errors == 0)

    # first_chunk = [c for c in sorted(rf.chunks, key=lambda x: x.offset)
    #                if c.offset > 0][0]
    # rf.display_chunk_info(first_chunk.x, first_chunk.z)
    # rf.delete_chunk(first_chunk.x, first_chunk.z)


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


def test_coords_from_str():
    """String Coordinates Parser"""
    errors = 0
    coords = mche.get_coords_from_str("5x3")
    print coords
    if not log_tp(coords[0][0] == 5 and coords[0][1] == 3, "Parse 5x3"):
        errors += 1

    test_coords = [[5,10], [12,13], [-5,-2], [-7,0], [0,-9]]
    test_string = ""
    for (x,z) in test_coords:
        test_string += "%dx%d," % (x, z)
    # Remove last ','
    test_string = test_string[0:-1]
    coords = mche.get_coords_from_str(test_string)
    if not log_tp(coords == test_coords, "Parse %s" % test_string):
        errors += 1

    e = None
    test_string = "zxy"
    try:
        coords = mche.get_coords_from_str(test_string)
    except ValueError:
        e = ValueError
    except:
        print "got unexpected exception"
    if not log_tp(e == ValueError, "Raise ValueError on %s" % test_string):
        errors += 1

    e = None
    test_string="1x-s12"
    try:
        coords = mche.get_coords_from_str(test_string)
    except ValueError:
        e = ValueError
    except:
        print "got unexpected exception"
    if not log_tp(e == ValueError, "Raise ValueError on %s" % test_string):
        errors += 1

    return errors == 0


def test_zone_from_str():
    """String Zone Parser"""
    errors = 0
    zones = "12x34_56x72"
    zones_exp = [[[12, 34], [56, 72]]]
    if not log_tp(mche.get_zone_from_str(zones) == zones_exp,
                  "Parse Zone %s" % zones):
        errors += 1
    return errors == 0

if __name__ == "__main__":
    logging.basicConfig(filename="test_mche.log", filemode='w',
                        level=logging.ERROR)

    log_test(test_chunk_eq)
    log_test(test_region_eq)
    log_test(test_delete_chunk)
    log_test(test_coords_from_str)
    log_test(test_zone_from_str)
    # log_test(test_read_write) # Long test
