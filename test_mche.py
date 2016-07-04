#!/usr/bin/env python
import logging
import filecmp
import mche
import os
import re
import sys
import timeit
# Test mche module
# TODO: remove created files

errors_cnt = 0


def log_tp(test, name):
    """
    Log Test point

    test : true when test is successful
    name : Name of the test to log
    """
    global errors_cnt
    if test:
        print "TP OK : %s" % name
    else:
        print "TP KO : %s" % name
        errors_cnt += 1
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
            mche_file = f + ".mche"
            rf.write(mche_file)
            rf_written = mche.RegionFile(mche_file)
            if rf != rf_written:
                diff += 1
            os.remove(mche_file)

    return diff == 0


def test_chunk_eq():
    """Chunk Equality Test"""
    filename = "/home/pi/mc/juco/region/r.0.0.mca"
    rf = mche.RegionFile(filename)
    rf_bis = mche.RegionFile(filename)
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
    errors = 0
    rf_name = "/home/pi/mc/juco/region/r.3.3.mca"
    mche_ext = ".mche"
    rf_mche_name = rf_name + mche_ext
    rf = mche.RegionFile(rf_name)

    # Check exception raised if chunk not present in region
    e = None
    try:
        rf.delete_chunk(0, 0)
    except AssertionError:
        e = AssertionError
    except:
        print "got wrong exception"
    if not log_tp(e == AssertionError,
                  "Raise Assertion if delete chunk not present in region"):
        errors += 1

    # 3, 11 is the last chunk stored in the file
    # remove it
    rf.delete_chunk(*rf.get_absolute_chunk_coords(3, 11))
    rf.write(rf_mche_name)
    # Read Region file with chunk removed
    rf_mche = mche.RegionFile(rf_mche_name)

    # Re-read original region file
    rf = mche.RegionFile(rf_name)

    # Compare
    chunks_preserved = True
    chunk_deleted = True
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
                if c2.is_generated():
                    chunk_deleted = False
                    errors += 1
                log_tp(chunk_deleted, "Chunk (%d, %d) shall be cleared"
                       % (x, z))
    log_tp(chunks_preserved, "Chunks others than (3, 11) are untouched")
    os.remove(rf_mche_name)
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
    rf2 = mche.RegionFile(filename2)
    rf_bis = mche.RegionFile(filename)

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

    test_coords = [(5, 10), (12, 13), (-5, -2), (-7, 0), (0, -9)]
    test_string = ""
    for (x, z) in test_coords:
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
    test_string = "1x-s12"
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
    zones_exp = [[(12, 34), (56, 72)]]
    zones_test = mche.get_zones_from_str(zones)
    if not log_tp(zones_test == zones_exp,
                  "Parse Zone %s" % zones):
        print zones_test
        print zones_exp
        errors += 1

    zones_exp = [[(-1, 0), (3, -5)], [(-20, -32), (45, -12)]]
    zones = mche.get_str_from_zones(zones_exp)
    if not log_tp(mche.get_zones_from_str(zones) == zones_exp,
                  "Parse zones %s" % zones):
        errors += 1

    return errors == 0


def test_coords_by_region():
    """Test Coordinates sorted by region"""
    errors = 0
    world = mche.World("/home/pi/mc/juco/")
    coords = world.get_coords_by_region([[0, 0], [12, 511], [511, 511],
                                         [-1, -1], [-512, -512],
                                         [0, -1], [511, -512],
                                         [-1, 0], [-512, 511]])
    for region, blk_coords in coords.items():
        (r_x, r_z) = re.findall("-?\d+", region)
        (r_x, r_z) = (int(r_x), int(r_z))
        for (x, z) in blk_coords:
            blk_ok = ((x < (r_x+1)*512) and (x >= r_x*512) and
                      (z < (r_z+1)*512) and (z >= r_z*512))
            if not log_tp(blk_ok, "Block (%d, %d) belongs to %s"
                          % (x, z, region)):
                errors += 1

    return errors == 0


def test_gaps():
    """Test gap counter"""
    errors = 0
    filename = "/home/pi/mc/juco/region/r.0.0.mca"
    rf = mche.RegionFile(filename)
    if not log_tp(rf.count_gaps() > 0, "Gaps present in file %s" % filename):
        errors += 1

    filename = "/home/pi/mc/juco/region/r.3.3.mca"
    rf = mche.RegionFile(filename)
    if not log_tp(rf.count_gaps() == 0,
                  "No Gaps present in file %s" % filename):
        errors += 1

    return errors == 0


def test_rm_gaps():
    """Test Removing Gaps from Region File"""
    errors = 0
    filename = "/home/pi/mc/juco/region/r.0.0.mca"
    rf = mche.RegionFile(filename)
    rf.remove_gaps()
    nogaps_file = filename + ".nogaps"
    rf.write(nogaps_file)
    rf_orig = mche.RegionFile(filename)
    rf_nogaps = mche.RegionFile(nogaps_file)
    if not log_tp(rf_orig == rf_nogaps,
                  "Region file without gaps equivalent to original"):
        rf_orig.diff(rf_nogaps)

        errors += 1
    os.remove(nogaps_file)
    return errors == 0


def test_rm_dim_gaps():
    """Test Removing Gaps from Dimension Region files"""
    errors = 0
    path = "/home/pi/mc/juco"
    ow_path = os.path.join(path, "region")
    world = mche.World(path)
    world.remove_gaps("overworld", ".nogaps")
    nogaps_files = [os.path.join(ow_path, f) for f in os.listdir(ow_path)
                    if re.match("r.-?\d+\.-?\d+\.mca.nogaps$", f)]
    mismatch = []
    for f in nogaps_files:
        orig_f = re.sub(".nogaps", "", f)
        rf_nogaps = mche.RegionFile(f)
        rf_orig = mche.RegionFile(orig_f)
        if rf_orig != rf_nogaps:
            mismatch.append((rf_orig, rf_nogaps))
    if not log_tp(len(mismatch) == 0,
                  "Region Files without gaps matches originals"):
        print "Mismatches on " + str(mismatch)
        errors += 1
    else:
        for f in nogaps_files:
            os.remove(f)
    return errors == 0


def test_chunk_in_region():
    """Test Chunk in region"""
    errors = 0
    region = "r.3.3.mca"
    rf = mche.RegionFile("/home/pi/mc/juco/region/%s" % region)
    coords = (0, 0)
    if not log_tp(not rf.is_chunk_in_region(*coords),
                  "Chunk %s not in region %s" % (coords, region)):
        errors += 1

    coords = (4*32, 3*32)
    if not log_tp(not rf.is_chunk_in_region(*coords),
                  "Chunk %s not in region %s" % (coords, region)):
        errors += 1

    coords = (3*32, 3*32)
    if not log_tp(rf.is_chunk_in_region(*coords),
                  "Chunk %s in region %s" % (coords, region)):
        errors += 1

    coords = (4*32-1, 4*32-1)
    if not log_tp(rf.is_chunk_in_region(*coords),
                  "Chunk %s in region %s" % (coords, region)):
        errors += 1

    region = "r.-1.-1.mca"
    rf = mche.RegionFile("/home/pi/mc/juco/region/%s" % region)
    coords = (-1, -1)
    if not log_tp(rf.is_chunk_in_region(*coords),
                  "Chunk %s in region %s" % (coords, region)):
        errors += 1

    coords = (0, 0)
    if not log_tp(not rf.is_chunk_in_region(*coords),
                  "Chunk %s not in region %s" % (coords, region)):
        errors += 1

    return errors == 0


def test_create_rf_heat_map():
    filename = "/home/pi/mc/juco/region/r.0.0.mca"
    dirname = "heatmap_region"
    dirname = "./"
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    rf = mche.RegionFile(filename, read=False)
    rf.create_gp_ts_map(dirname)


def test_create_world_heat_map():
    dirname = "heatmap_world"
    #dirname = "./"
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    world = mche.World("/home/pi/mc/juco/")
    world.create_gp_ts_map(dirname, "overworld")

if __name__ == "__main__":
    logging.basicConfig(filename="test_mche.log", filemode='w',
                        level=logging.DEBUG)

    #test_create_rf_heat_map()
    #test_create_world_heat_map()
    log_test(test_gaps)
    log_test(test_chunk_in_region)
    log_test(test_chunk_eq)
    log_test(test_region_eq)
    log_test(test_delete_chunk)
    log_test(test_coords_from_str)
    log_test(test_zone_from_str)
    log_test(test_coords_by_region)
    log_test(test_rm_gaps)
    # log_test(test_rm_dim_gaps)
    # log_test(test_read_write)  # Long test

    if errors_cnt != 0:
        sys.exit(1)
