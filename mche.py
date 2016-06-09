#!/usr/bin/env python
import sys
import struct
from binascii import hexlify

class RegionFile:
    """
    Class to handle Minecraft Region file

    Load them, prune a chunk, save to file
    """
    def __init__(self, filename):
        self.region_filename = filename

    def read(self):
        """
        Load Region file into memory to perform actions on region

        Usually done once during init
        """
        with open(self.region_filename, "rb") as f:
            # Read location (4096 bytes)
            self.location_field = f.read(4096)
            # byte offset of chunk coordinates (x,z) : 4((x%32) + (z%32)*32)
            self.location = list()
            for z in range(32):
                for x in range(32):
                    # Offset of current field
                    offset = 4*(x + 32*z)
                    # location field for current chunk coordinates
                    cur_loc = self.location_field[offset:offset+4]
                    # Offset of chunk datas in file
                    data_offset = int(hexlify(cur_loc[0:3]), 16)
                    # Number of sectors (4096 bytes) occupied by chunk
                    data_count = int(hexlify(cur_loc[3]), 16)
                    self.location.append({"offset" : data_offset,
                                          "sector_count" : data_count})

            print self.location_field[0:3].encode('hex')
            print self.location[0]
            print self.location[0]["offset"]
            print self.location[0]["sector_count"]

            #TODO
            # Read timestamps (4096 bytes)
            # Read chunk datas (number of data deduced from location fields


if __name__ == "__main__":
    rf = RegionFile("/home/pi/mc/juco/region/r.3.3.mca")
    rf.read()
