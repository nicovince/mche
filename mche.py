#!/usr/bin/env python
import os
import sys
import re
from binascii import hexlify

class RegionFile:
    """
    Class to handle Minecraft Region file

    Load them, prune a chunk, save to file
    """
    def __init__(self, filename):
        self.region_filename = filename
        self.x, self.z = self.get_region_coords()

    def read(self):
        """
        Load Region file into memory to perform actions on region

        Usually done once during init
        """
        with open(self.region_filename, "rb") as f:
            # Read location (4096 bytes)
            self.location_field = f.read(4096)
            self.chunks = list()
            # byte offset of chunk coordinates (x,z) : 4((x%32) + (z%32)*32)
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
                    self.chunks.append({"offset" : data_offset,
                                        "sector_count" : data_count,
                                        "x" : x,
                                        "z" : z})

            # Read timestamps (4096 bytes)
            self.timestamps_field = f.read(4096)
            for z in range(32):
                for x in range(32):
                    # Offset of current chunk's timestamp in timestamp field
                    offset = 4*(x + 32*z)
                    cur_ts = self.timestamps_field[offset:offset+4]
                    self.chunks[offset/4]["timestamp"] = int(hexlify(cur_ts), 16)
            #TODO
            # Read chunk datas (number of data deduced from location fields

    def display_chunk_info(self, x, z):
        """
        Display chunks info of chunk coordinates (x,z)

        Coordinates of chunks are relative coords from the region file
        """
        # Index of chunk in chunks list
        chunk_idx = (x%32) + 32*(z%32)
        print self.chunks[chunk_idx]
        print self.chunks[chunk_idx]["offset"]
        print self.chunks[chunk_idx]["sector_count"]
        print self.chunks[chunk_idx]["x"]
        print self.chunks[chunk_idx]["z"]
        print self.chunks[chunk_idx]["timestamp"]
        print self.get_chunk_coords_blk(x, z)

    def get_region_coords(self):
        """
        Get region coordinates from filename

        In region coordinates, not in blocks coordinates
        """
        filename = os.path.basename(self.region_filename)
        (x, z) = re.findall("-?\d+", filename)
        return (int(x), int(z))

    def get_chunk_coords_blk(self, x, z):
        """
        Get chunk coordinates in blocks

        Coordinates of chunk are given in relative coords from the region file ([0-31], [0-31])

        Return tuple of two points giving two opposite corners of the chunk
        ((x1, z1), (x2, z2))
        """
        x_region_offset = self.x * 512
        z_region_offset = self.z * 512
        x_chunk_offset = x * 16
        z_chunk_offset = z * 16
        x1 = x_region_offset + x_chunk_offset
        x2 = x_region_offset + x_chunk_offset + 15
        z1 = z_region_offset + z_chunk_offset
        z2 = z_region_offset + z_chunk_offset + 15
        return ((x1, z1), (x2,z2))




if __name__ == "__main__":
    rf = RegionFile("/home/pi/mc/juco/region/r.3.3.mca")
    rf.read()
    rf.display_chunk_info(0,0)
    rf.display_chunk_info(31,31)

    rf = RegionFile("/home/pi/mc/juco/region/r.0.0.mca")
    rf.read()
    rf.display_chunk_info(0,0)
    rf.display_chunk_info(31,31)

    rf = RegionFile("/home/pi/mc/juco/region/r.-1.-1.mca")
    rf.read()
    rf.display_chunk_info(0,0)
    rf.display_chunk_info(31,31)
