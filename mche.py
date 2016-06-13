#!/usr/bin/env python
import os
import sys
import re
from binascii import hexlify
from binascii import unhexlify


def get_byte_seq(data, n):
    """
    Return data as a byte string of length n
    """
    fmt = "%%0%dx" % (2*n)
    ret = unhexlify(fmt % data)
    assert len(ret) <= n, "%d bytes is too small to contain %s" % (n, hex(data))
    # perform left padding with null bytes
    #ret = '\x00' * (n - len(ret)) + ret

    return ret

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
                    self.chunks.append({"offset": data_offset,
                                        "sector_count": data_count,
                                        "x": x,
                                        "z": z})

            # Read timestamps (4096 bytes)
            self.timestamps_field = f.read(4096)
            for z in range(32):
                for x in range(32):
                    # Offset of current chunk's timestamp in timestamp field
                    offset = 4*(x + 32*z)
                    cur_ts = self.timestamps_field[offset:offset+4]
                    cur_ts = int(hexlify(cur_ts), 16)
                    self.chunks[offset/4]["timestamp"] = cur_ts

            # Read chunk datas (number of data deduced from location fields
            total_size = 8192
            for z in range(32):
                for x in range(32):
                    # chunk index in chunks list
                    chunk_idx = x + 32*z
                    size = 4096 * self.chunks[chunk_idx]["sector_count"]
                    if size > 0:
                        total_size += size
                        self.chunks[chunk_idx]["chunk_data"] = f.read(size)

            file_size = os.path.getsize(self.region_filename)
            assert total_size == file_size,\
                "Size read in file (%d) does not match file size (%d)"\
                % (total_size, file_size)

    def write(self, filename):
        """
        Write region file to given filename

        Filename can be the same as self.region_filename if you are sure of
        your changes or have made a backup earlier
        """
        with open(filename, "wb") as f:
            # Write locations fields
            for z in range(32):
                for x in range(32):
                    # relative chunk index in region
                    chunk_idx = x + 32*z
                    chunk = self.chunks[chunk_idx]
                    offset_field = get_byte_seq(chunk["offset"], 3)
                    sector_count_field = get_byte_seq(chunk["sector_count"], 1)
                    location_field = offset_field + sector_count_field
                    f.write(location_field)
            # TODO timestamps
            # TODO data chunks

    def display_chunk_info(self, x, z):
        """
        Display chunks info of chunk coordinates (x,z)

        Coordinates of chunks are relative coords from the region file
        """
        # Index of chunk in chunks list
        chunk_idx = (x % 32) + 32*(z % 32)
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

        x : relative chunk x-coordinate in region file [0-31]
        z : relative chunk z-coordinate in region file [0-31]

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
        return ((x1, z1), (x2, z2))

if __name__ == "__main__":
    rf = RegionFile("/home/pi/mc/juco/region/r.3.3.mca")
    rf.read()
    rf.display_chunk_info(0, 0)
    rf.display_chunk_info(31, 31)
    rf.write("/home/pi/mc/juco/region/r.3.3.mca.mche")

    # for c in sorted(rf.chunks, key=lambda x: x["offset"]):
    #     print c
