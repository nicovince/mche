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
    assert data < 2**(8*n),\
        "%d bytes is too small to contain %s" % (n, hex(data))
    # Create format for hexadecimal data
    fmt = "%%0%dx" % (2*n)
    ret = unhexlify(fmt % data)
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
            # go through chunk in the order they are stored in chunk data field
            for c in sorted(self.chunks, key=lambda x: x["offset"]):
                x = c["x"]
                z = c["z"]
                # chunk index in chunks list
                chunk_idx = x + 32*z
                size = 4096 * c["sector_count"]
                if size > 0:
                    total_size += size
                    c["chunk_data"] = f.read(size)

            # Check filesize is equal to size read
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
            # Write timestamps fields
            for z in range(32):
                for x in range(32):
                    chunk_idx = x + 32*z
                    chunk = self.chunks[chunk_idx]
                    timestamp_field = get_byte_seq(chunk["timestamp"], 4)
                    f.write(timestamp_field)

            # write chunk's datas
            for c in sorted(self.chunks, key=lambda x: x["offset"]):
                # skip chunks with offset 0
                x = c["x"]
                z = c["z"]
                if c["offset"] != 0:
                    f.write(c["chunk_data"])

    def display_chunk_info(self, x, z):
        """
        Display chunks info of chunk coordinates (x,z)

        Coordinates of chunks are relative coords from the region file
        """
        # Index of chunk in chunks list
        chunk_idx = (x % 32) + 32*(z % 32)
        print "offset : %d" % self.chunks[chunk_idx]["offset"]
        print "sector count : %d" % self.chunks[chunk_idx]["sector_count"]
        print "x : %d" % self.chunks[chunk_idx]["x"]
        print "z : %d" % self.chunks[chunk_idx]["z"]
        print "timestamp : %d" % self.chunks[chunk_idx]["timestamp"]
        print "Chunk coords : " + str(self.get_chunk_coords_blk(x, z))

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

    def delete_chunk(self, x, z):
        """
        Delete chunk at relative coords x, z

        Done by setting location and timestamp fields to zero.
        Locations of chunks stored after deleted chunk are updated
        """
        chunk_idx = x + 32*z
        deleted_chunk = self.chunks[chunk_idx]
        # List of chunks that needs to be updated
        update_chunks = [c for c in self.chunks
                         if c["offset"] > deleted_chunk["offset"]]
        # recompute offset of chunks stored after delted chunk
        for c in update_chunks:
            c["offset"] = c["offset"] - deleted_chunk["sector_count"]
        # reset offset, sector count and timestamp of deleted chunk
        deleted_chunk["offset"] = 0
        deleted_chunk["sector_count"] = 0
        deleted_chunk["timestamp"] = 0

    def get_chunk_coords(self, x_blk, z, blk):
        """
        Get relative chunk coords from absolute blocks coordinates
        """
        # TODO


if __name__ == "__main__":
    rf = RegionFile("/home/pi/mc/juco/region/r.3.3.mca")
    rf.read()
    rf.display_chunk_info(3, 11)
    # rf.delete_chunk(3, 11)
    first_chunk = [c for c in sorted(rf.chunks, key=lambda x: x["offset"])
                   if c["offset"] > 0][0]
    rf.display_chunk_info(first_chunk["x"], first_chunk["z"])
    rf.delete_chunk(first_chunk["x"], first_chunk["z"])

    rf.write("/home/pi/mc/juco/region/r.3.3.mca.mche")
