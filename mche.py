#!/usr/bin/env python
import os
import sys
import re
import logging
import itertools
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
        """
        Initialize instance with region filename attribute
        """
        self.region_filename = filename
        if not os.path.exists(filename):
            raise IOError
        self.x, self.z = self.get_region_coords()

    def __str__(self):
        start_x = self.x*512
        start_z = self.z*512
        end_x = start_x + 511
        end_z = start_z + 511
        return os.path.basename(self.region_filename) + " (%d, %d) - (%d, %d)"\
            % (start_x, start_z, end_x, end_z)

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
                    # chunk generated
                    if f.tell() < c["offset"] * 4096:
                        # Current position does not match position of current chunk
                        gap = c["offset"] * 4096 - f.tell()
                        logging.warning("Region %s, %d bytes gap before chunk "
                                        "(%d, %d)'s data at offset %d"
                                        % (self, gap, x, z, c["offset"]))
                        f.seek(c["offset"] * 4096)
                    total_size += size
                    c["chunk_data"] = f.read(size)
                    length = int(hexlify(c["chunk_data"][0:4]), 16)
                    c["length"] = length
                    if length > size:
                        logging.warning("chunk (%d, %d) uses %d sector (%d bytes), chunk data length is %d" % (x, z, c["sector_count"], size, length))
                        assert length > size


            # Check filesize is equal to size read
            file_size = os.path.getsize(self.region_filename)
            # TODO: investigate error when reading region (0, 0)
            # if total_size != file_size:
            #     for c in self.chunks:
            #         self.display_chunk_info(c["x"], c["z"])
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
        if deleted_chunk["offset"] == 0:
            logging.debug("chunk (%d, %d) has not been generated" % (x, z))
            return
        logging.debug("delete relative chunk (%d, %d) in region %s"
                      % (x, z, os.path.basename(self.region_filename)))
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

    def get_relative_chunk_coords(self, chunk_x, chunk_z):
        """
        Get relative coords for chunk's absolutescoordinates
        """
        return (chunk_x % 32, chunk_z % 32)

    def get_chunk_coords(self, block_x, block_z):
        """
        Get relative chunk coords from block's coordinates
        """
        return self.get_relative_chunk_coords(block_x >> 4, block_z >> 4)


class World:
    """
    Class to handle World

    delete zones from dimensions, single chunks, ...
    """
    dimensions = ["overworld", "nether", "theend"]

    def __init__(self, world_pathname):
        """
        Initialize class with path to world which will be edited
        """
        self.world = world_pathname
        if not os.path.exists(self.world):
            raise IOError

    def get_overwold_dir(self):
        """
        Return directory containing overworld region files
        """
        ow_dir = os.path.join(self.world, "region")
        if not os.path.exists(self.world):
            raise IOError
        return ow_dir

    def get_nether_dir(self):
        """
        Return directory containing nether region files
        """
        ow_dir = os.path.join(os.path.join(self.world, "DIM-1"), "region")
        if not os.path.exists(self.world):
            raise IOError
        return ow_dir

    def get_theend_dir(self):
        """
        Return directory containing theend region files
        """
        ow_dir = os.path.join(os.path.join(self.world, "DIM1"), "region")
        if not os.path.exists(self.world):
            raise IOError
        return ow_dir

    def get_dim_dir(self, dim):
        """
        Return directory containing specified dimension files
        """
        assert dim in self.dimensions, "Dimension %s is not valid" % dim
        if dim == "overworld":
            return self.get_overwold_dir()
        elif dim == "nether":
            return self.get_nether_dir()
        elif dim == "theend":
            return self.get_theend_dir()

    def get_region_file(self, dim, block_x, block_z):
        """
        Return region filename for block coordinates of given dimension
        """
        region = self.get_region_coords(block_x, block_z)
        filename = "r.%d.%d.mca" % (region[0], region[1])

        region_file = os.path.join(self.get_dim_dir(dim), filename)
        return region_file

    def get_region_coords(self, block_x, block_z):
        """
        Return tuple of region coordinates containing specified block
        """
        region_x = (block_x >> 4) >> 5
        region_z = (block_z >> 4) >> 5
        return (region_x, region_z)

    def get_chunk_coords(self, block_x, block_z):
        """
        Return tuple of chunk coordinates containing specified block
        """
        chunk_x = block_x >> 4
        chunk_z = block_z >> 4
        return (chunk_x, chunk_z)

    def get_chunks_from_region(self, region_x, region_z):
        """
        Return list of chunks coordinates which belongs in region
        """
        chunk_1 = (region_x * 32, region_z * 32)
        chunk_2 = (region_x * 32 + 31, region_z * 32 + 31)
        return list(itertools.product(range(chunk_1[0], chunk_2[0]),
                                      range(chunk_1[1], chunk_2[1])))

    def delete_chunk_block_coords(self, dim, x, z):
        """
        Delete chunk at block coordinates for given dimension
        """
        assert dim in self.dimensions, "Dimension %s is not valid" % dim
        region_filename = self.get_region_file(dim, x, z)
        logging.info("Delete chunk containing block (%d, %d) in region %s"
                     % (x, z, os.path.basename(region_filename)))
        rf = RegionFile(region_filename)
        rf.read()
        rf.delete_chunk(*rf.get_chunk_coords(x, z))
        rf.write(region_filename + ".mche")

    def delete_zone(self, dim, block_1, block_2):
        """
        Delete chunks in zone delimited by two points

        block_1 and block_2 are tuples of (x, z) coordinates
        """
        block_x1 = block_1[0]
        block_z1 = block_1[1]
        block_x2 = block_2[0]
        block_z2 = block_2[1]
        region_1 = self.get_region_coords(*block_1)
        region_2 = self.get_region_coords(*block_2)
        chunk_1 = self.get_chunk_coords(*block_1)
        chunk_2 = self.get_chunk_coords(*block_2)

        # Get regions which are included in zone
        regions = itertools.product(range(region_1[0], region_2[0]+1),
                                    range(region_1[1], region_2[1]+1))
        # Get list of chunks in zone
        chunks = list(itertools.product(range(chunk_1[0], chunk_2[0]+1),
                                        range(chunk_1[1], chunk_2[1]+1)))
        logging.debug("Remove chunks between blocks (%d, %d) and (%d, %d)" %
                      (block_x1, block_z1, block_x2, block_z2))
        logging.debug("First Region : (%d, %d)" % (region_1[0], region_1[1]))
        logging.debug("Last Region : (%d, %d)" % (region_2[0], region_2[1]))

        for (r_x, r_z) in regions:
            region_filename = self.get_region_file(dim, r_x*512, r_z*512)
            # Skip non existing regions
            if not os.path.exists(region_filename):
                continue

            logging.debug("Process %s to remove chunks" % region_filename)
            region = RegionFile(region_filename)
            # List of chunks in curent regions
            region_chunks = self.get_chunks_from_region(r_x, r_z)
            # List of chunks that needs to be deleted
            prune_chunks = list(set(region_chunks).intersection(chunks))
            assert len(prune_chunks) > 0,\
                "Region %s is candidate but no chunks to delete"
            # Read region and delete chunks
            region.read()
            for (c_x, c_z) in prune_chunks:
                region.delete_chunk(c_x % 32, c_z % 32)
            # Save region
            region.write(region_filename + ".mche")


def test_region_file():
    rf = RegionFile("/home/pi/mc/juco/region/r.3.3.mca")
    rf.read()
    rf.display_chunk_info(3, 11)
    rf.delete_chunk(3, 11)
    # first_chunk = [c for c in sorted(rf.chunks, key=lambda x: x["offset"])
    #                if c["offset"] > 0][0]
    # rf.display_chunk_info(first_chunk["x"], first_chunk["z"])
    # rf.delete_chunk(first_chunk["x"], first_chunk["z"])

    rf.write("/home/pi/mc/juco/region/r.3.3.mca.mche.RF")

if __name__ == "__main__":
    logging.basicConfig(filename="mche.log", level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())

    world = World("/home/pi/mc/juco")
    # world.delete_chunk_block_coords("overworld", 1584, 1712)
    world.delete_zone("overworld", (1584, 1712), (1584, 1712+510))
    # test_region_file()
