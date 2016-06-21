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


class Chunk:
    def __init__(self, x, z, offset, sector_count):
        self.x = x
        self.z = z
        self.offset = offset
        self.sector_count = sector_count
        self.timestamp = None
        # chunk_data includes padding to multiple of sector size
        # length : 4 bytes
        # compression kind : 1 byte
        # chunk payload : length - 1 bytes
        self.chunk_data = None
        # length of chunk data (withou
        self.length = None

    def __eq__(self, other):
        """
        Compare all fields of Chunk
        """
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        s = "x:%d - z:%d - offset:%d - sector_count:%d" % (self.x, self.z,
                                                           self.offset,
                                                           self.sector_count)
        s = s + " - timestamp:%d - length:%d" % (self.timestamp, self.length)
        s = s + " - hash(chunk_data):%d" % self.chunk_data.__hash__()
        return s

    def __str__(self):
        s = "(%d, %d) at sector %d (byte offset %d) using %d sectors" %\
            (self.x, self.z, self.offset, self.offset*4096, self.sector_count)
        if self.length is not None:
            s = s + ", chunk data uses %d bytes" % self.length
        return s

    def is_generated(self):
        """Return True if chunk has been generated"""
        return self.offset == 0 and self.sector_count == 0 \
                and self.timestamp == 0


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
        self.has_gap = False
        self.chunks = None
        self.location_field = None
        self.timestamps_field = None

    def __eq__(self, other):
        """
        Compare region coords and chunks equalities
        """
        if self.x != other.x:
            return False
        if self.z != other.z:
            return False
        for c_idx in range(1024):
            c1 = self.chunks[c_idx]
            c2 = other.chunks[c_idx]
            if c1 != c2:
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

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
                    self.chunks.append(Chunk(x, z, data_offset, data_count))

            # Read timestamps (4096 bytes)
            self.timestamps_field = f.read(4096)
            for z in range(32):
                for x in range(32):
                    # Offset of current chunk's timestamp in timestamp field
                    offset = 4*(x + 32*z)
                    cur_ts = self.timestamps_field[offset:offset+4]
                    cur_ts = int(hexlify(cur_ts), 16)
                    self.chunks[offset/4].timestamp = cur_ts

            # Read chunk datas (number of data deduced from location fields
            total_size = 8192
            # go through chunk in the order they are stored in chunk data field
            for c in sorted(self.chunks, key=lambda x: x.offset):
                x = c.x
                z = c.z
                # chunk index in chunks list
                chunk_idx = x + 32*z
                size = 4096 * c.sector_count
                if size > 0:
                    # chunk generated
                    if f.tell() < c.offset * 4096:
                        # Current position does not match position of current
                        # chunk This happen if a chunk size reduces below a
                        # multiple of sector size, the following chunks are not
                        # relocated
                        gap = c.offset * 4096 - f.tell()
                        logging.warning("Region %s, %d bytes gap before %s"
                                        % (self, gap, c))
                        self.has_gap = True
                        f.seek(c.offset * 4096)
                    total_size += size
                    c.chunk_data = f.read(size)
                    length = int(hexlify(c.chunk_data[0:4]), 16)
                    c.length = length
                    if length > size:
                        logging.warning("chunk (%d, %d) uses %d sector "
                                        "(%d bytes), chunk data length is %d"
                                        % (x, z, c.sector_count, size, length))
                        assert length > size
                # logging.debug("Stored %s", c)

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
                    offset_field = get_byte_seq(chunk.offset, 3)
                    sector_count_field = get_byte_seq(chunk.sector_count, 1)
                    location_field = offset_field + sector_count_field
                    f.write(location_field)
            # Write timestamps fields
            for z in range(32):
                for x in range(32):
                    chunk_idx = x + 32*z
                    chunk = self.chunks[chunk_idx]
                    timestamp_field = get_byte_seq(chunk.timestamp, 4)
                    f.write(timestamp_field)

            # write chunk's datas
            for c in sorted(self.chunks, key=lambda x: x.offset):
                x = c.x
                z = c.z
                # skip chunks with offset 0
                if c.offset != 0:
                    # Seek to correct position where chunk data needs to be
                    if f.tell() != c.offset * 4096:
                        pad_size = c.offset * 4096 - f.tell()
                        f.write('\0'*pad_size)
                    f.write(c.chunk_data)

    def display_chunk_info(self, x, z):
        """
        Display chunks info of chunk coordinates (x,z)

        Coordinates of chunks are relative coords from the region file
        """
        # Index of chunk in chunks list
        chunk_idx = (x % 32) + 32*(z % 32)
        print "offset : %d" % self.chunks[chunk_idx].offset
        print "sector count : %d" % self.chunks[chunk_idx].sector_count
        print "x : %d" % self.chunks[chunk_idx].x
        print "z : %d" % self.chunks[chunk_idx].z
        print "timestamp : %d" % self.chunks[chunk_idx].timestamp
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
        if deleted_chunk.offset == 0:
            logging.debug("chunk (%d, %d) has not been generated" % (x, z))
            return
        logging.debug("delete relative chunk (%d, %d) in region %s"
                      % (x, z, os.path.basename(self.region_filename)))
        # List of chunks that needs to be updated
        update_chunks = [c for c in self.chunks
                         if c.offset > deleted_chunk.offset]
        # recompute offset of chunks stored after delted chunk
        for c in update_chunks:
            c.offset = c.offset - deleted_chunk.sector_count
        # reset offset, sector count and timestamp of deleted chunk
        deleted_chunk.offset = 0
        deleted_chunk.sector_count = 0
        deleted_chunk.timestamp = 0

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

    @staticmethod
    def get_chunks_from_region(region_x, region_z):
        """
        Return list of chunks coordinates which belongs in region
        """
        chunk_1 = (region_x * 32, region_z * 32)
        chunk_2 = (region_x * 32 + 31, region_z * 32 + 31)
        return list(itertools.product(range(chunk_1[0], chunk_2[0]+1),
                                      range(chunk_1[1], chunk_2[1]+1)))

    def delete_chunk_block_coords(self, dim, x, z, ext=None):
        """
        Delete chunk at block coordinates for given dimension

        Region file is overwritten unless ext is provided, in which case it is
        used to suffix region file
        """
        if ext is None:
            ext = ""
        assert dim in self.dimensions, "Dimension %s is not valid" % dim
        region_filename = self.get_region_file(dim, x, z)
        logging.info("Delete chunk containing block (%d, %d) in region %s"
                     % (x, z, os.path.basename(region_filename)))
        rf = RegionFile(region_filename)
        rf.read()
        rf.delete_chunk(*rf.get_chunk_coords(x, z))
        rf.write(region_filename + ext)

    def delete_zone(self, dim, block_1, block_2, ext=None):
        """
        Delete chunks in zone delimited by two points

        block_1 and block_2 are tuples of (x, z) coordinates
        """
        if ext is None:
            ext = ""
        # set block1 as top-left, block2 as bottom-right
        block_x1 = min(block_1[0], block_2[0])
        block_z1 = min(block_1[1], block_2[1])
        block_x2 = max(block_1[0], block_2[0])
        block_z2 = max(block_1[1], block_2[1])
        region_1 = self.get_region_coords(block_x1, block_z1)
        region_2 = self.get_region_coords(block_x2, block_z2)
        chunk_1 = self.get_chunk_coords(block_x1, block_z1)
        chunk_2 = self.get_chunk_coords(block_x2, block_z2)

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
        logging.debug("list of chunks : %s" % chunks)
        logging.debug("chunk_1 : %s" % str(chunk_1))
        logging.debug("chunk_2 : %s" % str(chunk_2))

        for (r_x, r_z) in regions:
            region_filename = self.get_region_file(dim, r_x*512, r_z*512)
            # Skip non existing regions
            if not os.path.exists(region_filename):
                continue

            logging.debug("Process %s to remove chunks" % region_filename)
            region = RegionFile(region_filename)
            # List of chunks in curent regions
            region_chunks = World.get_chunks_from_region(r_x, r_z)
            # List of chunks that needs to be deleted
            prune_chunks = list(set(region_chunks).intersection(chunks))
            logging.debug("remove chunks %s from %s"
                          % (str(prune_chunks),
                             os.path.basename(region_filename)))
            assert len(prune_chunks) > 0,\
                "Region %s is candidate but no chunks to delete"
            # Read region and delete chunks
            region.read()
            for (c_x, c_z) in prune_chunks:
                logging.debug("Remove chunk (%d, %d)" % (c_x, c_z))
                region.delete_chunk(c_x % 32, c_z % 32)
            # Save region
            region.write(region_filename + ext)


# TODO
# options : arg
# --delete-chunk : chunk coords
# --delete-zone : chunk coords - chunk coords
# --del-chunk-at-block : block coords
# --del-zone-between-blocks : block coords - block coords
# --suffix, -s : ext
# --info, -i : gaps, number of chunks generated, histogram by timestamps
# --remove-gaps, -r : none
# --debug, -d : none
# --dimension : overworld, nether, theend

if __name__ == "__main__":
    logging.basicConfig(filename="mche.log", level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())

    # world = World("/home/pi/mc/juco")
    world = World("/home/nicolas/MinecraftServer/Creatif/juco")
    world.delete_zone("overworld", (393, -10), (405, -20))
