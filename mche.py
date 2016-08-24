#!/usr/bin/env python
import os
import sys
import re
import logging
import itertools
import argparse
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

def bb_intersect(bb1, bb2):
    """
    Return true when two bounding box intersects
    bb1 and bb2 are list of coordinates of the bounding box :
    [x, y, z, x', y', z']
    """
    [bb1_x1, bb1_y1, bb1_z1, bb1_x2, bb1_y2, bb1_z2] = bb1
    [bb2_x1, bb2_y1, bb2_z1, bb2_x2, bb2_y2, bb2_z2] = bb2
    # Check x intersection
    if (bb1_x1 > bb2_x2) or (bb1_x2 < bb2_x1):
        return False

    # Check z intersection
    if (bb1_z1 > bb2_z2) or (bb1_z2 < bb2_z1):
        return False

    return True


class Chunk:
    """
    Chunk object stored in RegionFile
    """
    def __init__(self, x, z, offset, sector_count):
        """
        Initialize with data found in the 4kB header

        x, z : chunk absolute coordinates
        offset : offset in multiple of 4096 where chunk data is stored
        sector_count : number of sector used to store chunk data
        """
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
        # length of chunk data (without padding)
        self.length = None

    def __eq__(self, other):
        """
        Compare coordinates, length, timestamp, sector count and chunk_data
        fields

        offset is ommited because it can change when removing gaps
        """
        return self.x == other.x and \
            self.z == other.z and \
            self.sector_count == other.sector_count and \
            self.timestamp == other.timestamp and \
            self.chunk_data == other.chunk_data and \
            self.length == other.length

    def __ne__(self, other):
        """x.__ne__(y) <==> x != y"""
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
        return not (self.offset == 0 and self.sector_count == 0 and
                    self.timestamp == 0)


class RegionFile:
    """
    Class to handle Minecraft Region file

    Load them, prune a chunk, save to file
    """
    def __init__(self, filename, read=True):
        """
        Initialize instance with region filename attribute
        """
        self.region_filename = filename
        self.x, self.z = self.get_coords()
        self.has_gap = False
        self.chunks = None
        self.location_field = None
        self.timestamps_field = None
        if read:
            self.read()

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
        ret = os.path.basename(self.region_filename)
        ret += " [%d, %d]" % (self.x, self.z)
        ret += " (%d, %d) - (%d, %d)" % (start_x, start_z, end_x, end_z)
        return ret

    def read_header(self, f):
        """Read 8kB header only"""
        if f.tell() != 0:
            f.seek(0)
        # Read location (4096 bytes)
        self.location_field = f.read(4096)
        self.chunks = list()
        # byte offset of chunk coordinates (x,z) : 4((x%32) + (z%32)*32)
        for z_rel in range(32):
            for x_rel in range(32):
                # Offset of current field
                offset = 4*(x_rel + 32*z_rel)
                # location field for current chunk coordinates
                cur_loc = self.location_field[offset:offset+4]
                # Offset of chunk datas in file
                data_offset = int(hexlify(cur_loc[0:3]), 16)
                # Number of sectors (4096 bytes) occupied by chunk
                data_count = int(hexlify(cur_loc[3]), 16)
                (x, z) = self.get_absolute_chunk_coords(x_rel, z_rel)
                self.chunks.append(Chunk(x, z, data_offset, data_count))

        # Read timestamps (4096 bytes)
        self.timestamps_field = f.read(4096)
        for z_rel in range(32):
            for x_rel in range(32):
                # Offset of current chunk's timestamp in timestamp field
                offset = 4*(x_rel + 32*z_rel)
                cur_ts = self.timestamps_field[offset:offset+4]
                cur_ts = int(hexlify(cur_ts), 16)
                self.chunks[offset/4].timestamp = cur_ts

    def read_chunk_datas(self, f):
        """Read chunk data, header must have been read first"""
        assert self.chunks is not None, "Header has not been read"
        total_size = 8192

        # Position file at the end of the header
        if f.tell() != 8192:
            f.seek(8192)
        # Read chunk datas (number of data deduced from location fields
        # go through chunk in the order they are stored in chunk data field
        for c in sorted(self.chunks, key=lambda x: x.offset):
            x = c.x
            z = c.z
            (x_rel, z_rel) = self.get_relative_chunk_coords(x, z)
            # chunk index in chunks list
            chunk_idx = x_rel + 32*z_rel
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

    def read(self):
        """
        Load Region file into memory to perform actions on region

        Usually done once during init
        """
        with open(self.region_filename, "rb") as f:
            self.read_header(f)
            self.read_chunk_datas(f)

    def write(self, filename):
        """
        Write region file to given filename

        Filename can be the same as self.region_filename if you are sure of
        your changes or have made a backup earlier
        """
        with open(filename, "wb") as f:
            # Write locations fields
            for z_rel in range(32):
                for x_rel in range(32):
                    # relative chunk index in region
                    chunk_idx = x_rel + 32*z_rel
                    chunk = self.chunks[chunk_idx]
                    offset_field = get_byte_seq(chunk.offset, 3)
                    sector_count_field = get_byte_seq(chunk.sector_count, 1)
                    location_field = offset_field + sector_count_field
                    f.write(location_field)
            # Write timestamps fields
            for z_rel in range(32):
                for x_rel in range(32):
                    chunk_idx = x_rel + 32*z_rel
                    chunk = self.chunks[chunk_idx]
                    timestamp_field = get_byte_seq(chunk.timestamp, 4)
                    f.write(timestamp_field)

            # write chunk's datas
            for c in sorted(self.chunks, key=lambda x: x.offset):
                # skip chunks with offset 0
                if c.offset != 0:
                    # Seek to correct position where chunk data needs to be
                    if f.tell() != c.offset * 4096:
                        pad_size = c.offset * 4096 - f.tell()
                        f.write('\0'*pad_size)
                    f.write(c.chunk_data)

    def count_gaps(self):
        """Count gaps in bytes between chunks"""
        gap_size = 0
        if self.chunks is None:
            with open(self.region_filename, "rb") as f:
                self.read_header(f)
        # First offset shall be 2 because of 8kB header
        expected_offset = 2
        # Iterate through chunks in offset order
        for c in sorted(self.chunks, key=lambda x: x.offset):
            # Skip ungenerated chunks
            if c.offset != 0:
                assert expected_offset <= c.offset, \
                    "Chunk's offset (%d) cannot be lower than %d" %\
                    expected_offset
                gap_size += 4096*(c.offset - expected_offset)
                expected_offset = c.offset + c.sector_count
        filesize = os.path.getsize(self.region_filename)
        gap_size += filesize - 4096*expected_offset
        assert gap_size < filesize, \
            "gap size (%d) cannog be greater than filesize (%d)." %\
            (gap_size, filesize)
        if gap_size > 0:
            self.has_gap = True
            logging.debug("%d bytes lost between chunks in %s" %
                          (gap_size, self.region_filename))
        else:
            logging.debug("No gap between chunks in %s" %
                          (self.region_filename))
        return gap_size

    def is_chunk_in_region(self, x, z):
        """
        Check if chunk coords is present in region
        """
        return ((self.x*32 <= x) and (x < (self.x + 1)*32) and
                (self.z*32 <= z) and (z < (self.z + 1)*32))

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

    @staticmethod
    def get_coords_str(region_filename):
        """Get region coordinates from filename"""
        filename = os.path.basename(region_filename)
        (x, z) = re.findall("-?\d+", filename)
        return (int(x), int(z))

    def get_coords(self):
        """
        Get region coordinates

        In region coordinates, not in blocks coordinates
        """
        return RegionFile.get_coords_str(self.region_filename)

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

    def get_chunk_index(self, x, z):
        """Get Chunk index in given region"""
        (x_rel, z_rel) = self.get_relative_chunk_coords(x, z)
        return x_rel + 32*z_rel

    def delete_chunk(self, x, z):
        """
        Delete chunk at absolute coords x, z

        Done by setting location and timestamp fields to zero.
        Locations of chunks stored after deleted chunk are updated
        """
        assert self.is_chunk_in_region(x, z), \
            "Chunk (%d, %d) is not in region %s" % (x, z, self.region_filename)

        chunk_idx = self.get_chunk_index(x, z)
        deleted_chunk = self.chunks[chunk_idx]
        if deleted_chunk.offset == 0:
            logging.debug("chunk (%d, %d) has not been generated" % (x, z))
            return
        (x_rel, z_rel) = self.get_relative_chunk_coords(x, z)
        basename_rf = os.path.basename(self.region_filename)
        logging.debug("delete relative chunk (%d, %d) for absolute chunk "
                      "(%d, %d) in region %s" % (x_rel, z_rel, x, z,
                                                 basename_rf))
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
        Get relative coords for chunk's absolutes coordinates
        """
        return (chunk_x % 32, chunk_z % 32)

    def get_chunk_coords(self, block_x, block_z):
        """
        Get relative chunk coords from block's coordinates
        """
        return self.get_relative_chunk_coords(block_x >> 4, block_z >> 4)

    def get_absolute_chunk_coords(self, chunk_rel_x, chunk_rel_z):
        """
        Get absolute chunk coords for relative coordinates
        """
        return (self.x * 32 + chunk_rel_x, self.z * 32 + chunk_rel_z)

    def remove_gaps(self):
        """
        Remove gaps between chunks

        Return number of bytes saved
        """
        # offset of next chunk if no gap, start at 2 because of 8kB header
        next_offset = 2
        gap = 0
        # get list of generated chunks in region
        chunks = [c for c in sorted(self.chunks, key=lambda x: x.offset)
                  if c.is_generated()]
        assert (len(chunks) > 0)
        for c in chunks:
            # Save offset of next chunk before removing gaps
            # used to display space saved
            gap_offset = c.offset + c.sector_count
            if next_offset != c.offset:
                logging.debug("Remove gap : chunk (%d, %d) @offset : "
                              "%d moved to %d" %
                              (c.x, c.z, c.offset, next_offset))
                c.offset = next_offset
                gap = next_offset - c.offset
            # set next_offset to be after current chunk
            next_offset = c.offset + c.sector_count
        bytes_saved = (gap_offset - next_offset)*4096
        logging.info("Saved %d bytes by removing gaps on %s" %
                     (bytes_saved, self.region_filename))
        return bytes_saved

    def diff(self, other):
        """
        Print differences between two region files

        Return true when there is a diff
        """
        is_diff = False
        if self.x != other.x:
            is_diff = True
            print "x coords differs : %d - %d" % (self.x, other.x)
        if self.z != other.z:
            is_diff = True
            print "z coords differs : %d - %d" % (self.z, other.z)
        for idx in range(1024):
            if self.chunks[idx] != other.chunks[idx]:
                is_diff = True
                print("chunk %d differs : (%d, %d) - (%d, %d)" %
                      (idx, self.chunks[idx].x, self.chunks[idx].z,
                       other.chunks[idx].x, other.chunks[idx].z))
                print repr(self.chunks[idx])
                print repr(other.chunks[idx])
        return is_diff

    def get_timestamp_datas(self):
        """
        Return list of triplet to draw a heat map of chunks timestamp

        First two values of the triplet are the chunk's coords, last value is
        the timestamp
        """
        # Read only header if nothing has been read yet
        if self.chunks is None:
            with open(self.region_filename, "rb") as f:
                self.read_header(f)

        # Build data
        data = []
        ddata = dict()
        for c in self.chunks:
            data.append((c.x, c.z, c.timestamp))
            if c.x not in ddata:
                ddata[c.x] = dict()
            ddata[c.x][c.z] = c.timestamp
        return ddata

    def create_gp_ts_map(self, dirname):
        """
        Create Gnuplot files to draw heatmap of chunk's timestamps

        dirname : directory where gnuplot files are created

        Filenames are derived from region filename
        """
        # data file
        timestamp_datas = self.get_timestamp_datas()
        file_prefix = os.path.basename(re.sub(".mca", "",
                                              self.region_filename))
        Mche.create_heatmap(timestamp_datas, dirname, file_prefix)


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

    def get_region_files(self, dim):
        """Return list of region files for dimension"""
        path = self.get_dim_dir(dim)
        region_files = [os.path.join(path, e) for e in os.listdir(path)
                        if re.match("r.-?\d+\.-?\d+\.mca$", e)]
        return region_files

    def count_gaps(self, dimension):
        """
        Count number of bytes in gaps between chunks

        Number of bytes for each region is printed
        """
        print "dimension : %s" % dimension
        total = 0
        for f in self.get_region_files(dimension):
            rf = RegionFile(f, read=False)
            gap = rf.count_gaps()
            total += gap
            print " - %s : %d bytes" % (os.path.basename(f), gap)
        print "-> total gaps in %s : %d bytes" % (dimension, total)

    def remove_gaps(self, dim, suffix=None):
        """
        Remove gaps for region files in given dimension

        Files are saved with suffix name

        Return space saved
        """
        if suffix is None:
            suffix = ""
        files = self.get_region_files(dim)
        # space saved by removing gaps in bytes
        space = 0
        for f in files:
            rf = RegionFile(f)
            if rf.has_gap:
                space += rf.remove_gaps()
                rf.write(f + suffix)
        logging.info("Saved %d bytes by removing gaps in %s region files"
                     % (space, dim))
        return space

    def get_region_name(self, block_x, block_z):
        """
        Return region name for block coordinates
        """
        region = self.get_region_coords(block_x, block_z)
        filename = "r.%d.%d.mca" % (region[0], region[1])
        return filename

    def get_region_file(self, dim, block_x, block_z):
        """
        Return region filename for block coordinates of given dimension
        """
        filename = self.get_region_name(block_x, block_z)

        region_file = os.path.join(self.get_dim_dir(dim), filename)
        return region_file

    def get_region_coords(self, block_x, block_z):
        """
        Return tuple of region coordinates containing specified block
        """
        region_x = (block_x >> 4) >> 5
        region_z = (block_z >> 4) >> 5
        return (region_x, region_z)

    def get_region_idx_range(self, dim):
        """Get Range of region index for dimension"""
        files = self.get_region_files(dim)
        (min_x, min_z) = RegionFile.get_coords_str(files[0])
        (max_x, max_z) = (min_x, min_z)
        for f in files:
            (x, z) = RegionFile.get_coords_str(f)
            min_x = min(min_x, x)
            min_z = min(min_z, z)
            max_x = max(max_x, x)
            max_z = max(max_z, z)
        return ((min_x, max_x), (min_z, max_z))

    @staticmethod
    def get_chunk_coords(block_x, block_z):
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

    def get_coords_by_region(self, blk_coords):
        """
        Sort coords by region

        blk_coords : list of blocks coordinates

        Return dictionnary of list of coords indexed by region name which
        contains the coordinates in the value
        """
        ret = dict()
        for (x, z) in blk_coords:
            region_name = self.get_region_name(x, z)
            if region_name not in ret:
                ret[region_name] = [(x, z)]
            else:
                ret[region_name].append((x, z))
        return ret

    def get_chunks_coords_by_region(self, chunks_coords):
        """
        Sort chunks coords by region.

        chunks_coords : list of chunks coordinates

        Return dictionary  of coords indexed by region name to which they
        belong.
        """
        ret = dict()
        for (x, z) in chunks_coords:
            region_name = self.get_region_name(x*16, z*16)
            if region_name not in ret:
                ret[region_name] = [(x, z)]
            else:
                ret[region_name].append((x, z))
        return ret

    def delete_chunks(self, dim, coords, ext=None):
        """
        Delete list of chunks at chunks coordinates for given dimension

        Region files are saved using ext suffix
        """
        logging.debug("Remove chunks " + str(coords) + " from " + dim)
        if ext is None:
            ext = ""
        assert dim in self.dimensions, "Dimension %s is not valid" % dim

        # organize chunks marked for deletion in dictionary indexed by region
        # filename
        chunks_by_region = self.get_chunks_coords_by_region(coords)

        # iterate on region file / list of chunk coords to be deleted
        for r, coords in chunks_by_region.items():
            rf_name = os.path.join(self.get_dim_dir(dim), r)
            rf = RegionFile(rf_name)
            for (x, z) in coords:
                rf.delete_chunk(x, z)
            rf.write(rf_name + ext)

    def create_gp_ts_map(self, dirname, dim):
        """Create Gnuplot Timestamp map for given dimension"""
        region_files = self.get_region_files(dim)

        # Initial column dictionary
        ranges = self.get_region_idx_range(dim)
        ((min_x, max_x), (min_z, max_z)) = ranges
        init_dict = {key: 0 for key in range(32*min_z, 32*(max_z+1))}

        # Initialize full dict
        datas = dict()
        for x in range(32*min_x, 32*(max_x+1)):
            datas[x] = dict(init_dict)

        # Fill dict with actual values
        for f in region_files:
            rf = RegionFile(f, False)
            rf_data = rf.get_timestamp_datas()
            # merge
            for (key, item_dict) in rf_data.items():
                for (z, ts) in item_dict.items():
                    assert datas[key][z] == 0
                    datas[key][z] = ts

        Mche.create_heatmap(datas, dirname, dim)


class Mche:
    """Defines methods to run script according to options passed"""

    def __init__(self, opts):
        """Initialize object with options parsed from command line"""
        self.__dict__ = opts
        self.world = World(self.world_name)

    def run(self):
        """Execute requested operations based on options"""
        coords = self.get_delete_coords()

        # Delete requested chunks
        if len(coords) > 0:
            self.world.delete_chunks(self.dimension, coords, self.suffix)

        # Remove gaps between chunks
        if self.remove_gaps:
            world.remove_gaps()

        # Generate informations
        if self.info:
            # Heatmap
            folder = "heatmap_%s" % self.dimension
            # create folder
            if not os.path.exists(folder):
                os.mkdir(folder)
            self.world.create_gp_ts_map(folder, self.dimension)
            print "Heatmap created under %s" % folder
            # gaps
            self.world.count_gaps(self.dimension)

    @staticmethod
    def order_zone(coords1, coords2):
        """Return tuple of coords ordered (top-left, bottom-right)"""
        coords_x1 = min(coords1[0], coords2[0])
        coords_z1 = min(coords1[1], coords2[1])
        coords_x2 = max(coords1[0], coords2[0])
        coords_z2 = max(coords1[1], coords2[1])
        return ((coords_x1, coords_z1), (coords_x2, coords_z2))

    def get_delete_coords(self):
        """Build list of chunks coordinates that needs to be deleted"""
        # adds chunks given by chunks coords
        if self.del_chunk_coords is not None:
            ret = list(self.del_chunk_coords)
        else:
            ret = list()

        # adds chunks given by block coords
        if self.del_chunk_blk_coords is not None:
            for (blk_x, blk_z) in self.del_chunk_blk_coords:
                chunk_coords = World.get_chunk_coords(blk_x, blk_z)
                ret.append(chunk_coords)

        # adds chunks given by chunks zones
        if self.del_zone_coords is not None:
            for zone in self.del_zone_coords:
                (chunk_1, chunk_2) = Mche.order_zone(*zone)
                chunks = list(itertools.product(range(chunk_1[0],
                                                      chunk_2[0]+1),
                                                range(chunk_1[1],
                                                      chunk_2[1]+1)))
                ret.extend(chunks)

        # Adds chunks given by blocks zones
        if self.del_zone_blk_coords is not None:
            print "here"
            for zone in self.del_zone_blk_coords:
                (coords_1, coords_2) = Mche.order_zone(*zone)
                chunk_1 = World.get_chunk_coords(*coords_1)
                chunk_2 = World.get_chunk_coords(*coords_2)
                chunks = list(itertools.product(range(chunk_1[0],
                                                      chunk_2[0]+1),
                                                range(chunk_1[1],
                                                      chunk_2[1]+1)))
                ret.extend(chunks)

        # Uniquify coords to remove overlaps
        return list(set(ret))

    @staticmethod
    def create_heatmap(datas, path, file_prefix):
        """
        Create Gnuplot script for heatmap of datas

        datas : list of triplet to draw heatmap from, first two are corods,
        third is heatness
        path : directory where datas are saved (.dat, .gnu)
        file_prefix : filename prefix where datas/script are stored
        """
        # list of timestamps
        ts = list()
        # data file
        dat_file = os.path.join(path, file_prefix + ".dat")
        with open(dat_file, "wb") as f:
            for (k_x, d_x) in sorted(datas.items()):
                for (k_z, d_xz) in sorted(d_x.items()):
                    f.write("%d %d %d\n" % (k_x, k_z, d_xz))
                    if d_xz != 0:
                        ts.append(d_xz)
                f.write("\n")

        # gnuplot script
        gp_file = os.path.join(path, file_prefix + ".gnu")
        png_file = file_prefix + ".png"
        with open(gp_file, "wb") as f:
            # gnuplot count number of second since 2000, != than epoch
            gp_offset = 946684800
            f.write("set terminal png\n")
            f.write('set title "Heat map of chunks modification dates"\n')
            f.write('set output "%s"\n' % png_file)
            f.write("\n")

            # palette
            f.write('# Configure palette\n')
            f.write('set cbdata time\n')
            f.write('set format cb "\%m-\%Y"\n')
            f.write('set timefmt "\%s"\n')
            f.write("set cbrange [%d:%d]\n" % (min(ts)-gp_offset,
                                               max(ts)-gp_offset))
            f.write("\n")

            # xy tics
            f.write('# Configure xy axis\n')
            f.write('set xtics 32\n')
            f.write('set ytics 32\n')
            f.write('set yrange [*:*] reverse\n')
            f.write("\n")

            f.write("set view map\n")
            f.write("plot '%s' using 1:2:($3-%d) with image notitle\n" %
                    (os.path.basename(dat_file), gp_offset))
            f.write("# vim: set syntax=gnuplot:\n")


def get_coords_from_str(s):
    """
    Return list of coords from string

    a string coordinate is formatted as 'X0xZ0' where X0 and Z0 are integers
    Multiple coords are separated with a comma (eg : 'X0xZ0,X1xZ1')
    """
    # list of coords as strings
    l_s = s.split(",")
    ret = list()
    for c_s in l_s:
        # Split coord into two integers
        coords_s = c_s.split("x")
        # Check we have only two integers
        assert len(coords_s) == 2, "Coords should have two values, %s has %d" \
            % (c_s, len(coords_s))
        # Cast from string to integers
        coords = tuple([int(c) for c in coords_s])
        ret.append(coords)
    return ret


def get_str_from_coords(coords):
    """Return string describing list of coords"""
    s = ""
    for coord in coords:
        if type(coord) == int:
            s += "%dx" % coord
        else:
            (x, z) = coord
            s += "%dx%d," % (x, z)
    # remove last ',' or 'x'
    s = s[0:-1]
    return s


def opt_chunk_delete(parser, value):
    """Callback to parse coordinates and put them into a list"""
    return get_coords_from_str(value)


def get_zones_from_str(s):
    """
    Return tuple of two coords from string

    string if formated as 'X0xZ0_X1xZ1'
    Multiple zones are separated with comma
    """
    zones = s.split(",")
    ret = list()
    for z in zones:
        coords = z.split("_")
        assert len(coords) == 2, "Zone should have two coords"
        zone = list()
        for c in coords:
            zone.append(get_coords_from_str(c)[0])
        ret.append(zone)
    return ret


def get_str_from_zones(zones):
    """Return string describing zones"""
    s = ""
    for z in zones:
        s += "%s_%s," % (get_str_from_coords(z[0]), get_str_from_coords(z[1]))
    # remove last ','
    s = s[0:-1]
    return s


def opt_zone_delete(parser, value):
    """Callback to parse zones and put them into a list"""
    return get_zones_from_str(value)


def main():
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description="Manipulate Region Files of "
                                     "Minecraft World",
                                     formatter_class=formatter_class)
    parser.add_argument("world_name", help="World Folder")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Enable debug trace in log file",
                        default=False)
    parser.add_argument("--delete-chunk", dest="del_chunk_coords",
                        type=lambda x: opt_chunk_delete(parser, x),
                        help="Delete one or multiple chunk at provided"
                        "coordinates. A single chunk coordinate is written as "
                        "X0xZ0 where X0 and Z0 are integers. Multiple chunk "
                        "coordinates are separated by comma "
                        "(eg : X0xZ0,X1xZ1)")
    parser.add_argument("--del-chunk-at-block", dest="del_chunk_blk_coords",
                        type=lambda x: opt_chunk_delete(parser, x),
                        help="Delete one or multiple chunk at provided block "
                        "coordinates. A single chunk coordinate is written as "
                        "X0xZ0 where X0 and Z0 are integers. Multiple blocks "
                        "coordinates are separated by comma "
                        "(eg : X0xZ0,X1xZ1)")
    parser.add_argument("--delete-zone", dest="del_zone_coords",
                        type=lambda x: opt_zone_delete(parser, x),
                        help="Delete zone in rectangle specified by the "
                        "chunks coordinates of the two opposite corners. "
                        "Coords of opposite chunks are separated by an "
                        "underscore (eg : X0xZ0_X1xZ1). "
                        "Multiple zones are separated by a comma.")
    parser.add_argument("--del-zone-between-blocks",
                        dest="del_zone_blk_coords",
                        type=lambda x: opt_zone_delete(parser, x),
                        help="Delete zone in rectangle specified by the "
                        "blocks coordinates of the two opposite corners. "
                        "Coords of opposite blocks are separated by an "
                        "underscore (eg : X0xZ0_X1xZ1). "
                        "Multiple zones are separated by a comma.")
    parser.add_argument("--dimension", action="store", dest="dimension",
                        choices=World.dimensions,
                        type=str, default="overworld",
                        help="Specify dimension to work on")
    parser.add_argument("--suffix", "-s", action="store", dest="suffix",
                        type=str, default="",
                        help="Write mca file to suffixed file, default "
                        "overwite")
    parser.add_argument("--remove-gaps", action="store_true", default=False,
                        help="Remove gaps in Region Files between chunks")
    parser.add_argument("--info", "-i", action="store_true", default=False,
                        help="Gather informations on Region files (gaps "
                        "between chunks, number of chunks generated, map of "
                        "chunks generated colored by timestamp date of last "
                        "modification")

    args = parser.parse_args()

    # Setup logger
    logging.basicConfig(filename="mche.log")
    logging.getLogger().addHandler(logging.StreamHandler())
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    mche = Mche(args.__dict__)
    mche.run()
    sys.exit(0)

if __name__ == "__main__":
    main()
