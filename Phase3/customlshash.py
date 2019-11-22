
import os
import numpy as np
from numpy.distutils.extension import basestring
from storage import storage
wind_size = 5
offset = np.random.randint(wind_size)

try:
    from bitarray import bitarray
except ImportError:
    bitarray = None

class LSHash(object):

    def __init__(self, hash_size, input_dim, num_hashtables=1,
                 storage_config=None, matrices_filename=None, overwrite=False):

        self.hash_size = hash_size
        self.input_dim = input_dim
        self.num_hashtables = num_hashtables

        if storage_config is None:
            storage_config = {'dict': None}
        self.storage_config = storage_config

        if matrices_filename and not matrices_filename.endswith('.npz'):
            raise ValueError("The specified file name must end with .npz")
        self.matrices_filename = matrices_filename
        self.overwrite = overwrite
        self._init_uniform_planes()
        self._init_hashtables()

    def _init_uniform_planes(self):

        if "uniform_planes" in self.__dict__:
            return

        if self.matrices_filename:
            file_exist = os.path.isfile(self.matrices_filename)
            if file_exist and not self.overwrite:
                try:
                    npzfiles = np.load(self.matrices_filename)
                except IOError:
                    print("Cannot load specified file as a numpy array")
                    raise
                else:
                    npzfiles = sorted(npzfiles.items(), key=lambda x: x[0])
                    self.uniform_planes = [t[1] for t in npzfiles]
            else:
                self.uniform_planes = [self._generate_uniform_planes()
                                       for _ in range(self.num_hashtables)]
                try:
                    np.savez_compressed(self.matrices_filename,
                                        *self.uniform_planes)
                except IOError:
                    print("IOError when saving matrices to specificed path")
                    raise
        else:
            self.uniform_planes = [self._generate_uniform_planes()
                                   for _ in range(self.num_hashtables)]

    def _init_hashtables(self):
        self.hash_tables = [storage(self.storage_config, i)
                            for i in range(self.num_hashtables)]

    def _generate_uniform_planes(self):
        return np.random.randn(self.hash_size, self.input_dim)

    def _hash(self, planes, input_point):
        try:
            input_point = np.array(input_point)  # for faster dot product
            projections = np.dot(planes, input_point)
        except TypeError as e:
            print("""The input point needs to be an array-like object with
                          numbers only elements""")
            raise
        except ValueError as e:
            print("""The input point needs to be of the same dimension as
                          `input_dim` when initializing this LSHash instance""", e)
            raise
        else:
            return "".join(['1' if i > 0 else '0' for i in projections])

    def _as_np_array(self, json_or_tuple):
        tuples = json_or_tuple

        if isinstance(tuples[0], tuple):
            return np.asarray(tuples[0])

        elif isinstance(tuples, (tuple, list)):
            try:
                return np.asarray(tuples)
            except ValueError as e:
                print("The input needs to be an array-like object", e)
                raise
        else:
            raise TypeError("query data is not supported")

    def index(self, input_point, extra_data=None, image_id=''):
        if isinstance(input_point, np.ndarray):
            input_point = input_point.tolist()


        value = tuple(input_point)
        for i, table in enumerate(self.hash_tables):
            table.append_val(self._hash(self.uniform_planes[i], input_point),
                             value, image_id)

    def query(self, query_point, num_results=None, distance_func=None, image_id=''):
        candidates = set()
        if not distance_func:
            distance_func = "euclidean"

        if distance_func == "euclidean":
            d_func = LSHash.euclidean_dist_square
        elif distance_func == "true_euclidean":
            d_func = LSHash.euclidean_dist
        else:
            raise ValueError("The distance function name is invalid.")

        for i, table in enumerate(self.hash_tables):
            binary_hash = self._hash(self.uniform_planes[i], query_point)
            candidates.update(table.get_list(binary_hash))

        candidates = [(ix[0], ix[1], d_func(query_point, self._as_np_array(ix)))
                      for ix in candidates]
        candidates.sort(key=lambda x: x[2])
        print('Query Image:', image_id)
        return candidates[:num_results] if num_results else candidates



    @staticmethod
    def euclidean_dist(x, y):
        diff = np.array(x) - y
        return np.sqrt(np.dot(diff, diff))

    @staticmethod
    def euclidean_dist_square(x, y):
        diff = np.array(x) - y
        return np.dot(diff, diff)

    @staticmethod
    def euclidean_dist_centred(x, y):
        diff = np.mean(x) - np.mean(y)
        return np.dot(diff, diff)
