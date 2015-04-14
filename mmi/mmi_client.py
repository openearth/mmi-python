import zmq
import logging

from mmi import send_array, recv_array

logger = logging.getLogger(__name__)


class MMIClient(object):
    def __init__(self, uuid, mmi_metadata):
        """
        The 'database' has mmi module metadata.

        the metadata must contain the key "ports"
        "ports":  {'PUSH': 58452, 'REQ': 53956, 'SUB': 60285}
        """
        logger.debug("Initializing MMI Client [%s]..." % uuid)
        self.uuid = uuid
        self.database = mmi_metadata
        self.ports = mmi_metadata['ports']

        self.sockets = {}
        self.context = zmq.Context()

        logger.debug("Connecting to push/pull server...")
        if 'PUSH' in self.ports:
            logger.debug("MMI PUSH is available")
            self.sockets['PUSH'] = self.context.socket(zmq.PUSH)
            # TODO: is this correct?
            url = 'tcp://%s:%d' % (self.database['node'], self.ports['PUSH'])
            self.sockets['PUSH'].connect(url)

        if 'SUB' in self.ports:
            logger.debug("MMI SUB is available")
            self.sockets['SUB'] = self.context.socket(zmq.SUB)
            url = 'tcp://%s:%d' % (self.database['node'], self.ports['SUB'])
            self.sockets['SUB'].connect(url)

        if 'REQ' in self.ports:
            logger.debug("MMI REQ is available")
            self.sockets['REQ'] = self.context.socket(zmq.REQ)
            url = 'tcp://%s:%d' % (self.database['node'], self.ports['REQ'])
            self.sockets['REQ'].connect(url)

    def __getitem__(self, key):
        """For direct indexing the MMIClient object as a dict"""
        return self.database[key]

    # from here: BMI commands that gets translated to MMI.
    def initialize(self, configfile=None):
        """
        """
        pass

    def finalize(self):
        """
        """
        pass

    def update(self, dt=-1):
        """
        """
        pass

    def get_var_count(self):
        """
        """
        pass

    def get_var_name(self, i):
        pass

    def get_var_type(self, name):
        # TODO
        logger.debug('get_var_type')
        metadata = {'get_var_type': name}
        send_array(self.sockets['REQ'], None, metadata=metadata)
        arr, result_meta = recv_array(self.sockets['REQ'])
        return result_meta['get_var_type']
        pass

    def inq_compound(self, name):
        pass

    def inq_compound_field(self, name, index):
        pass

    def make_compound_ctype(self, varname):
        pass

    def get_var_rank(self, name):
        # TODO
        logger.debug('get_var_rank')
        metadata = {'get_var_rank': name}
        send_array(self.sockets['REQ'], None, metadata=metadata)
        arr, result_meta = recv_array(self.sockets['REQ'])
        return int(result_meta['get_var_rank'])

    def get_var_shape(self, name):
        # TODO
        logger.debug('get_var_shape')
        metadata = {'get_var_shape': name}
        send_array(self.sockets['REQ'], None, metadata=metadata)
        arr, result_meta = recv_array(self.sockets['REQ'])
        return tuple(result_meta['get_var_shape'])

    def get_start_time(self):
        pass

    def get_end_time(self):
        pass

    def get_current_time(self):
        pass

    def get_time_step(self):
        pass

    def get_var(self, name):
        # TODO
        logger.debug('get_var')
        metadata = {'get_var': name}
        send_array(self.sockets['REQ'], None, metadata=metadata)
        arr, result_meta = recv_array(self.sockets['REQ'])
        return arr

    def set_var(self, name, var):
        pass

    def set_var_slice(self, name, start, count, var):
        pass

    def set_var_index(self, name, index, var):
        pass

    def set_structure_field(self, name, id, field, value):
        pass

    def set_logger(self, logger):
        pass

    def __enter__(self):
        pass

    def __exit__(self, type, value, tb):
        pass
