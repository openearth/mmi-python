import zmq
import logging

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
            self.sockets['SUB'] = self.context.socket(zmq.PUSH)
            url = 'tcp://%s:%d' % (self.database['node'], self.ports['SUB'])
            self.sockets['SUB'].connect(url)

        if 'REQ' in self.ports:
            logger.debug("MMI REQ is available")
            self.sockets['REQ'] = self.context.socket(zmq.PUSH)
            url = 'tcp://%s:%d' % (self.database['node'], self.ports['REQ'])
            self.sockets['REQ'].connect(url)

    def __getitem__(self, key):
        """For direct indexing the MMIClient object as a dict"""
        return self.database[key]
