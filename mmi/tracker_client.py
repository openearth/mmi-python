import urllib2
import json


class MMITracker(object):
    def __init__(self, tracker_url):
        """
        The 'database' has module uuids as key and its metadata as value.
        """
        self.tracker_url = tracker_url
        self.database = {}

    def update(self):
        """Update myself with the tracker server"""
        response = urllib2.urlopen(self.tracker_url)
        contents = response.read()
        self.database = json.loads(contents)

    def key_occurrence(self, key, update=True):
        """
        Return a dict containing the value of the provided key and its uuid as
        value.
        """
        if update:
            self.update()
        result = {}
        for k, v in self.database.items():
            if key in v:
                result[str(v[key])] = k
        return result

    def zmq_address(self, key):
        """
        Return a ZeroMQ address to the module with the provided key.
        """
        zmq_address = "tcp://" + self.database[key]['node'] + ":" + str(self.database[key]['ports']['REQ'])
        return zmq_address
