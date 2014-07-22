import json
import itertools
from threading import Thread
import logging

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

import numpy as np
import six
import zmq


from zmq.eventloop.zmqstream import ZMQStream
from zmq.eventloop import ioloop
ioloop.install()


import tornado.websocket
import tornado.web
import tornado.ioloop


class WebSocket(tornado.websocket.WebSocketHandler):
    def __init__(self, application, request, **kwargs):
        # self.zmqstream = kwargs.pop('zmqstream')
        tornado.websocket.WebSocketHandler.__init__(self, application, request,
                                            **kwargs)
        self.metadata = None

    def initialize(self, zmqstream, database):
        self.zmqstream = zmqstream
        self.database = database
    def open(self, key):
        logger.debug("websocket opened for key %s", key)
    def on_message(self, message):
        # unicode, metadata message
        logger.debug("got message %s", message)
        if isinstance(message, six.text_type):
            self.metadata = json.loads(message)
            logger.debug("got metadata %s", self.metadata)
        else:
            # assume we already have metadata
            assert self.metadata, "got message without preceding metadata"
            dtype = self.metadata['dtype']
            shape = self.metadata['shape']
            # use the zmqstream as a socket (send, send_json)
            socket = self.zmqstream
            # unpack array
            A = np.fromstring(message, dtype=dtype)
            A = A.reshape(shape)
            # pass it along to the socket
            send_array(socket, A, self.metadata)
            self.metadata = None
    def on_close(self):
        logger.debug("websocket closed")

class MainHandler(tornado.web.RequestHandler):
    def initialize(self, database):
        self.database = database
    def get(self):
        self.write("%s" % (self.database,))

class ModelHandler(tornado.web.RequestHandler):
    def initialize(self, database):
        self.database = database
    def post(self, key):
        # TODO: show a list of running models
        self.database[key] = json.loads(self.request.body)

def main():
    ctx = zmq.Context()
    socket = ctx.socket(zmq.PUB)
    socket.bind("tcp://*:5600")
    zmqstream = ZMQStream(socket)
    database = {}
    application = tornado.web.Application([
        (r"/", MainHandler, {"database": database}),
        # todo use an id scheme to attach to multiple models
        (r"/models/(.*)", ModelHandler, {"database": database}),
        (r"/mmi/(.*)", WebSocket, {"zmqstream": zmqstream, "database": database}),
    ])
    application.listen(8888)
    tornado.ioloop.IOLoop.instance().start()

if __name__ == "__main__":
    main()
