import uuid
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

from . import send_array

class WebSocket(tornado.websocket.WebSocketHandler):
    def __init__(self, application, request, **kwargs):
        # self.zmqstream = kwargs.pop('zmqstream')
        tornado.websocket.WebSocketHandler.__init__(self, application, request,
                                            **kwargs)
        self.metadata = None

    def initialize(self, database, ctx):
        self.database = database
        self.ctx = ctx
    def open(self, key):
        logger.debug("websocket opened for key %s", key)
        self.key = key
        # openeing corresponding socket of model

        # open push socket to forward incoming zmq messages
        socket = self.ctx.socket(zmq.PUSH)
        pull = self.database[key]["ports"]['PULL']
        socket.connect("tcp://localhost:%d" % (pull,))
        self.pushstream = ZMQStream(socket)

        socket = self.ctx.socket(zmq.SUB)
        pub = self.database[key]["ports"]['PUB']
        socket.connect("tcp://localhost:%d" % (pub,))
        # Accept all messages
        socket.setsockopt(zmq.SUBSCRIBE, '')
        self.substream = ZMQStream(socket)

        def send(messages):
            """forward messages to this websocket"""
            logger.info("received %s messages", len(messages))
            for message in messages:
                try:
                    json.loads(message)
                    binary = False
                except ValueError:
                    binary = True
                self.write_message(message, binary)
        self.substream.on_recv(send)

    def on_message(self, message):
        # unicode, metadata message
        logger.debug("got message %s", message)

        # Let's try and forward it.
        # use the zmqstream as a socket (send, send_json)
        socket = self.pushstream

        if isinstance(message, six.text_type):
            metadata = json.loads(message)
            logger.debug("got metadata %s", metadata)
            if not ("dtype" in metadata):
                # no array expected, pass along:
                A = None
                logger.debug("sending metadata %s to %s", metadata, socket)
                send_array(socket, None, metadata)
                self.metadata = None
            else:
                # We expect another message
                self.metadata = metadata
        else:
            # assume we already have metadata
            assert self.metadata, "got message without preceding metadata"
            dtype = self.metadata['dtype']
            shape = self.metadata['shape']
            # unpack array
            A = np.fromstring(message, dtype=dtype)
            A = A.reshape(shape)
            # pass it along to the socket (push)
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
    def get(self, key=None):
        """Register a new model (models)"""
        self.set_header("Access-Control-Allow-Origin", "*")
        if key is not None:
            result = json.dumps(self.database[key])
        else:
            result = json.dumps(self.database)
        self.write(result)
    def post(self):
        """Register a new model (models)"""
        key = uuid.uuid4().hex
        self.database[key] = json.loads(self.request.body)
        result = json.dumps({"uuid": key})
        self.write(result)
    def put(self, key, *args, **kwargs):
        # TODO: show a list of running models
        self.database[key] = json.loads(self.request.body)
    def delete(self, key, *args, **kwargs):
        del self.database[key]


def main():
    ctx = zmq.Context()
    # register socket
    socket = ctx.socket(zmq.PULL)
    socket.bind("tcp://*:6000")
    zmqstream = ZMQStream(socket)
    database = {}
    application = tornado.web.Application([
        (r"/", MainHandler, {"database": database}),
        # todo use an id scheme to attach to multiple models
        (r"/models", ModelHandler, {"database": database}),
        (r"/models/(.*)?", ModelHandler, {"database": database}),
        (r"/mmi/(.*)", WebSocket, {"database": database, "ctx": ctx}),
    ])
    application.listen(8888)
    tornado.ioloop.IOLoop.instance().start()

if __name__ == "__main__":
    main()
