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

ctx = zmq.Context()

class WebSocket(tornado.websocket.WebSocketHandler):
    def __init__(self, application, request, **kwargs):
        tornado.websocket.WebSocketHandler.__init__(self, application, request,
                                            **kwargs)
        self.metadata = None
        self.zmqstream = None

    def open(self):
        logger.debug("websocket opened")
        self.socket = ctx.socket(zmq.PUB)
        self.socket.bind("tcp://*:5600")
        self.stream = ZMQStream(self.socket)

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
    def get(self):
        # TODO: show a list of running models
        self.write("Hello, world")

def main():
    application = tornado.web.Application([
        (r"/", MainHandler),
        # todo use an id scheme to attach to multiple models
        (r"/model", WebSocket),
    ])
    application.listen(8888)
    tornado.ioloop.IOLoop.instance().start()

if __name__ == "__main__":
    main()
