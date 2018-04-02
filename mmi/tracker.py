import uuid
import json
import logging

# compatibility
import six

# numpy
import numpy as np

# messages
import zmq
from zmq.eventloop.zmqstream import ZMQStream
from zmq.eventloop import ioloop

# server
import tornado.websocket
import tornado.web
import tornado.ioloop

# mmi
from . import send_array
from .tracker_views import views

ioloop.install()

SOCKET_NAMES = {
    int(getattr(zmq, name)): name
    for name
    in {"PUSH", "PULL", "SUB",
        "PUB", "REQ", "REP", "PAIR"}
}


logging.basicConfig()
logger = logging.getLogger(__name__)




class WebSocket(tornado.websocket.WebSocketHandler):
    def __init__(self, application, request, **kwargs):
        # self.zmqstream = kwargs.pop('zmqstream')
        tornado.websocket.WebSocketHandler.__init__(
            self, application, request, **kwargs)
        self.metadata = None

    def initialize(self, database, ctx):
        # TODO: use database that supports timeout, persistency, logging, key value store
        # perhaps redis
        self.database = database
        self.ctx = ctx

    def open(self, key):
        logger.debug("websocket opened for key %s", key)
        self.key = key
        # openeing corresponding socket of model

        # open push socket to forward incoming zmq messages
        push = self.ctx.socket(zmq.PUSH)
        push_port = self.database[key]["ports"]['PUSH']
        node = "localhost"
        # node = self.database[key]["node"]

        push.connect("tcp://%s:%d" % (node, push_port))
        self.pushstream = ZMQStream(push)

        sub = self.ctx.socket(zmq.SUB)
        sub_port = self.database[key]["ports"]['SUB']
        sub.connect("tcp://%s:%d" % (node, sub_port))
        # Accept all messages
        sub.setsockopt(zmq.SUBSCRIBE, '')
        self.substream = ZMQStream(sub)

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
        logger.debug("got message %20s, text: %s", message, isinstance(message, six.text_type))

        # Let's try and forward it.
        # use the zmqstream as a socket (send, send_json)
        socket = self.pushstream

        # TODO: can we just forward the bytes without deserializing?
        if isinstance(message, six.text_type):
            metadata = json.loads(message)
            logger.debug("got metadata %s", metadata)
            if not ("dtype" in metadata):
                # no array expected, pass along:
                A = None
                logger.debug("sending metadata %s to %s of type %s", metadata, socket, SOCKET_NAMES[socket.socket.type])
                send_array(socket, None, metadata)
                self.metadata = None
            else:
                # We expect another message
                self.metadata = metadata
        else:
            # assume we already have metadata
            if not self.metadata:
                logger.warn("got message without preceding metadata")
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

    def check_origin(self, origin):
        """connect from everywhere"""
        return True


class MainHandler(tornado.web.RequestHandler):

    def initialize(self, database):
        self.database = database

    def get(self):
        self.set_header("Content-Type", "application/json")
        self.write(json.dumps(self.database))


class ModelHandler(tornado.web.RequestHandler):

    def initialize(self, database, ctx=None):
        self.database = database
        self.ctx = ctx

    def get(self, key=None, view=None):
        """Register a new model (models)"""
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Content-Type", "application/json")
        if key is not None:
            value = {}
            value.update(self.database[key])
            if view is not None:
                # generate a context with the relevant variables
                context = {}
                context["value"] = value
                context["ctx"] = self.ctx
                result = json.dumps(getattr(views, view)(context))
            else:
                result = json.dumps(value)
        else:
            result = json.dumps(self.database.values())

        self.write(result)

    def post(self):
        """Register a new model (models)"""
        self.set_header("Content-Type", "application/json")
        key = uuid.uuid4().hex
        metadata = json.loads(self.request.body.decode())
        metadata["uuid"] = key
        self.database[key] = metadata
        result = json.dumps({"uuid": key})
        self.write(result)

    def put(self, key, *args, **kwargs):
        # TODO: show a list of running models
        self.database[key] = json.loads(self.request.body)

    def delete(self, key, *args, **kwargs):
        del self.database[key]


def app():
    # register socket
    ctx = zmq.Context()
    # Use something more webscale when needed (like dbm)
    database = {}
    application = tornado.web.Application([
        (r"/", MainHandler, {"database": database}),
        # todo use an id scheme to attach to multiple models
        (r"/models/?", ModelHandler, {"database": database}),
        (r"/models/(\w+)/?", ModelHandler, {"database": database}),
        (r"/models/(?P<key>\w+)/(?P<view>\w*)/?", ModelHandler, {"database": database, "ctx": ctx}),
        (r"/mmi/(\w+)/?", WebSocket, {"database": database, "ctx": ctx}),
    ])
    # You reuse this app in other wsgi applications
    # Encapsulate it using a tornado.wsgi.WSGIAdapter
    # Won't work with websockets for now....
    # TODO some connection heartbeat that removes stale connections
    return application


def main():
    # Use common port not in services...
    # Normally you'd run this behind an nginx
    logger.info('mmi-tracker')
    application = app()
    application.listen(22222)
    logger.info('serving at port 22222')
    tornado.ioloop.IOLoop.instance().start()


if __name__ == "__main__":
    main()
