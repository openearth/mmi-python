import json

import tornado.ioloop
import tornado.web
import tornado.websocket

import logging
import numpy as np
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world")

class ArrayWebSocket(tornado.websocket.WebSocketHandler):
    """Message loader"""
    def open(self):
        logger.debug("opening socket")
        self.header = None
    def on_message(self, message):
        header =self.header
        if header is None:
            logger.debug("got msg {}".format(message))
            self.header = json.loads(message)
            self.write_message(u"Got msg: " + message)
        else:
            logger.debug("got msg {}".format(message))
            arr = np.frombuffer(message, dtype=header["dtype"]).reshape(header["shape"])
            logger.debug("got arr {}".format(arr))
            self.header = None
    def on_close(self):
        print "WebSocket closed"

application = tornado.web.Application([
    (r"/", MainHandler),
    (r"/websocket", ArrayWebSocket),
    (r"/(.*)", tornado.web.StaticFileHandler, {"path": "."})
])



if __name__ == "__main__":
    application.listen(6011)
    tornado.ioloop.IOLoop.instance().start()
