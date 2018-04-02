#!/usr/bin/env python
"""
Usage:
  mmi-curl <url> [--request=<command>] [--metadata=<json>]
  mmi-curl -h | --help

Positional arguments:
  url  connection string used for sending/receiving messages

Optional arguments:
  -h, --help               show this help message and exit
  -d --metadata=<json>     send a message with metadata [default: {}]
  -X --request=<command>   use the following method of connection (REQ|SUB|PUSH) [default: SUB]

"""
import logging
import json

import zmq
import zmq.eventloop.zmqstream
import docopt

from . import send_array, recv_array

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main():
    arguments = docopt.docopt(__doc__)
    logger.info(arguments)
    context = zmq.Context()
    metadata = json.loads(arguments['--metadata'])
    url = arguments['<url>']
    if arguments['--request'] == 'SUB':
        sub = context.socket(zmq.SUB)
        sub.connect(url)
        sub.setsockopt(zmq.SUBSCRIBE, '')
        while True:
            arr, metadata = recv_array(sub)
            logger.info("metadata: %s\n array: %s", metadata, arr)
    elif arguments['--request'] == 'REQ':
        req = context.socket(zmq.REQ)
        req.connect(url)
        value = None
        send_array(req, value, metadata=metadata)
        # wait for reply
        arr, metadata = recv_array(req)
        logger.info("metadata: %s\n array: %s", metadata, arr)
    elif arguments['--request'] == 'PUSH':
        req = context.socket(zmq.PUSH)
        req.connect(url)
        value = None
        send_array(req, value, metadata=metadata)
        logger.info("metadata: %s", metadata)
