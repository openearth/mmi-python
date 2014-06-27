#!/usr/bin/env python
"""
Usage:
  mmi-runner <engine> <configfile> [-o <outputvar>...] [-g <globalvar>...] [--interval <interval>] [--disable-logger]
  mmi-runner -h | --help

Positional arguments:
  engine model  engine
  configfile    model configuration file

Optional arguments:
  -h, --help               show this help message and exit
  --interval <interval>    publish results every <interval> timesteps
  -o <outputvar>           output variables, will be broadcasted each <interval> timestep
  -g <globalvar>           global variables, will be send if requested
  --disable-logger         do not inject logger into the BMI library

"""

import logging

import datetime
import logging
import itertools
import argparse

import docopt

import zmq
import zmq.eventloop.zmqstream
from zmq.eventloop import ioloop
import numpy as np

import bmi.wrapper
from mmi import send_array, recv_array

logging.basicConfig()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ioloop.install()

OUTPUTVARS=[]
INITVARS=[]
def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '-n', '--interval', dest='interval',
        help='publishing results every [n] tiemsteps',
        type=int,
        default=1)
    argparser.add_argument(
        '-o', '--outputvariables', dest='outputvariables',
        metavar='O',
        nargs='*',
        help='variables to be published',
        default=OUTPUTVARS)
    argparser.add_argument(
        '-g', '--global', dest='globalvariables',
        metavar='G',
        nargs='*',
        help='variables that can be send back to a reply (not changed during run)',
        default=INITVARS)
    argparser.add_argument(
        'ini',
        help='model configuration file')
    return argparser.parse_args()


# see or an in memory numpy message:
# http://zeromq.github.io/pyzmq/serialization.html


def process_incoming(model, poller, rep, pull, data):
    """
    process incoming messages

    data is a dict with several arrays
    """
    # Check for new messages
    items = poller.poll(100)
    for sock, n in items:
        for i in range(n):
            A, metadata = recv_array(sock)
            logger.info("got metadata: %s", metadata)
            if metadata.get("action") == "send grid":
                logger.info("sending grid")
                # temporary implementation
                sock.send_pyobj(data)
            elif "action" in metadata:
                logger.info("found action applying update")
                # TODO: support same operators as MPI_ops here....,
                # TODO: reduce before apply
                action = metadata['action']
                arr = model.get_var(metadata['name'])
                S = tuple(slice(*x) for x in action['slice'])
                print(repr(arr[S]))
                if action['operator'] == 'setitem':
                    arr[S] = data
                elif action['operator'] == 'add':
                    arr[S] += data

            else:
                logger.warn("got message from unknown socket {}".format(sock))
    else:
        logger.info("No incoming data")

def main():
    arguments = docopt.docopt(__doc__)

    logger.info(arguments)
    # make a socket that replies to message with the grid

    # You probably want to read:
    # http://zguide.zeromq.org/page:all

    context = zmq.Context()
    # Socket to handle init data
    rep = context.socket(zmq.REP)
    rep.bind(
        "tcp://*:{port}".format(port=5556)
    )
    pull = context.socket(zmq.PULL)
    pull.connect(
        "tcp://localhost:{port}".format(port=5557)
    )
    # for sending model messages
    pub = context.socket(zmq.PUB)
    pub.bind(
        "tcp://*:{port}".format(port=5558)
    )

    poller = zmq.Poller()
    poller.register(rep, zmq.POLLIN)
    poller.register(pull, zmq.POLLIN)

    bmi.wrapper.logger.setLevel(logging.WARN)

    # for replying to grid requests
    with bmi.wrapper.BMIWrapper(engine=arguments['<engine>'],
                                configfile=arguments['<configfile>']) as model:
        model.initialize()

        # Start a reply process in the background, with variables available
        # after initialization, sent all at once as py_obj
        data = {
            var: model.get_var(var)
            for var
            in arguments['-g']
        }
        process_incoming(model, poller, rep, pull, data)

        # Keep on counting indefinitely
        counter = itertools.count()

        for i in counter:

            process_incoming(model, poller, rep, pull, data)

            # Calculate
            model.update(-1)

            # check counter

            if arguments.get('--interval') and (i % arguments['--interval']):
                continue

            for key in arguments['-o']:
                value = model.get_var(key)
                metadata = {'name': key, 'iteration': i}
                # 4ms for 1M doubles
                logger.info("sending {}".format(metadata))
                send_array(pub, value, metadata=metadata)

if __name__ == '__main__':
    main()

