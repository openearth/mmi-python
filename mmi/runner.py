#!/usr/bin/env python
"""
Usage:
  mmi-runner <engine> <configfile> [-o <outputvar>...] [-g <globalvar>...] [--interval <interval>] [--disable-logger] [--pause] [--mpi <method>] [--track <server>] [--port <port>]
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
  --pause                  start in paused mode, send update messages to progress
  --mpi <method>           communicate with mpi nodes using one of the methods: root (communicate with rank 0), all (one socket per rank)
  --port <port>            base port, port is computed as req/rep = port + rank*3 + 0, push/pull = port + rank*3 + 1, pub/sub = port + rank*3 + 2 [default: 5600]
  --track <tracker>        server to subscribe to for tracking

"""
import os
import logging
import json
import urlparse
import datetime
import logging
import itertools
import argparse
import atexit

import docopt
import requests
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


# see or an in memory numpy message:
# http://zeromq.github.io/pyzmq/serialization.html
def register(server, metadata):
    """register model at tracking server"""
    logger.info("register at server %s with %s", server, metadata)
    # connect to server
    result = requests.post(urlparse.urljoin(server, 'models'), data=json.dumps(metadata))
    logger.info("got back %s", result.json())
    metadata["tracker"] = result.json()

def unregister(server, metadata):
    """unregister model at tracking server"""
    uuid = metadata["tracker"]["uuid"]
    # connect to server
    result = requests.delete(urlparse.urljoin(server, 'models' + "/" + uuid))
    logger.info("unregistered at server %s with %s: %s", server, uuid, result)

def process_incoming(model, sockets, data):
    """
    process incoming messages

    data is a dict with several arrays
    """
    # Check for new messages
    if not sockets:
        return
    # unpack sockets
    poller = sockets['poller']
    rep = sockets['rep']
    pull = sockets['pull']
    pub = sockets['pub']
    items = poller.poll(10)
    for sock, n in items:
        for i in range(n):
            A, metadata = recv_array(sock)
            logger.debug("got metadata: %s", metadata)
            var = None
            # bmi actions
            if "update" in metadata:
                dt = float(metadata["update"])
                logger.debug("updating with dt %s", dt)
                model.update(dt)
                metadata["dt"] = dt
            elif "get_var" in metadata:
                name = metadata["get_var"]
                logger.debug("sending variable %s", name)
                # temporary implementation
                var = model.get_var(name)
                metadata['name'] = name
                # assert socket is req socket

            elif "get_var_count" in metadata:
                # temporary implementation
                n = model.get_var_count()
                metadata['get_var_count'] = n
                # assert socket is req socket
            elif "get_var_name" in metadata:
                i = int(metadata["get_var_name"])
                name = model.get_var_name(i)
                metadata['get_var_name'] = name
                # assert socket is req socket
            elif "set_var" in metadata:
                name = metadata["set_var"]
                logger.debug("setting variable %s", name)
                model.set_var(name, A)
                metadata["name"] = name  # !?
            elif "set_var_slice" in metadata:
                name = metadata["set_var_slice"]
                logger.debug("setting variable %s", name)
                start = metadata["start"]
                count = metadata["count"]
                model.set_var_slice(name, start, count, A)
                metadata["name"] = name  # !?
            elif "set_var_index" in metadata:
                name = metadata["set_var_index"]
                logger.debug("setting variable %s using index index", name)
                index = metadata["index"]
                # TODO: test if this is fast enough.
                # Otherwise move to BMI ++ but that is
                # a bit of a burden on implementers
                var = model.get_var(name).copy()
                var.flat[index] = A
                model.set_var(name, var)
                metadata["name"] = name  # !?
            elif "get_current_time" in metadata:
                metadata["get_current_time"]
                t = model.get_current_time()
                metadata['get_current_time'] = t
            elif "get_time_step" in metadata:
                metadata["get_time_step"]
                dt = model.get_time_step()
                metadata['get_time_step'] = dt
            elif "get_end_time" in metadata:
                metadata["get_end_time"]
                t = model.get_end_time()
                metadata['get_end_time'] = t
            elif "get_start_time" in metadata:
                metadata["get_start_time"]
                t = model.get_start_time()
                metadata['get_start_time'] = t
                # assert socket is req socket
            # custom actions
            elif "remote" in metadata:
                assert metadata["remote"] in {"play", "stop", "pause", "rewind"}
                model.state = metadata["remote"]
            elif "operator" in metadata:
                # TODO: support same operators as MPI_ops here....,
                # TODO: reduce before apply
                # TODO: assert pull socket
                S = tuple(slice(*x) for x in action['slice'])
                print(repr(arr[S]))
                if action['operator'] == 'setitem':
                    arr[S] = data
                elif action['operator'] == 'add':
                    arr[S] += data
            else:
                logger.warn("got message from unknown socket {}".format(sock))
            if sock.socket_type == zmq.REP:
                # reply
                send_array(rep, var, metadata=metadata)
            # any getter requested through the pull socket?
            elif any(x.startswith("get_") for x in metadata) and sock.socket_type == zmq.PULL:
                # return through the pub socket
                send_array(pub, var, metadata=metadata)



def create_sockets(ports):
    context = zmq.Context()
    poller = zmq.Poller()

    # Socket to handle init data
    rep = context.socket(zmq.REP)
    rep.bind(
        "tcp://*:{port}".format(port=ports["REQ"])
    )
    pull = context.socket(zmq.PULL)
    pull.bind(
        "tcp://*:{port}".format(port=ports["PULL"])
    )
    # for sending model messages
    pub = context.socket(zmq.PUB)
    pub.bind(
        "tcp://*:{port}".format(port=ports["PUB"])
    )

    poller.register(rep, zmq.POLLIN)
    poller.register(pull, zmq.POLLIN)
    sockets = dict(
        poller=poller,
        rep=rep,
        pull=pull,
        pub=pub
    )
    return sockets

def main():
    arguments = docopt.docopt(__doc__)
    paused = arguments['--pause']
    logger.info(arguments)
    # make a socket that replies to message with the grid


    # if we are running mpi we want to know the rank
    if arguments['--mpi']:
        import mpi4py.MPI
        comm = mpi4py.MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        # or we assume it's 0
        rank = 0
        size = 1

    port = int(arguments['--port'])
    ports = {
        "REQ": port + 0,
        "PULL": port + 1,
        "PUB": port + 2
    }


    # if we want to communicate with separate domains
    # we have to setup a socket for each of them
    if arguments['--mpi'] == 'all':
        # use a socket for each rank rank
        for port in ports:
            ports[port] += (rank * 3)

    # now we can create the zmq sockets and poller
    sockets = {}
    if rank == 0 or arguments['--mpi'] == 'all':
        # Create sockets
        sockets = create_sockets(ports)

    # not so verbose
    # bmi.wrapper.logger.setLevel(logging.WARN)

    wrapper = bmi.wrapper.BMIWrapper(engine=arguments['<engine>'],
                                     configfile=arguments['<configfile>'])
    # for replying to grid requests
    with wrapper as model:
        model.state = "play"
        if arguments["--pause"]:
            model.state = "pause"
            logger.info("model initialized and started in pause mode, wating for requests ...")
        else:
            logger.info("model started and initialized, running ...")

        if arguments["--track"]:
            server = arguments["--track"]

            metadata = {
                "engine": arguments['<engine>'],
                "configfile": arguments['<configfile>'],
                "ports": ports,
                "rank": rank,
                "size": size
            }
            metadata_filename = os.path.join(
                os.path.dirname(arguments['<configfile>']),
                "metadata.json"
            )
            if os.path.isfile(metadata_filename):
                metadata.update(json.load(open(metadata_filename)))

            register(server, metadata)

        if arguments["--track"]:
            atexit.register(unregister, server, metadata)
        # Start a reply process in the background, with variables available
        # after initialization, sent all at once as py_obj
        data = {
            var: model.get_var(var)
            for var
            in arguments['-g']
        }
        process_incoming(model, sockets, data)

        # Keep on counting indefinitely
        counter = itertools.count()

        for i in counter:

            while model.state == "pause":
                # keep waiting for messages when paused
                process_incoming(model, sockets, data)
            else:
                # otherwise process messages once and continue
                process_incoming(model, sockets, data)
            logger.debug("i %s", i)

            # paused ...
            model.update(-1)

            # check counter
            if arguments.get('--interval') and (i % arguments['--interval']):
                continue

            for key in arguments['-o']:
                value = model.get_var(key)
                metadata = {'name': key, 'iteration': i}
                # 4ms for 1M doubles
                logger.debug("sending {}".format(metadata))
                if 'pub' in sockets:
                    send_array(sockets['pub'], value, metadata=metadata)

if __name__ == '__main__':

    main()

