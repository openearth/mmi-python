#!/usr/bin/env python
"""
Usage:
  mmi-runner <engine> <configfile> [-o <outputvar>...] [-g <globalvar>...] [--interval <interval>] [--disable-logger] [--pause] [--mpi <method>] [--track <server>] [--port <port>] [--bmi-class]
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
  --port <port>            "random" or integer base port, port is computed as req/rep = port + rank*3 + 0, push/pull = port + rank*3 + 1, pub/sub = port + rank*3 + 2 [default: random]
  --track <tracker>        server to subscribe to for tracking
  --bmi-class              when used - the engine is assumed to be the full name of a Python class that implements bmi [default: bmi.wrapper.BMIWrapper]

"""
import os
import logging
import json
from six.moves.urllib.parse import urljoin
import itertools
import atexit
import platform

import docopt
import requests
import zmq
import zmq.eventloop.zmqstream
from zmq.eventloop import ioloop


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
    logger.debug("register at server %s with %s", server, metadata)
    # connect to server
    result = requests.post(urljoin(server, 'models'), data=json.dumps(metadata))
    logger.debug("got back %s", result.json())
    metadata["tracker"] = result.json()


def unregister(server, metadata):
    """unregister model at tracking server"""
    uuid = metadata["tracker"]["uuid"]
    # connect to server
    result = requests.delete(urljoin(server, 'models' + "/" + uuid))
    logger.debug("unregistered at server %s with %s: %s", server, uuid, result)


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
    # pull = sockets['pull']
    pub = sockets['pub']
    items = poller.poll(10)
    for sock, n in items:
        for i in range(n):

            A, metadata = recv_array(sock)
            var = None

            # bmi actions
            if "update" in metadata:
                dt = float(metadata["update"])
                model.update(dt)
                metadata["dt"] = dt
            elif "get_var" in metadata:
                name = metadata["get_var"]
                # temporary implementation
                if metadata.get("copy", False):
                    var = model.get_var(name).copy()
                else:
                    var = model.get_var(name)
                if var is None:
                    logger.warning("Get_var returns None for %s" % name)
                metadata['name'] = name
                # assert socket is req socket
            elif "get_var_count" in metadata:
                # temporary implementation
                n = model.get_var_count()
                metadata['get_var_count'] = n
                # assert socket is req socket
            elif "get_var_rank" in metadata:
                # temporary implementation
                var_name = metadata['get_var_rank']
                n = model.get_var_rank(var_name)
                metadata['get_var_rank'] = n
                # assert socket is req socket
            elif "get_var_shape" in metadata:
                # temporary implementation
                var_name = metadata['get_var_shape']
                n = model.get_var_shape(var_name)
                metadata['get_var_shape'] = tuple([int(item) for item in n])
                # assert socket is req socket
            elif "get_var_type" in metadata:
                # temporary implementation
                var_name = metadata['get_var_type']
                n = model.get_var_type(var_name)
                metadata['get_var_type'] = n
                # assert socket is req socket
            elif "get_var_name" in metadata:
                i = int(metadata["get_var_name"])
                name = model.get_var_name(i)
                metadata['get_var_name'] = name
                # assert socket is req socket
            elif "set_var" in metadata:
                name = metadata["set_var"]
                # logger.debug("setting variable %s", name)
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
                assert metadata["remote"] in {
                    "play", "stop", "pause", "rewind", "quit"}
                model.state = metadata["remote"]
            elif "operator" in metadata:
                # TODO: support same operators as MPI_ops here....,
                # TODO: reduce before apply
                # TODO: assert pull socket
                pass
                # S = tuple(slice(*x) for x in action['slice'])
                # print(repr(arr[S]))
                # if action['operator'] == 'setitem':
                #     arr[S] = data
                # elif action['operator'] == 'add':
                #     arr[S] += data
            elif "initialize" in metadata:
                config_file = metadata["initialize"]
                model.initialize(str(config_file))
            elif "finalize" in metadata:
                model.finalize()
            else:
                logger.warn("got unknown message {} from socket {}".format(str(metadata), sock))
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
    # this was inconsequent: here REQ is for the client, we reply with REP.
    # PULL and PUB is seen from here, not from the client.
    # Is now renamed to PUSH and SUB: everything is seen from outside.
    if "REQ" in ports:
        rep.bind(
            "tcp://*:{port}".format(port=ports["REQ"])
        )
    else:
        ports["REQ"] = rep.bind_to_random_port(
            "tcp://*"
        )

    pull = context.socket(zmq.PULL)
    if "PUSH" in ports:
        pull.bind(
            "tcp://*:{port}".format(port=ports["PUSH"])
        )
    else:
        ports["PUSH"] = pull.bind_to_random_port(
            "tcp://*"
        )

    # for sending model messages
    pub = context.socket(zmq.PUB)
    if "SUB" in ports:
        pub.bind(
            "tcp://*:{port}".format(port=ports["SUB"])
        )
    else:
        ports["SUB"] = pub.bind_to_random_port(
            "tcp://*"
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


def runner(arguments, wrapper_kwargs={}, extra_metadata={}):
    """
    MMI Runner
    """
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

    if arguments["--port"] == "random":
        # ports will be filled in
        ports = {}
    else:
        port = int(arguments['--port'])
        ports = {
            "REQ": port + 0,
            "PUSH": port + 1,
            "SUB": port + 2
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
        logger.debug("ZMQ MMI Ports:")
        for key, value in ports.items():
            logger.debug("%s = %d" % (key, value))

    # not so verbose
    # bmi.wrapper.logger.setLevel(logging.WARN)

    if arguments.get('--bmi-class'):
        full_name = arguments['<engine>']
        s = full_name.split('.')
        class_name = s[-1]
        module_name = full_name[:-len(class_name)-1]
        import importlib
        wrapper_module = importlib.import_module(module_name)
        wrapper_class = getattr(wrapper_module, class_name)
        model = wrapper_class()

    else:
        wrapper_class = bmi.wrapper.BMIWrapper
        model = wrapper_class(
            engine=arguments['<engine>'],
            configfile=arguments['<configfile>'],
            **wrapper_kwargs)

    # for replying to grid requests
    model.state = "play"
    if arguments["--pause"]:
        model.state = "pause"
        logger.info("model initialized and started in pause mode, waiting for requests ...")
    else:
        logger.info("model started and initialized, running ...")

    if arguments["--track"]:
        server = arguments["--track"]

        metadata = {}
        # update connection information from external service
        # You might want to disable this if you have some sort of sense of privacy
        try:
            metadata["ifconfig"] = requests.get("http://ipinfo.io/json").json()
        except:
            pass
        # except requests.exceptions.ConnectionError:
        #     logger.exception("Could not read ip info from ipinfo.io")
        #     pass
        # except simplejson.scanner.JSONDecodeError:
        #     logger.exception("Could not parse ip info from ipinfo.io")
        #     pass
        # node
        metadata["node"] = platform.node()
        metadata.update({
            "engine": arguments['<engine>'],
            "configfile": arguments['<configfile>'],
            "ports": ports,
            "rank": rank,
            "size": size
        })

        metadata_filename = os.path.join(
            os.path.dirname(arguments['<configfile>']),
            "metadata.json"
        )
        if os.path.isfile(metadata_filename):
            metadata.update(json.load(open(metadata_filename)))
        metadata.update(extra_metadata)

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
    logger.info("Entering timeloop...")
    for i in counter:
        while model.state == "pause":
            # keep waiting for messages when paused
            process_incoming(model, sockets, data)
        else:
            # otherwise process messages once and continue
            process_incoming(model, sockets, data)
        if model.state == "quit":
            break

        # paused ...
        model.update(-1)

        # check counter
        if arguments.get('--interval') and (i % int(arguments['--interval'])):
            continue

        for key in arguments['-o']:
            value = model.get_var(key)
            metadata = {'name': key, 'iteration': i}
            # 4ms for 1M doubles
            logger.debug("sending {}".format(metadata))
            if 'pub' in sockets:
                send_array(sockets['pub'], value, metadata=metadata)

    logger.info("Finalizing...")
    model.finalize()

def main():
    arguments = docopt.docopt(__doc__)
    runner(arguments)


if __name__ == '__main__':
    logging.basicConfig()

    logger.info("mmi-runner")
    main()
