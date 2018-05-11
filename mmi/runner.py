import importlib
import os
import logging
import json
from six.moves.urllib.parse import urljoin
import itertools
import atexit
import platform

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


class Runner(object):
    def __init__(
            self,
            engine,
            configfile,
            output_vars=None,
            interval=1,
            tracker=None,
            mpi=False,
            port='random',
            bmi_class='bmi.wrapper.BMIWrapper',
            pause=False,
            *args,
            **kwargs
    ):
        # store options
        self.engine = engine
        self.configfile = configfile
        if output_vars is None:
            self.output_vars = []
        else:
            self.output_vars = output_vars
        self.interval = interval
        self.tracker = tracker
        self.mpi = mpi
        self.port = port
        self.bmi_class = bmi_class
        self.args = args
        self.kwargs = kwargs

        # load mpi
        self.mpi_info = self.initialize_mpi(mpi)
        self.ports = self.create_ports(port, mpi=mpi, rank=self.mpi_info['rank'])
        self.sockets = self.create_sockets()

        self.model = self.create_bmi_model(engine, bmi_class)
        if pause:
            self.model.state = 'pause'
        else:
            # default to play if state is not present
            if not hasattr(self.model, 'state'):
                self.model.state = 'play'
        self.metadata = {}
        self.fill_metadata()

    @staticmethod
    def initialize_mpi(mpi=False):
        """initialize mpi settings"""
        if mpi:
            import mpi4py.MPI
            comm = mpi4py.MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
        else:
            comm = None
            rank = 0
            size = 1
        return {
            "comm": comm,
            "rank": rank,
            "size": size,
            "mode": mpi
        }

    @staticmethod
    def create_ports(port, mpi, rank):
        """create a list of ports for the current rank"""
        if port == "random" or port is None:
            # ports will be filled in using random binding
            ports = {}
        else:
            port = int(port)
            ports = {
                "REQ": port + 0,
                "PUSH": port + 1,
                "SUB": port + 2
            }
        # if we want to communicate with separate domains
        # we have to setup a socket for each of them
        if mpi == 'all':
            # use a socket for each rank rank
            for port in ports:
                ports[port] += (rank * 3)
        return ports

    @staticmethod
    def import_from_string(full_class_name):
        """return a class based on it's full class name"""
        s = full_class_name.split('.')
        class_name = s[-1]
        module_name = full_class_name[:-len(class_name)-1]
        module = importlib.import_module(module_name)
        # the class, it's common to spell with k as class is reserved
        klass = getattr(module, class_name)
        return klass

    def create_bmi_model(self, engine, bmi_class=None, wrapper_kwargs=None):
        """initialize a bmi mode using an optional class"""
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        if bmi_class is None:
            wrapper_class = bmi.wrapper.BMIWrapper
        else:
            wrapper_class = self.import_from_string(bmi_class)
        try:
            """most models use engine as a first argument"""
            model = wrapper_class(
                engine,
                **wrapper_kwargs
            )
        except TypeError as e:
            """but old python engines are engines, so they don't, but they should """
            logger.warn(
                'Model wrapper %s does not accept engine as a first argument',
                wrapper_class
            )
            model = wrapper_class(
                **wrapper_kwargs
            )
        return model

    def register(self):
        """register model at tracking server"""
        # connect to tracker
        result = requests.post(urljoin(self.tracker, 'models'), data=json.dumps(self.metadata))
        logger.debug("registered at server %s: %s", self.tracker, result)
        self.metadata["tracker"] = result.json()

    def unregister(self):
        """unregister model at tracking server"""
        uuid = self.metadata["tracker"]["uuid"]
        # connect to server
        result = requests.delete(urljoin(self.tracker, 'models' + "/" + uuid))
        logger.debug("unregistered at server %s with %s: %s", self.tracker, uuid, result)

    def fill_metadata(self):
        self.metadata["node"] = platform.node()
        self.metadata.update({
            "engine": self.engine,
            "configfile": self.configfile,
            "ports": self.ports,
            "mpi": {
                "rank": self.mpi_info['rank'],
                "size": self.mpi_info['size']
            }
        })
        metadata_filename = os.path.join(
            os.path.dirname(self.configfile),
            "metadata.json"
        )
        if os.path.isfile(metadata_filename):
            self.metadata.update(json.load(open(metadata_filename)))
        # update connection information from external service
        # You might want to disable this if you have some sort of sense of privacy
        try:
            self.metadata["ifconfig"] = requests.get("http://ipinfo.io/json").json()
        except IOError:
            pass

    def process_incoming(self):
        """
        process incoming messages

        data is a dict with several arrays
        """

        model = self.model
        sockets = self.sockets

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
                elif "set_current_time" in metadata:
                    t = float(metadata["set_current_time"])
                    model.set_current_time(t)
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

    def create_sockets(self):
        """create zmq sockets"""

        ports = self.ports

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

    def run(self):
        """run the model"""

        model = self.model
        configfile = self.configfile
        interval = self.interval
        sockets = self.sockets

        model.initialize(configfile)
        if model.state == 'pause':
            logger.info(
                "model initialized and started in pause mode, waiting for requests"
            )
        else:
            logger.info("model started and initialized, running")

        if self.tracker:
            self.register()
            atexit.register(self.unregister)

        self.process_incoming()

        # Keep on counting indefinitely
        counter = itertools.count()
        logger.info("Entering timeloop...")
        for i in counter:
            while model.state == "pause":
                # keep waiting for messages when paused
                # process_incoming should set model.state to play
                self.process_incoming()
            else:
                # otherwise process messages once and continue
                self.process_incoming()
            if model.state == "quit":
                break

            # lookup dt or use -1 (default)
            dt = model.get_time_step() or -1
            model.update(dt)

            # check counter, if not a multiple of interval, skip this step
            if i % interval:
                continue

            for key in self.output_vars:
                value = model.get_var(key)
                metadata = {'name': key, 'iteration': i}
                # 4ms for 1M doubles
                logger.debug("sending {}".format(metadata))
                if 'pub' in sockets:
                    send_array(sockets['pub'], value, metadata=metadata)

        logger.info("Finalizing...")
        model.finalize()
