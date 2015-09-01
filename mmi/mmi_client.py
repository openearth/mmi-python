import zmq

from mmi import send_array, recv_array
from bmi.api import IBmi


class MMIClient(IBmi):
    def __init__(self, zmq_address, poll_timeout=10000, zmq_flags=0):
        """
        Constructor
        """

        # Open ZeroMQ socket
        context = zmq.Context()

        self.socket = context.socket(zmq.REQ)
        self.socket.connect(zmq_address)

        self.poll = zmq.Poller()
        self.poll.register(self.socket, zmq.POLLIN)

        self.poll_timeout = poll_timeout
        self.zmq_flags = zmq_flags

    def _close_sockets(self):
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.close()
        self.poll.unregister(self.socket)

    # from here: BMI commands that get translated to MMI.
    def initialize(self, configfile=None):
        """
        Initialize the module
        """

        method = "initialize"

        A = None
        metadata = {method: configfile}

        send_array(self.socket, A, metadata)
        A, metadata = recv_array(
            self.socket, poll=self.poll, poll_timeout=self.poll_timeout,
            flags=self.zmq_flags)

    def finalize(self):
        """
        Finalize the module
        """

        method = "finalize"

        A = None
        metadata = {method: -1}

        send_array(self.socket, A, metadata)
        A, metadata = recv_array(
            self.socket, poll=self.poll, poll_timeout=self.poll_timeout,
            flags=self.zmq_flags)

    def get_var_count(self):
        """
        Return number of variables
        """

        method = "get_var_count"

        A = None
        metadata = {method: -1}

        send_array(self.socket, A, metadata)
        A, metadata = recv_array(
            self.socket, poll=self.poll, poll_timeout=self.poll_timeout,
            flags=self.zmq_flags)

        return metadata[method]

    def get_var_name(self, i):
        """
        Return variable name
        """

        method = "get_var_name"

        A = None
        metadata = {method: i}

        send_array(self.socket, A, metadata)
        A, metadata = recv_array(
            self.socket, poll=self.poll, poll_timeout=self.poll_timeout,
            flags=self.zmq_flags)

        return metadata[method]

    def get_var_type(self, name):
        """
        Return variable name
        """

        method = "get_var_type"

        A = None
        metadata = {method: name}

        send_array(self.socket, A, metadata)
        A, metadata = recv_array(
            self.socket, poll=self.poll, poll_timeout=self.poll_timeout,
            flags=self.zmq_flags)

        return metadata[method]

    def get_var_rank(self, name):
        """
        Return variable rank
        """

        method = "get_var_rank"

        A = None
        metadata = {method: name}

        send_array(self.socket, A, metadata)
        A, metadata = recv_array(
            self.socket, poll=self.poll, poll_timeout=self.poll_timeout,
            flags=self.zmq_flags)

        return metadata[method]

    def get_var_shape(self, name):
        """
        Return variable shape
        """

        method = "get_var_shape"

        A = None
        metadata = {method: name}

        send_array(self.socket, A, metadata)
        A, metadata = recv_array(
            self.socket, poll=self.poll, poll_timeout=self.poll_timeout,
            flags=self.zmq_flags)

        return metadata[method]

    def get_var(self, name):
        """
        Return an nd array from model library
        """

        method = "get_var"

        A = None
        metadata = {method: name}

        send_array(self.socket, A, metadata)
        A, metadata = recv_array(
            self.socket, poll=self.poll, poll_timeout=self.poll_timeout,
            flags=self.zmq_flags)

        return A

    def set_var(self, name, var):
        """
        Set the variable name with the values of var
        """

        method = "set_var"

        A = var
        metadata = {method: name}

        send_array(self.socket, A, metadata)
        A, metadata = recv_array(
            self.socket, poll=self.poll, poll_timeout=self.poll_timeout,
            flags=self.zmq_flags)

    def get_start_time(self):
        """
        Return start time
        """

        method = "get_start_time"

        A = None
        metadata = {method: -1}

        send_array(self.socket, A, metadata)
        A, metadata = recv_array(
            self.socket, poll=self.poll, poll_timeout=self.poll_timeout,
            flags=self.zmq_flags)

        return metadata[method]

    def get_end_time(self):
        """
        Return end time of simulation
        """

        method = "get_end_time"

        A = None
        metadata = {method: -1}

        send_array(self.socket, A, metadata)
        A, metadata = recv_array(
            self.socket, poll=self.poll, poll_timeout=self.poll_timeout,
            flags=self.zmq_flags)

        return metadata[method]

    def get_current_time(self):
        """
        Return current time of simulation
        """

        method = "get_current_time"

        A = None
        metadata = {method: -1}

        send_array(self.socket, A, metadata)
        A, metadata = recv_array(
            self.socket, poll=self.poll, poll_timeout=self.poll_timeout,
            flags=self.zmq_flags)

        return metadata[method]

    def update(self, dt):
        """
        Advance the module with timestep dt
        """

        method = "update"

        A = None
        metadata = {method: dt}

        send_array(self.socket, A, metadata)
        A, metadata = recv_array(
            self.socket, poll=self.poll, poll_timeout=self.poll_timeout,
            flags=self.zmq_flags)

    # TODO:  Do we really need these two?
    def inq_compound(self, name):
        """
        Return the number of fields of a compound type.
        """
        pass

    def inq_compound_field(self, name, index):
        """
        Lookup the type,rank and shape of a compound field
        """
        pass

    def remote(self, action):
        """
        Function specific for MMI, not BMI.
        action is one of: "play", "stop", "pause", "rewind", "quit"
        """
        method = "remote"

        A = None
        metadata = {method: action}

        send_array(self.socket, A, metadata)
        A, metadata = recv_array(
            self.socket, poll=self.poll, poll_timeout=self.poll_timeout,
            flags=self.zmq_flags)

        return metadata[method]
