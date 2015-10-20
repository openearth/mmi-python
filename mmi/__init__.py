"""
Model Message Interface
"""

__version__ = '0.1.6'
import datetime

import numpy as np
import zmq
import zlib


class NoResponseException(Exception):
    pass


class EmptyResponseException(Exception):
    pass


def send_array(
    socket, A=None, metadata=None, flags=0, copy=False, track=False, compress=None,
    chunksize=100000000):
    """send a numpy array with metadata over zmq

    message is mostly multipart:
    metadata | array part 1 | array part 2, etc

    only metadata:
    metadata
    """

    # create a metadata dictionary for the message
    md = {}
    # always add a timestamp
    md['timestamp'] = datetime.datetime.now().isoformat()

    # copy extra metadata
    if metadata:
        md.update(metadata)

    # if we don't have an array
    if A is None:
        # send only json
        md['parts'] = 0
        socket.send_json(md, flags)
        # and we're done
        return

    # add array metadata
    md['dtype'] = str(A.dtype)
    md['shape'] = A.shape
    # determine number of parts
    md['parts'] = np.prod(A.shape) / chunksize + 1
    try:
        # If an array has a fill value assume it's an array with missings
        # store the fill_Value in the metadata and fill the array before sending.
        # asscalar should work for scalar, 0d array or nd array of size 1
        md['fill_value'] = np.asscalar(A.fill_value)
        A = A.filled()
    except AttributeError:
        # no masked array, nothing to do
        pass

    # send json, followed by array (in x parts)
    socket.send_json(md, flags | zmq.SNDMORE)

    # split array at first dimension and send parts
    for i, a in enumerate(np.array_split(A, md['parts'])):
        # Make a copy if required and pass along the memoryview
        msg = memoryview(np.ascontiguousarray(a))
        flags_ = flags
        if i != md['parts'] - 1:
            flags_ |= zmq.SNDMORE
        socket.send(msg, flags_, copy=copy, track=track)
    return


def recv_array(
    socket, flags=0, copy=False, track=False, poll=None, poll_timeout=10000):
    """recv a metadata and an optional numpy array from a zmq socket

    Optionally provide poll object to use recv_array with timeout

    poll_timeout is in millis
    """
    if poll is None:
        md = socket.recv_json(flags=flags)
    else:
        # one-try "Lazy Pirate" method: http://zguide.zeromq.org/php:chapter4
        socks = dict(poll.poll(poll_timeout))
        if socks.get(socket) == zmq.POLLIN:
            reply = socket.recv_json(flags=flags)
            if not reply:
                raise EmptyResponseException(
                    "Recv_array got an empty response (1)")
            md = reply
        else:
            raise NoResponseException(
                "Recv_array got no response within timeout (1)")

    if md['parts'] == 0:
        # No array expected
        A = None

    else:
        A = np.zeros(0)
        for i in range(md['parts']):
            # if socket.getsockopt(zmq.RCVMORE):
            if poll is None:
                msg = socket.recv(flags=flags, copy=copy, track=track)
            else:
                # one-try "Lazy Pirate" method: http://zguide.zeromq.org/php:chapter4
                socks = dict(poll.poll(poll_timeout))
                if socks.get(socket) == zmq.POLLIN:
                    reply = socket.recv(flags=flags, copy=copy, track=track)
                    if not reply:
                        raise EmptyResponseException(
                            "Recv_array got an empty response (2)")
                    msg = reply
                else:
                    raise NoResponseException(
                        "Recv_array got no response within timeout (2)")
            buf = buffer(msg)
            a = np.frombuffer(buf, dtype=md['dtype'])
            A = np.concatenate((A, a))  # glue all parts together
        A.reshape(md['shape'])

        if 'fill_value' in md:
            A = np.ma.masked_equal(A, md['fill_value'])

    return A, md
