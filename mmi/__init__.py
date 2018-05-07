"""
Model Message Interface
"""

import datetime
import sys

import numpy as np
import zmq


__version__ = '0.5.0'


if sys.version_info > (3, ):
    buffer = memoryview


class NoResponseException(Exception):
    pass


class EmptyResponseException(Exception):
    pass


def send_array(
        socket, A=None, metadata=None, flags=0,
        copy=False, track=False, compress=None,
        chunksize=50 * 1000 * 1000
):
    """send a numpy array with metadata over zmq

    message is mostly multipart:
    metadata | array part 1 | array part 2, etc

    only metadata:
    metadata

    the chunksize roughly determines the size of the parts being sent
    if the chunksize is too big, you get an error like:
        zmq.error.Again: Resource temporarily unavailable
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

    # support single values (empty shape)
    if isinstance(A, float) or isinstance(A, int):
        A = np.asarray(A)

    # add array metadata
    md['dtype'] = str(A.dtype)
    md['shape'] = A.shape
    # determine number of parts
    md['parts'] = int(np.prod(A.shape) // chunksize + 1)
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

    # although the check is not strictly necessary, we try to maintain fast
    # pointer transfer when there is only 1 part
    if md['parts'] == 1:
        msg = memoryview(np.ascontiguousarray(A))
        socket.send(msg, flags, copy=copy, track=track)
    else:
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
        socket,
        flags=0,
        copy=False,
        track=False,
        poll=None,
        poll_timeout=10000
):
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
            # note that reply can be an empty array
            md = reply
        else:
            raise NoResponseException(
                "Recv_array got no response within timeout (1)")

    if md['parts'] == 0:
        # No array expected
        A = None
    elif md['parts'] == 1:
        # although the check is not strictly necessary, we try to maintain fast
        # pointer transfer when there is only 1 part

        if poll is None:
            msg = socket.recv(flags=flags, copy=copy, track=track)
        else:
            # one-try "Lazy Pirate" method: http://zguide.zeromq.org/php:chapter4
            socks = dict(poll.poll(poll_timeout))
            if socks.get(socket) == zmq.POLLIN:
                reply = socket.recv(flags=flags, copy=copy, track=track)
                # note that reply can be an empty array
                msg = reply
            else:
                raise NoResponseException(
                    "Recv_array got no response within timeout (2)")
        buf = buffer(msg)
        A = np.frombuffer(buf, dtype=md['dtype'])
        A = A.reshape(md['shape'])

        if 'fill_value' in md:
            A = np.ma.masked_equal(A, md['fill_value'])

    else:
        # multi part array
        A = np.zeros(np.prod(md['shape']), dtype=md['dtype'])
        arr_position = 0
        for i in range(md['parts']):
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
            A[arr_position:arr_position + a.shape[0]] = a[:]
            arr_position += a.shape[0]
        A = A.reshape(md['shape'])

        if 'fill_value' in md:
            A = np.ma.masked_equal(A, md['fill_value'])

    return A, md
