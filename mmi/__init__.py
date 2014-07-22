"""
Model Message Interface
"""

__version__ = '0.1.4'
import datetime

import numpy as np
import zmq
import zlib

def send_array(socket, A=None, metadata=None, flags=0, copy=False, track=False, compress=None):
    """send a numpy array with metadata over zmq"""

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
        socket.send_json(md, flags)
        # and we're done
        return

    # add array metadata
    md['dtype'] = str(A.dtype)
    md['shape'] = A.shape
    try:
        # If an array has a fill value assume it's an array with missings
        # store the fill_Value in the metadata and fill the array before sending.
        # asscalar should work for scalar, 0d array or nd array of size 1
        md['fill_value'] = np.asscalar(A.fill_value)
        A = A.filled()
    except AttributeError:
        # no masked array, nothing to do
        pass

    # send json, followed by array
    socket.send_json(md, flags | zmq.SNDMORE)
    # Make a copy if required and pass along the memoryview
    msg = memoryview(np.ascontiguousarray(A))
    socket.send(msg, flags, copy=copy, track=track)
    return


def recv_array(socket, flags=0, copy=False, track=False):
    """recv a metadata and an optional numpy array from a zmq socket"""
    md = socket.recv_json(flags=flags)
    if socket.getsockopt(zmq.RCVMORE):
        msg = socket.recv(flags=flags, copy=copy, track=track)
        buf = buffer(msg)
        A = np.frombuffer(buf, dtype=md['dtype'])
        A = A.reshape(md['shape'])
        if 'fill_value' in md:
            A = np.ma.masked_equal(A, md['fill_value'])
    else:
        # No array expected
        A = None
    return A, md


