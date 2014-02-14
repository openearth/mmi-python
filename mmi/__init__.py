__version__ = '0.1'
import datetime

import numpy as np
import zmq

def send_array(socket, A, flags=0, copy=False, track=False, metadata=None):
    """send a numpy array with metadata"""
    md = dict(
        dtype=str(A.dtype),
        shape=A.shape,
        timestamp=datetime.datetime.now().isoformat()
    )
    if metadata:
        md.update(metadata)
    if A is None:
        # send only json
        socket.send_json(md, flags)
    else:
        # send json, followed by array
        socket.send_json(md, flags | zmq.SNDMORE)
    # Make a copy if required and pass along the buffer
    msg = buffer(np.ascontiguousarray(A))
    socket.send(msg, flags, copy=copy, track=track)
    return


def recv_array(socket, flags=0, copy=False, track=False):
    """recv a metadata and an optional numpy array"""
    md = socket.recv_json(flags=flags)
    if socket.getsockopt(zmq.RCVMORE):
        msg = socket.recv(flags=flags, copy=copy, track=track)
        buf = buffer(msg)
        A = np.frombuffer(buf, dtype=md['dtype'])
        A.reshape(md['shape'])
    else:
        # No array expected
        A = None
    return A, md
