# test server
import zmq
import numpy as np
import time
from mmi import send_array

if __name__ == '__main__':
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.connect('tcp://localhost:3666')

    print('start...')
    size = 1024
    while 1:
        print('sending [size=%d x 10000]...' % size)
        send_array(socket, np.zeros((size, 10000), dtype=np.int8))
        size *= 2
        time.sleep(1)
