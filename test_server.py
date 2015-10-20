# test server
import zmq
from mmi import recv_array

if __name__ == '__main__':
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind('tcp://*:3666')

    poll = zmq.Poller()
    poll.register(socket, zmq.POLLIN)

    print('waiting for something to happen...')
    while 1:
        print('receiving...')
        arr, md = recv_array(socket)
        print('received! ', arr, md)
