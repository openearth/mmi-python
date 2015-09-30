#server 

import zmq
import random
import sys
import time
import numpy as np

port = "5557"
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.bind("tcp://127.0.0.1:%s" % port)

# 320 Mbit
data = np.empty((1000000, 4))

while True:
    t1 = time.time()
    socket.send(data)
    t2 = time.time()    
    print "Time to send: " + str(t2 - t1)
    
    t1 = time.time()
    msg = socket.recv()
    t2 = time.time()    
    print "Time to receive: " + str(t2 - t1)
    
    time.sleep(1)