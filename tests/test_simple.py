import unittest
import mmi

import numpy as np
import numpy.testing
import zmq


# global context
ctx = zmq.Context()
class TestRecv(unittest.TestCase):
    def test_sndrcv(self):
        A = np.array([1,2,3])
        req = ctx.socket(zmq.REQ)
        req.connect('tcp://localhost:9002')
        rep = ctx.socket(zmq.REP)
        rep.bind('tcp://*:9002')
        mmi.send_array(req, A)
        B, metadata = mmi.recv_array(rep)
        numpy.testing.assert_array_equal(A, B)


    def test_missing(self):
        A = np.array([1, 2, 3, 4])
        A = np.ma.masked_less(A, 2)
        req = ctx.socket(zmq.REQ)
        req.connect('tcp://localhost:9002')
        rep = ctx.socket(zmq.REP)
        rep.bind('tcp://*:9002')
        mmi.send_array(req, A)
        B, metadata = mmi.recv_array(rep)
        numpy.testing.assert_array_equal(A, B)



