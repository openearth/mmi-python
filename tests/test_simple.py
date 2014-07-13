import unittest
import mmi

import numpy as np
import numpy.testing
import zmq


# global context
ctx = zmq.Context()
class TestRecv(unittest.TestCase):
    def test_sndrcv(self):
        """send an array"""
        A = np.array([1,2,3])
        req = ctx.socket(zmq.REQ)
        req.connect('tcp://localhost:9002')
        rep = ctx.socket(zmq.REP)
        rep.bind('tcp://*:9002')
        mmi.send_array(req, A)
        B, metadata = mmi.recv_array(rep)
        numpy.testing.assert_array_equal(A, B)
    def test_sndrcv_compressed(self):
        """send an array"""
        A = np.array([1,2,3])
        req = ctx.socket(zmq.REQ)
        req.connect('tcp://localhost:9002')
        rep = ctx.socket(zmq.REP)
        rep.bind('tcp://*:9002')
        mmi.send_array(req, A, )
        B, metadata = mmi.recv_array(rep)
        numpy.testing.assert_array_equal(A, B)

    def test_metadata_only(self):
        """send a message with only metadata"""
        req = ctx.socket(zmq.REQ)
        req.connect('tcp://localhost:9002')
        rep = ctx.socket(zmq.REP)
        rep.bind('tcp://*:9002')
        mmi.send_array(req, A=None)
        _, metadata = mmi.recv_array(rep)
        self.assertTrue('timestamp' in metadata)

    def test_missing(self):
        """send an array with missing data"""
        A = np.array([1, 2, 3, 4])
        A = np.ma.masked_less(A, 2)
        req = ctx.socket(zmq.REQ)
        req.connect('tcp://localhost:9002')
        rep = ctx.socket(zmq.REP)
        rep.bind('tcp://*:9002')
        mmi.send_array(req, A)
        B, metadata = mmi.recv_array(rep)
        numpy.testing.assert_array_equal(A, B)

    def test_missing_scalar(self):
        """send an array with missing data as a scalar"""
        A = np.array([1, 2, 3, 4])
        A = np.ma.masked_less(A, 2)
        # test if it works if we use a numpy scalar
        A.fill_value = np.int32(9999)
        req = ctx.socket(zmq.REQ)
        req.connect('tcp://localhost:9002')
        rep = ctx.socket(zmq.REP)
        rep.bind('tcp://*:9002')
        mmi.send_array(req, A)
        B, metadata = mmi.recv_array(rep)
        numpy.testing.assert_array_equal(A, B)



