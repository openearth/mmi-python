#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `mmi` package."""

import pytest

from click.testing import CliRunner

import mmi
import mmi.cli

import numpy as np
import numpy.testing
import zmq

import logging

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def context():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # global context
    ctx = zmq.Context()
    yield ctx


@pytest.fixture()
def arr():
    arr = np.array([1, 2, 3])
    return arr


@pytest.fixture
def req(context):
    """return a request socket"""
    req = context.socket(zmq.REQ)
    req.connect('tcp://localhost:9002')
    logger.debug('connect: %s', req)
    yield req
    logger.debug('closing: %s', req)
    req.close()


@pytest.fixture
def rep(context):
    """return a reply socket"""
    rep = context.socket(zmq.REP)
    rep.bind('tcp://*:9002')
    logger.debug('bind: %s', req)
    yield rep
    logger.debug('closing: %s', rep)
    rep.close()


def test_sndrcv(arr, req, rep):
    """send an array"""
    mmi.send_array(req, arr)
    received, metadata = mmi.recv_array(rep)
    numpy.testing.assert_array_equal(arr, received)


def test_metadata_only(req, rep):
    """send a message with only metadata"""
    mmi.send_array(req, A=None)
    _, metadata = mmi.recv_array(rep)
    assert 'timestamp' in metadata


def test_missing(arr, req, rep):
    """send an array with missing data"""
    arr_masked = np.ma.masked_less(arr, 2)
    mmi.send_array(req, arr_masked)
    received, metadata = mmi.recv_array(rep)
    numpy.testing.assert_array_equal(arr_masked, received)


def test_missing_scalar(arr, req, rep):
    """send an array with missing data as a scalar"""
    arr_masked = np.ma.masked_less(arr, 2)
    # test if it works if we use a numpy scalar
    arr_masked.fill_value = np.int32(9999)
    mmi.send_array(req, arr_masked)
    received, metadata = mmi.recv_array(rep)
    numpy.testing.assert_array_equal(arr_masked, received)


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(mmi.cli.main)
    assert result.exit_code == 0
    assert 'mmi.cli.main' in result.output
    help_result = runner.invoke(mmi.cli.main, ['--help'])
    assert help_result.exit_code == 0
    assert '--help  Show this message and exit.' in help_result.output


def test_command_line_runner():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(mmi.cli.runner)
    assert result.exit_code == 0
    assert 'mmi.cli.runner' in result.output
    help_result = runner.invoke(mmi.cli.runner, ['--help'])
    assert help_result.exit_code == 0
    assert '--help  Show this message and exit.' in help_result.output


def test_command_line_tracker():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(mmi.cli.tracker)
    assert result.exit_code == 0
    assert 'mmi.cli.tracker' in result.output
    help_result = runner.invoke(mmi.cli.tracker, ['--help'])
    assert help_result.exit_code == 0
    assert '--help  Show this message and exit.' in help_result.output


def test_command_line_curl():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(mmi.cli.tracker)
    assert result.exit_code == 0
    assert 'mmi.cli.tracker' in result.output
    help_result = runner.invoke(mmi.cli.tracker, ['--help'])
    assert help_result.exit_code == 0
    assert '--help  Show this message and exit.' in help_result.output
