#!/usr/bin/env python
"""
Usage:
  mmi-runner <engine> <configfile> [-o <outputvar>...] [-g <globalvar>...] [--interval <interval>] [--disable-logger] [--pause] [--mpi <method>] [--track <server>] [--port <port>] [--bmi-class <bmi-class>]
  mmi-runner -h | --help

Positional arguments:
  engine model  engine
  configfile    model configuration file

Optional arguments:
  -h, --help               show this help message and exit
  --interval <interval>    publish results every <interval> timesteps [default: 1]
  -o <outputvar>           output variables, will be broadcasted each <interval> timestep
  --disable-logger         do not inject logger into the BMI library
  --pause                  start in paused mode, send update messages to progress [default: False]
  --mpi <method>           communicate with mpi nodes using one of the methods: root (communicate with rank 0), all (one socket per rank)
  --port <port>            "random" or integer base port, port is computed as req/rep = port + rank*3 + 0, push/pull = port + rank*3 + 1, pub/sub = port + rank*3 + 2 [default: random]
  --track <tracker>        server to subscribe to for tracking
  --bmi-class <bmi-class>  when used the engine is assumed to be the full name of a Python class that implements bmi [default: bmi.wrapper.BMIWrapper]
"""  # noqa: E501


import warnings
import logging

import docopt

import mmi.runner

logger = logging.getLogger(__name__)


def main():
    """run mmi runner"""
    logging.basicConfig()
    logger.info("mmi-runner")
    warnings.warn(
        "You are using the mmi-runner script, please switch to `mmi runner`",
        DeprecationWarning
    )
    arguments = docopt.docopt(__doc__)
    kwargs = parse_args(arguments)
    runner = mmi.runner.Runner(
        **kwargs
    )
    runner.run()


def parse_args(arguments, wrapper_kwargs={}):
    """
    MMI Runner
    """
    # make a socket that replies to message with the grid

    # if we are running mpi we want to know the rank
    args = {}
    positional = [
        'engine',
        'configfile',
    ]
    for key in positional:
        args[key] = arguments['<' + key + '>']

    #  integer if not 'random'
    port = arguments['--port']
    args['port'] = port if port == 'random' else int(port)

    # integer, default 1
    interval = arguments['--interval']
    args['interval'] = int(interval) if interval else 1
    # boolean
    args['pause'] = bool(arguments['--pause'])
    args['bmi_class'] = arguments['--bmi-class']
    args['output_vars'] = arguments['-o']
    args['tracker'] = arguments['--track']

    return args


if __name__ == '__main__':
    main()
