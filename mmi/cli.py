# -*- coding: utf-8 -*-

"""Console script for mmi."""

import logging
import sys

import click
import tornado.ioloop

import mmi.runner
import mmi.tracker

logging.basicConfig()
logger = logging.getLogger(__name__)


@click.group()
def cli(args=None):
    """Console script for mmi."""
    click.echo("Model Message Interface")
    return 0


@cli.command()
def tracker():
    """start a tracker to register running models"""
    application = mmi.tracker.app()
    application.listen(22222)
    logger.info('serving at port 22222')
    tornado.ioloop.IOLoop.instance().start()


@cli.command()
@click.argument('engine')
@click.argument(
    'configfile',
    type=click.Path(exists=True)
)
@click.option(
    '--output',
    '-o',
    'output_vars',
    multiple=True,
    help='output variables, will be broadcasted each <interval> timestep'
)
@click.option(
    '--interval',
    type=int,
    default=1,
    help='publish results every <interval> timesteps'
)
@click.option(
    '--pause',
    is_flag=True,
    help='start in paused mode, send update messages to progress'
)
@click.option(
    '--mpi',
    type=click.Choice(['root', 'all']),
    help="communicate with mpi nodes using one of the methods: root (communicate with rank 0), all (one socket per rank)"
)
@click.option('--tracker', help='server to subscribe to for tracking')
@click.option(
    '--port',
    type=int,
    help='''"random" or integer base port,
    port is computed as req/rep = port + rank*3 + 0, push/pull = port + rank*3 + 1, pub/sub = port + rank*3 + 2 [default: random]
    '''
)
@click.option(
    '--bmi-class',
    'bmi_class',
    default='bmi.wrapper.BMIWrapper',
    help='the full name of a python class that implements bmi'
)
def runner(
        engine,
        configfile,
        output_vars,
        interval,
        pause,
        mpi,
        tracker,
        port,
        bmi_class
):
    """
    run a BMI compatible model
    """
    # keep track of info
    # update mpi information or use rank 0
    runner = mmi.runner.Runner(
        engine=engine,
        configfile=configfile,
        output_vars=output_vars,
        interval=interval,
        pause=pause,
        mpi=mpi,
        tracker=tracker,
        port=port,
        bmi_class=bmi_class
    )
    runner.run()


if __name__ == "__main__":
    sys.exit(cli())  # pragma: no cover
