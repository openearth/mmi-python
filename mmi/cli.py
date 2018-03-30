# -*- coding: utf-8 -*-

"""Console script for mmi."""
import sys
import click


@click.group()
def cli(args=None):
    """Console script for mmi."""
    click.echo("Model Message Interface")
    return 0


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
    '--global',
    '-g',
    'global_vars',
    multiple=True,
    help='global variables, will be send on request'
)
@click.option(
    '--interval',
    type=float,
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
@click.option('--track', help='server to subscribe to for tracking')
@click.option(
    '--port',
    type=int,
    help='"random" or integer base port, port is computed as req/rep = port + rank*3 + 0, push/pull = port + rank*3 + 1, pub/sub = port + rank*3 + 2 [default: random]'
)
@click.option(
    '--bmi-class',
    'bmi_class',
    default='bmi.wrapper.BMIWrapper',
    help='the full name of a python class that implements bmi'
)
def runner(engine, configfile, output_vars, global_vars, interval, pause, mpi, track, port, bmi_class):
    """
    mmi-runner <engine> <configfile> [-o <outputvar>...] [-g <globalvar>...] [--interval <interval>] [--disable-logger] [--pause] [--mpi <method>] [--track <server>] [--port <port>] [--bmi-class]
    """





if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
