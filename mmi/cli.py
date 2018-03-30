# -*- coding: utf-8 -*-

"""Console script for mmi."""
import sys
import click

import mmi.runner
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
    '--global',
    '-g',
    'global_vars',
    multiple=True,
    help='global variables, will be send on request'
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
    help='"random" or integer base port, port is computed as req/rep = port + rank*3 + 0, push/pull = port + rank*3 + 1, pub/sub = port + rank*3 + 2 [default: random]'
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
        global_vars,
        output_vars,
        interval,
        pause,
        mpi, tracker,
        port,
        bmi_class
):
    """
    Run a BMI compatible model
    """
    # keep track of info
    metadata = {}


    # update mpi information or use rank 0
    mpi_info = mmi.runner.initialize_mpi(mpi)
    ports_info = mmi.runner.create_ports(port, mpi, mpi_info['rank'])
    sockets = mmi.runner.create_sockets(ports_info)
    model = mmi.runner.create_bmi_model(engine, bmi_class)

    metadata.update(mpi_info)
    metadata.update(ports_info)

    mmi.runner.run(
        model,
        configfile,
        tracker,
        sockets,
        global_vars,
        output_vars,
        interval,
        port,
        metadata
    )


if __name__ == "__main__":
    sys.exit(cli())  # pragma: no cover
