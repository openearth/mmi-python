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
@click.argument('configfile')
def runner(engine, configfile):
    """
    mmi-runner <engine> <configfile> [-o <outputvar>...] [-g <globalvar>...] [--interval <interval>] [--disable-logger] [--pause] [--mpi <method>] [--track <server>] [--port <port>] [--bmi-class]
    """





if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
