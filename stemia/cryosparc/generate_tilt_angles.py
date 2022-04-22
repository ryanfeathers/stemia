import click


@click.command()
@click.argument('star_file', type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.argument('tilt_angle', type=float)
@click.argument('tilt_axis', type=float)
@click.option('-r', '--radians', is_flag=True, help='Provide angles in radians instead of degrees')
@click.option('-o', '--star-output', type=click.Path(dir_okay=False, resolve_path=True),
              help='where to put the updated version of the star file [default: <STAR_FILE>_tilted.star]')
@click.option('-f', '--overwrite', is_flag=True, help='overwrite output if exists')
def cli(star_file, tilt_angle, tilt_axis, radians, star_output, overwrite):
    """
    Generate angle priors for a tilted dataset.

    Read a Relion STAR_FILE with in-plane angles and generate priors
    for rot and tilt angles based on a TILT_ANGLE around a TILT_AXIS.
    """
    from pathlib import Path

    import numpy as np
    import starfile

    star_output = star_output or Path(star_file).stem + '_tilted.star'
    if Path(star_output).is_file() and not overwrite:
        raise click.UsageError(f'{star_output} exists but "-f" flag was not passed')

    click.secho(f'Reading {star_file}...')
    data = starfile.read(star_file, always_dict=True)

    if not radians:
        tilt_angle = np.deg2rad(tilt_angle)
        tilt_axis = np.deg2rad(tilt_axis)

    click.secho('Calculating angles...')
    inplane = np.deg2rad(data['particles']['rlnAnglePsi'])
    inplane_from_axis = tilt_axis - inplane

    # tilt is max when perpendicular to tilt axis
    tilt = tilt_angle * np.sin(inplane_from_axis)
    # rot is max when parallel to tilt axis
    rot = tilt_angle * np.cos(inplane_from_axis)

    data['particles']['rlnAngleRot'] = np.rad2deg(rot)
    data['particles']['rlnAngleTilt'] = np.rad2deg(tilt)

    click.secho(f'Writing {star_output}...')
    starfile.write(data, star_output, overwrite=overwrite, sep=' ')