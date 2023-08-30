import click
from enum import Enum, auto


class ProcessingStep(str, Enum):
    fix = auto()
    align = auto()
    tilt_mdocs = auto()
    reconstruct = auto()
    stack_halves = auto()
    reconstruct_halves = auto()
    denoise = auto()

    def __str__(self):
        return self.name


@click.command()
@click.argument('warp_dir', type=click.Path(exists=True, dir_okay=True, resolve_path=True))
@click.option('-m', '--mdoc-dir', type=click.Path(exists=True, dir_okay=True, resolve_path=True))
@click.option('-o', '--output-dir', type=click.Path(exists=True, dir_okay=True, resolve_path=True),
              help='output directory for all the processing. If None, defined as warp_dir/aretomo')
@click.option('-d', '--dry-run', is_flag=True, help='only print some info, without running the commands.')
@click.option('-v', '--verbose', is_flag=True, help='print individual commands')
@click.option('-j', '--just', type=str, multiple=True,
              help='reconstruct just these tomograms')
@click.option('-e', '--exclude', type=str, multiple=True,
              help='exclude these tomograms from the run')
@click.option('-t', '--sample-thickness', type=int, default=400,
              help='unbinned thickness of the SAMPLE (ice or lamella) used for alignment')
@click.option('-z', '--z-thickness', type=int, default=1200,
              help='unbinned thickness of the RECONSTRUCTION.')
@click.option('-b', '--binning', type=int, default=4, help='binning for aretomo reconstruction (relative to warp binning)')
@click.option('-a', '--tilt-axis', type=float, help='starting tilt axis for AreTomo, if any')
@click.option('-p', '--patches', type=int, help='number of patches for local alignment in aretomo (NxN), if any')
@click.option('-r', '--roi-dir', type=click.Path(exists=True, dir_okay=True, resolve_path=True),
              help='directory containing ROI files. Extension does not matter, but names should be same as TS.')
@click.option('-f', '--overwrite', is_flag=True, help='overwrite any previous existing run')
@click.option('--train', is_flag=True, default=False, help='whether to train a new denosing model')
@click.option('--topaz-patch-size', type=int, default=32, help='patch size for denoising in topaz.')
@click.option('--start-from', type=click.Choice(ProcessingStep.__members__), default='fix',
              help='use outputs from a previous run, starting processing at this step')
@click.option('--stop-at', type=click.Choice(ProcessingStep.__members__), default='denoise',
              help='terminate processing after this step')
@click.option('--ccderaser', type=str, default='ccderaser', help='command for ccderaser')
@click.option('--aretomo', type=str, default='AreTomo', help='command for aretomo')
@click.option('--gpus', type=str, help='Comma separated list of gpus to use for aretomo. Default to all.')
@click.option('--tiltcorr/--no-tiltcorr', default=True, help='do not correct sample tilt')
def cli(warp_dir, mdoc_dir, output_dir, dry_run, verbose, just, exclude, sample_thickness, z_thickness, binning, tilt_axis, patches, roi_dir, overwrite, train, topaz_patch_size, start_from, stop_at, ccderaser, aretomo, gpus, tiltcorr):
    """
    Run aretomo in batch on data preprocessed in warp.

    Needs to be ran after imod stacks were generated. Requires ccderaser and AreTomo>=1.3.0.
    Assumes the default Warp directory structure with generated imod stacks.
    """
    from inspect import cleandoc
    from pathlib import Path

    from rich.panel import Panel
    from rich.progress import Progress
    from rich import print

    from .parse import parse_data

    if gpus is not None:
        gpus = [int(gpu) for gpu in gpus.split(',')]

    warp_dir = Path(warp_dir)
    if mdoc_dir is None:
        mdoc_dir = warp_dir
    mdoc_dir = Path(mdoc_dir)
    if output_dir is None:
        output_dir = warp_dir / 'stemia'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if roi_dir is not None:
        roi_dir = Path(roi_dir)

    with Progress() as progress:
        tilt_series, tilt_series_excluded, tilt_series_unprocessed = parse_data(
            progress,
            warp_dir,
            mdoc_dir=mdoc_dir,
            output_dir=output_dir,
            roi_dir=roi_dir,
            just=just,
            exclude=exclude,
            train=train,
        )

        aretomo_kwargs = dict(
            cmd=aretomo,
            tilt_axis=tilt_axis,
            patches=patches,
            thickness_align=sample_thickness,
            thickness_recon=z_thickness,
            binning=binning,
            gpus=gpus,
            tilt_corr=tiltcorr,
        )

        meta_kwargs = dict(
            overwrite=overwrite,
            dry_run=dry_run,
            verbose=verbose,
        )

        start_from = ProcessingStep[start_from]
        stop_at = ProcessingStep[stop_at]

        steps = {step: start_from <= val <= stop_at for step, val in ProcessingStep.__members__.items()}
        if not train:
            steps['stack_halves'] = False
            steps['reconstruct_halves'] = False

        nl = '\n'
        print(Panel(cleandoc(f'''
            [bold]Warp directory[/bold]: {warp_dir}
            [bold]Mdoc directory[/bold]: {mdoc_dir}
            [bold]Tilt series - NOT READY[/bold]: {''.join(f'{nl}{" " * 12}- {ts}' for ts in tilt_series_unprocessed)}
            [bold]Tilt series - READY[/bold]: {''.join(f'{nl}{" " * 12}- {ts["name"]}' for ts in tilt_series)}
            [bold]Tilt series - EXCLUDED[/bold]: {''.join(f'{nl}{" " * 12}- {ts}' for ts in tilt_series_excluded)}
            [bold]Processing steps[/bold]: {''.join(f'{nl}{" " * 12}- [{"green" if v else "red"}]{k}[/{"green" if v else "red"}] ' for k, v in steps.items())}
            [bold]Run options[/bold]: {''.join(f'{nl}{" " * 12}- {k}: {v}' for k, v in meta_kwargs.items())}
            [bold]AreTomo options[/bold]: {''.join(f'{nl}{" " * 12}- {k}: {v}' for k, v in aretomo_kwargs.items())}
        ''')))

        if steps['fix']:
            from .fix import fix_batch
            if verbose:
                print('\n[green]Fixing with ccderaser...')
            fix_batch(progress, tilt_series, cmd=ccderaser, **meta_kwargs)

        if steps['align']:
            from .aretomo import aretomo_batch
            if verbose:
                print('\n[green]Aligning with AreTomo...')
            aretomo_batch(
                progress,
                tilt_series,
                label='Aligning',
                **aretomo_kwargs,
                **meta_kwargs,
            )

        if steps['tilt_mdocs']:
            if not tiltcorr:
                print('No need to tilt mdocs!')
            else:
                from .fix_mdoc import tilt_mdocs_batch
                if verbose:
                    print('\n[green]Tilting mdocs...')
                (mdoc_dir / 'mdoc_tilted').mkdir(parents=True, exist_ok=True)
                tilt_mdocs_batch(
                    progress,
                    tilt_series,
                    **meta_kwargs,
                )

        if steps['reconstruct']:
            from .aretomo import aretomo_batch
            if verbose:
                print('\n[green]Reconstructing with AreTomo...')
            aretomo_batch(
                progress,
                tilt_series,
                reconstruct=True,
                label='Reconstructing',
                **aretomo_kwargs,
                **meta_kwargs,
            )

        if steps['stack_halves']:
            from .stack import prepare_half_stacks
            for half in ('even', 'odd'):
                if verbose:
                    print(f'\n[green]Preparing {half} stacks for denoising...')
                prepare_half_stacks(progress, tilt_series, half=half, **meta_kwargs)

        if steps['reconstruct_halves']:
            from .aretomo import aretomo_batch
            for half in ('even', 'odd'):
                if verbose:
                    print(f'\n[green]Reconstructing {half} tomograms for deonoising...')
                (output_dir / half).mkdir(parents=True, exist_ok=True)
                aretomo_batch(
                    progress,
                    tilt_series,
                    suffix=f'_{half}',
                    reconstruct=True,
                    label=f'Reconstructing {half} halves',
                    **aretomo_kwargs,
                    **meta_kwargs,
                )

        if steps['denoise']:
            from .topaz import topaz_batch
            if verbose:
                print('\n[green]Denoising tomograms...')
            topaz_batch(
                progress,
                tilt_series,
                outdir=output_dir,
                train=train,
                patch_size=topaz_patch_size,
                **meta_kwargs
            )