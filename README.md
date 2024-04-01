# STEMIA

[![License](https://img.shields.io/pypi/l/stemia.svg?color=green)](https://github.com/brisvag/stemia/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/stemia.svg?color=green)](https://pypi.org/project/stemia)
[![Python Version](https://img.shields.io/pypi/pyversions/stemia.svg?color=green)](https://python.org)

**S**cripts and **T**ools for **E**lectron **M**icroscopy **I**mage **A**nalysis.

This is a simple personal collection of (sometimes...) useful scripts and tools for cryoem/cryoet.

## Installation

```bash
pip install stemia
```

You can quickly list all the available tools with

```
stemia -l
```

## Completion

You can enable completion for your bash shell by running:

```
eval "$(_STEMIA_COMPLETE=bash_source stemia)"
```

See [the click docs](https://click.palletsprojects.com/en/8.1.x/shell-completion/#enabling-completion) for how to do it for other shells.

<!-- autogenerated content start here -->

## Tools

Everything is accessible through the main command line interface `stemia`.

Try `stemia -h` for help, or `stemia -l` for the command tree:

```
.stemia
├── aretomo:  A collection of AreTomo-related tools and scripts.
│   ├── aln2xf:  Convert AreTomo `aln` file to imod `xf` format.
│   └── batch:  Run AreTomo on a full directory.
├── cryosparc:  A collection of Cryosparc-related tools and scripts.
│   ├── csplot:  Read a cryosparc job directory and plot interactively any column.
│   ├── fix_filament_ids:  Replace cryosparc filament ids with small unique integers.
│   ├── generate_tilt_angles:  Generate angle priors for a tilted dataset.
│   ├── merge_defects_gainref:  Merge serialEM defects and gainref for cryosparc usage.
│   └── time_wasted:  Print the total amount of time wasted on a project.
├── image:  Simple image manipulation and processing.
│   ├── center_filament:  Center an mrc image (stack) containing filament(s).
│   ├── classify_densities:  Do hierarchical classification of particle stacks based on densities.
│   ├── create_mask:  Create a mask for INPUT.
│   ├── extract_z_snapshots:  Grab z slices at regular intervals from a tomogram as jpg images.
│   ├── flip_z:  Flip the z axis for particles in a RELION star file.
│   ├── fourier_crop:  Bin mrc images to the specified pixel size using fourier cropping.
│   ├── project_profiles:  Project re-extracted and straightened membranes and get some stats.
│   │   ├── prepare:  Generate and select 2D chunked projections for the input data.
│   │   ├── compute:  Take the outputs from prepare and compute statistics and plots.
│   │   └── aggregate:  Aggregate the generated data into general stats about given subsets.
│   └── rescale:  Rescale an mrc image to the specified pixel size.
├── imod:  A collection of IMOD-related tools and scripts.
│   └── find_NAD_params:  Test a range of k and iteration values for nad_eed_3d.
├── relion:  A collection of Relion-related tools and scripts.
│   ├── align_filament_particles:  Fix filament PsiPriors so they are consistent within a filament.
│   └── edit_star:  Simple search-replace utility for star files.
└── warp:  A collection of Warp-related tools and scripts.
    ├── fix_mdoc:  Fix mdoc files to point to the right data and follow warp format.
    ├── offset_angle:  Offset tilt angles in warp xml files.
    ├── parse_xml:  Parse a warp xml file and print its content.
    ├── prepare_isonet:  Update an isonet starfile with preprocessing data from warp.
    ├── spoof_mdoc:  Create dummy mdocs for warp.
    ├── summarize:  Summarize the state of a Warp project.
    └── preprocess_serialem:  Prepare and unpack data from sterialEM for Warp.
```
    
### stemia aretomo aln2xf

```
Usage: stemia aretomo aln2xf [OPTIONS] ALN_FILE

  Convert AreTomo `aln` file to imod `xf` format.

Options:
  -f, --overwrite  overwrite existing output
  --help           Show this message and exit.
```

### stemia aretomo batch

```
Usage: stemia aretomo batch [OPTIONS]

  Run AreTomo on a full directory.

Options:
  --help  Show this message and exit.
```

### stemia cryosparc csplot

```
Usage: stemia cryosparc csplot [OPTIONS] JOB_DIR

  Read a cryosparc job directory and plot interactively any column.

  All the related data from parent jobs will also be loaded. An interactive
  ipython shell will be opened with data loaded into a pandas dataframe.

  JOB_DIR:     a cryosparc job directory.

Options:
  --drop-na         drop rows that contain NaN values (e.g: micrographs with
                    no particles)
  --no-particles    do not read particles data
  --no-micrographs  do not read micrographs data
  --help            Show this message and exit.
```

### stemia cryosparc fix_filament_ids

```
Usage: stemia cryosparc fix_filament_ids [OPTIONS] STAR_FILE

  Replace cryosparc filament ids with small unique integers.

  Relion will fail with cryosparc IDs because of overflows.

Options:
  -o, --star-output FILE  where to put the updated version of the star file
                          [default: <STAR_FILE>_fixed_id.star]
  -f, --overwrite         overwrite output if exists
  --help                  Show this message and exit.
```

### stemia cryosparc generate_tilt_angles

```
Usage: stemia cryosparc generate_tilt_angles [OPTIONS] STAR_FILE TILT_ANGLE
                                             TILT_AXIS

  Generate angle priors for a tilted dataset.

  Read a Relion STAR_FILE with in-plane angles and generate priors for rot and
  tilt angles based on a TILT_ANGLE around a TILT_AXIS.

Options:
  -r, --radians           Provide angles in radians instead of degrees
  -o, --star-output FILE  where to put the updated version of the star file
                          [default: <STAR_FILE>_tilted.star]
  -f, --overwrite         overwrite output if exists
  --help                  Show this message and exit.
```

### stemia cryosparc merge_defects_gainref

```
Usage: stemia cryosparc merge_defects_gainref [OPTIONS] DEFECTS GAINREF

  Merge serialEM defects and gainref for cryosparc usage.

  requires active sbrgrid.

Options:
  -d, --output-defects FILE
  -o, --output-gainref FILE
  -f, --overwrite            overwrite output if exists
  --help                     Show this message and exit.
```

### stemia cryosparc time_wasted

```
Usage: stemia cryosparc time_wasted [OPTIONS] [PROJECT_DIRS]...

  Print the total amount of time wasted on a project.

Options:
  -u, --useful_jobs TEXT  ID of job that gave useful results. Its running time
                          and that of its parents will be used to calculate
                          useful time. Can be passed multiple times.
  --help                  Show this message and exit.
```

### stemia image center_filament

```
Usage: stemia image center_filament [OPTIONS] INPUT [OUTPUT]

  Center an mrc image (stack) containing filament(s).

  Can update particles in a RELION .star file accordingly. If OUTPUT is not
  given, default to INPUT_centered.mrc

Options:
  -s, --update-star FILE        a RELION .star file to update with new
                                particle positions
  -o, --star-output FILE        where to put the updated version of the star
                                file. Only used if -s is passed [default:
                                STARFILE_centered.star]
  --update-by [class|particle]  whether to update particle positions by
                                classes or 1 by 1. Only used if -s is passed
                                [default: class]
  -f, --overwrite               overwrite output if exists
  -n, --n-filaments INTEGER     number of filaments on the image  [default: 2]
  -p, --percentile INTEGER      percentile for binarisation  [default: 85]
  --help                        Show this message and exit.
```

### stemia image classify_densities

```
Usage: stemia image classify_densities [OPTIONS] [STACKS]...

  Do hierarchical classification of particle stacks based on densities.

Options:
  -c, --max-classes INTEGER
  --help                     Show this message and exit.
```

### stemia image create_mask

```
Usage: stemia image create_mask [OPTIONS] INPUT OUTPUT

  Create a mask for INPUT.

  Axis order is zyx!

Options:
  -t, --mask-type [sphere|cylinder|threshold]
  -c, --center TEXT               center of the mask (comma-separated floats)
  -a, --axis INTEGER              main symmetry axis (for cylinder)
  -r, --radius FLOAT              radius of the mask. If thresholding,
                                  equivalent to "hard padding"  [required]
  -i, --inner-radius FLOAT        inner radius of the mask (if any)
  -p, --padding FLOAT             smooth padding
  --ang / --px                    whether the radius and padding are in
                                  angstrom or pixels
  --threshold FLOAT               threshold for binarization of the input map
  -f, --overwrite                 overwrite output if exists
  --help                          Show this message and exit.
```

### stemia image extract_z_snapshots

```
Usage: stemia image extract_z_snapshots [OPTIONS] [INPUTS]...

  Grab z slices at regular intervals from a tomogram as jpg images.

  INPUTS: any number of paths of volume images

Options:
  -o, --output-dir PATH
  --mrc                   also output mrc files
  -n, --n-slices INTEGER  number of equidistant slices to extract
  --keep-extrema          whether to keep slices at z=0 and z=-1 (if false,
                          slices is reduced by 2)
  -a, --average INTEGER   number of slices to average over
  -s, --size TEXT         size of final image (X,Y)
  -r, --range TEXT        range of slices to image (A,B)
  --axis INTEGER          axis along which to do the slicing
  --help                  Show this message and exit.
```

### stemia image flip_z

```
Usage: stemia image flip_z [OPTIONS] STAR_PATH

  Flip the z axis for particles in a RELION star file.

  STAR_PATH: star file to flip along z

  Assumes all tomograms have the same shape.

Options:
  -o, --output FILE
  -m, --mrc_path FILE
  --star_pixel_size FLOAT
  --mrc_pixel_size FLOAT
  --z_shape INTEGER
  --help                   Show this message and exit.
```

### stemia image fourier_crop

```
Usage: stemia image fourier_crop [OPTIONS] [INPUTS]...

  Bin mrc images to the specified pixel size using fourier cropping.

Options:
  -b, --binning FLOAT  binning amount  [required]
  -f, --overwrite      overwrite output if exists
  --help               Show this message and exit.
```

### stemia image project_profiles prepare

```
Usage: stemia image project_profiles prepare [OPTIONS] [PATHS]...

  Generate and select 2D chunked projections for the input data.

Options:
  -o, --output PATH         [required]
  -s, --chunk-size INTEGER
  -f, --overwrite
  --help                    Show this message and exit.
```

### stemia image project_profiles compute

```
Usage: stemia image project_profiles compute [OPTIONS] PROJ_DIR

  Take the outputs from prepare and compute statistics and plots.

Options:
  -f, --overwrite
  --help           Show this message and exit.
```

### stemia image project_profiles aggregate

```
Usage: stemia image project_profiles aggregate [OPTIONS] [INPUTS]...

  Aggregate the generated data into general stats about given subsets.

  Inputs are subdirectories of the project_dir from compute.

Options:
  -o, --output-name TEXT  Title/filename given to the aggregated outputs.
  --help                  Show this message and exit.
```

### stemia image rescale

```
Usage: stemia image rescale [OPTIONS] INPUT OUTPUT TARGET_PIXEL_SIZE

  Rescale an mrc image to the specified pixel size.

  TARGET_PIXEL_SIZE: target pixel size in Angstrom

Options:
  --input-pixel-size FLOAT  force input pizel size and ignore mrc header
  -f, --overwrite           overwrite output if exists
  --help                    Show this message and exit.
```

### stemia imod find_NAD_params

```
Usage: stemia imod find_NAD_params [OPTIONS] INPUT

  Test a range of k and iteration values for nad_eed_3d.

Options:
  -k, --k-values TEXT
  -i, --iterations TEXT
  -s, --std TEXT
  --help                 Show this message and exit.
```

### stemia relion align_filament_particles

```
Usage: stemia relion align_filament_particles [OPTIONS] STAR_FILE

  Fix filament PsiPriors so they are consistent within a filament.

  Read a Relion STAR_FILE with in-plane angles and filament info and flip any
  particle that's not consistent with the rest of the filament.

  If a consensus cannot be reached, or the filament has too few particles,
  discard the whole filament.

Options:
  -o, --star-output FILE          where to put the updated version of the star
                                  file [default: <STAR_FILE>_aligned.star]
  -t, --tolerance FLOAT           angle in degrees within which neighbouring
                                  particles are considered aligned
  -c, --consensus-threshold FLOAT
                                  require an angle consensus at least higher
                                  than this to use a filament.
  -d, --drop-below INTEGER        drop filaments if they have fewer than this
                                  number of particles
  -r, --rotate-bad-particles      rotate bad particles to match the rest of
                                  the filament
  -f, --overwrite                 overwrite output if exists
  --help                          Show this message and exit.
```

### stemia relion edit_star

```
Usage: stemia relion edit_star [OPTIONS] [STAR_FILES]...

  Simple search-replace utility for star files.

  Full regex functionality works (e.g: reusing groups in output)

Options:
  -s, --suffix-output TEXT  suffix added to the output files before extension
  -c, --column TEXT         column(s) to modify
  -i, --regex-in TEXT       regex sed-like search pattern(s)
  -o, --regex-out TEXT      regex sed-like substitution to apply to the
                            column(s)
  -f, --overwrite           overwrite output if exists
  --help                    Show this message and exit.
```

### stemia warp fix_mdoc

```
Usage: stemia warp fix_mdoc [OPTIONS] MDOC_DIR

  Fix mdoc files to point to the right data and follow warp format.

Options:
  -d, --data-dir PATH
  --dates              fix date format
  --paths              fix image paths
  --help               Show this message and exit.
```

### stemia warp offset_angle

```
Usage: stemia warp offset_angle [OPTIONS] [WARP_DIR]

  Offset tilt angles in warp xml files.

Options:
  --help  Show this message and exit.
```

### stemia warp parse_xml

```
Usage: stemia warp parse_xml [OPTIONS] XML_FILE

  Parse a warp xml file and print its content.

Options:
  --help  Show this message and exit.
```

### stemia warp prepare_isonet

```
Usage: stemia warp prepare_isonet [OPTIONS] WARP_DIR ISO_STAR

  Update an isonet starfile with preprocessing data from warp.

Options:
  --help  Show this message and exit.
```

### stemia warp spoof_mdoc

```
Usage: stemia warp spoof_mdoc [OPTIONS] [RAWTLT_FILES]...

  Create dummy mdocs for warp.

  RAWTLT_FILES: simple file with one tilt angle per line. Order should match
  sorted filenames.

Options:
  -d, --dose-per-image FLOAT  electron dose per tilt image (or per frame if
                              inputs are movies)  [required]
  -p, --pixel-size FLOAT
  -e, --extension [tif|mrc]
  -f, --overwrite
  --help                      Show this message and exit.
```

### stemia warp summarize

```
Usage: stemia warp summarize [OPTIONS] [WARP_DIR]

  Summarize the state of a Warp project.

  Reports for each tilt series: - discarded: number of discarded tilts -
  total: total number oftilts in raw data - stacked: number of image slices in
  imod output directory - mismatch: whether stacked != (total - discarded) -
  resolution: estimated resolution if processed

Options:
  --help  Show this message and exit.
```

### stemia warp preprocess_serialem

```
Usage: stemia warp preprocess_serialem [OPTIONS] RAW_DATA_DIR

  Prepare and unpack data from sterialEM for Warp.

  You must be in a new directory for this to work; new files will be placed
  there with the same name as the original tifs.

  RAW_DATA_DIR: the directory containing the raw data

Options:
  --help  Show this message and exit.
```
