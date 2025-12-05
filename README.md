# `warpx_analysis`

This package reproduces the analysis previously performed by a Mathematica notebook.

## Requirements

Python 3.13 or later is required.

## Installation

The package should be installed into a virtual environment
(for example,
uv, conda, or venv).

With an environment active,
the package may be installed using Pip:

    pip install git+https://github.com/edbennett/warpx_analysis

Required packages will be installed automatically.

## Usage

The code will give a help message when run with:

    python -m warpx_analysis --help

``` text
usage: python -m warpx_analysis [-h] [--frame_plots_filename FRAME_PLOTS_FILENAME] [--position_plot_filename POSITION_PLOT_FILENAME] [--radius_plot_filename RADIUS_PLOT_FILENAME]
                                [--density_plot_filename DENSITY_PLOT_FILENAME] [--parallel] [--plot_styles PLOT_STYLES]
                                input_file [input_file ...]

Analyse WarpX output without Mathematica

positional arguments:
  input_file            HDF5 filename

options:
  -h, --help            show this help message and exit
  --frame_plots_filename FRAME_PLOTS_FILENAME
                        Where to output plots of density slices
  --position_plot_filename POSITION_PLOT_FILENAME
                        Where to output plot of the plasma position
  --radius_plot_filename RADIUS_PLOT_FILENAME
                        Where to output plot of plasma weighted mean radius
  --density_plot_filename DENSITY_PLOT_FILENAME
                        Where to output plot of plasma density
  --parallel            Use multiple cores for the computation
  --plot_styles PLOT_STYLES
                        Style sheet for plots
```

The options `--frame_plots_filename`,
`---position_plot_filename`,
`--radius_plot_filename`,
and `--density_plot_filename`,
if present,
specify where each output plot will be placed.
If omitted,
then plots will be displayed on screen,
with one exception:
if the program is running in parallel,
then the individual density slice plots will not be generated.
To suppress output of a plot entirely,
set its output filename to `/dev/null`.
