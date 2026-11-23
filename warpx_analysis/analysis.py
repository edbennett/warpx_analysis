#!/usr/bin/env python3

import collections
import contextlib
import glob
import multiprocessing
import pathlib
import tempfile
from argparse import ArgumentParser

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pypdf
from matplotlib.backends.backend_pdf import PdfPages
from scipy import constants

STEP_INCREMENT_NS = 0.2
X_SIZE = 0.12
Y_SIZE = 0.12
Z_SIZE = 0.12
OUTER_CIRCLE_RADIUS = 0.051
INNER_CIRCLE_RADIUS = 0.01
ORIGIN = (0, 0)
INNER_CIRCLE_POSITION = (0.04, 0)


ResultTuple = collections.namedtuple(
    "ResultTuple",
    [
        "time",
        "max_rho_x",
        "max_rho_y",
        "centroid_x",
        "centroid_y",
        "radial_mean",
        "radial_std",
        "max_planar_rho",
        "mean_planar_rho",
        "constrained_radial_mean",
    ],
)


def get_args():
    parser = ArgumentParser(description="Analyse WarpX output without Mathematica")
    parser.add_argument(
        "input_files", nargs="+", metavar="input_file", help="HDF5 filename"
    )
    parser.add_argument(
        "--frame_plots_filename",
        default=None,
        help="Where to output plots of density slices",
    )
    parser.add_argument(
        "--position_plot_filename",
        default=None,
        help="Where to output plot of the plasma position",
    )
    parser.add_argument(
        "--radius_plot_filename",
        default=None,
        help="Where to output plot of plasma weighted mean radius",
    )
    parser.add_argument(
        "--density_plot_filename",
        default=None,
        help="Where to output plot of plasma density",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use multiple cores for the computation",
    )
    parser.add_argument("--plot_styles", default=None, help="Style sheet for plots")
    return parser.parse_args()


def get_step_index(h5file):
    (step_index_str,) = h5file["data"].keys()
    return int(step_index_str)


def get_centroid(rho_slice):
    """
    Find plasma centre.
    Mean position of the density field rho in a given slice.
    """

    sum_rho_slice = rho_slice.sum()
    centroid_position = []

    for dimension_index, dimension_size in enumerate(rho_slice.shape):
        dim_slice = rho_slice.sum(axis=dimension_index)
        centroid_position.append(
            np.dot(dim_slice, np.arange(dimension_size)) / sum_rho_slice
        )

    return tuple(centroid_position[::-1])


def get_middle_rho_slice(rho):
    """
    Extract XY central plane data.
    """
    middle_slice_index = rho.shape[0] // 2 + 1
    return rho[middle_slice_index]


def plot_rho_slice(ax, rho_slice, extent, **params):
    """
    Plot the given slice on the given axes with the given extent,
    and set the plot title.
    """
    ax.set_xlabel("$x$")
    return ax.imshow(
        rho_slice,
        origin="lower",
        extent=extent,
        interpolation="gaussian",
        cmap="magma",
        **params,
    )


def save_or_show(fig, plot_target):
    """
    Output the given `fig`,
    and flush it from the buffer.
    `plot_target` may be one of:

        None: Output to the screen.

        "/dev/null": Do not output, only close the fig.

        Any other `str`: Output to the specified filename.

        A Matplotlib backend: Output via the given backend.
    """
    if plot_target:
        if isinstance(plot_target, str):
            if plot_target != "/dev/null":
                fig.savefig(plot_target)
        else:
            plot_target.savefig(fig)

        plt.close(fig)
    else:
        plt.show()


def get_cmap_norm(*slices):
    """
    Given a set of Numpy arrays,
    find a colourmap encompassing the global maximum and minimum.
    """
    min_value = min(single_slice.min() for single_slice in slices)
    max_value = max(single_slice.max() for single_slice in slices)
    return {"vmin": min_value, "vmax": max_value}


def plot_slices(rho_xy_slice, rho_xz_slice, step_index, centroid_y, plot_target):
    """
    Plot the two provided slices with appropriate labels,
    side-by-side.
    """
    cmap_norm = get_cmap_norm(rho_xy_slice, rho_xz_slice)

    # xy plot
    fig, (xy_ax, xz_ax) = plt.subplots(
        ncols=2,
        layout="constrained",
        sharey=True,
        figsize=(4.5, 2),
    )
    xy_ax.set_ylabel("$y$")
    xy_ax.set_title(
        f"Step No. {step_index} – {round(STEP_INCREMENT_NS * step_index, 3)} ns"
    )

    plot_rho_slice(
        xy_ax,
        rho_xy_slice,
        extent=(-X_SIZE / 2, X_SIZE / 2, -Y_SIZE / 2, Y_SIZE / 2),
        **cmap_norm,
    )

    for position, radius, colour in [
        (ORIGIN, OUTER_CIRCLE_RADIUS, "red"),
        (INNER_CIRCLE_POSITION, INNER_CIRCLE_RADIUS, "green"),
    ]:
        xy_ax.add_patch(plt.Circle(position, radius, color=colour, fill=False))

    # xz plot
    rounded_y = round(centroid_y)
    slice_position = round(
        (rounded_y - rho_xz_slice.shape[1] / 2) * Y_SIZE / rho_xz_slice.shape[1],
        3,
    )

    xz_ax.set_title(f"$x = {slice_position} \\mathrm{{m}}$, cell = {rounded_y}")
    rho_xz_plot = plot_rho_slice(
        xz_ax,
        rho_xz_slice,
        extent=(-Z_SIZE / 2, Z_SIZE / 2, -X_SIZE / 2, X_SIZE / 2),
        **cmap_norm,
    )

    fig.colorbar(rho_xz_plot, label=r"$\rho_{e^{-}}\;(\mathrm{m}^3)$", aspect=10)
    save_or_show(fig, plot_target)


def get_radial_means(rho_xy_slice, centroid):
    """
    Estimate the mean of the given slice in the centroid region.
    """
    centroid_x, centroid_y = centroid
    ny, nx = rho_xy_slice.shape

    dx = X_SIZE / nx
    dy = Y_SIZE / ny
    distance_from_centroid = (
        ((dy * (np.arange(ny) - centroid_y)) ** 2)[:, np.newaxis]
        + (dx * (np.arange(nx) - centroid_x)) ** 2
    ) ** 0.5

    radial_mean = np.average(distance_from_centroid, weights=rho_xy_slice)
    radial_std = (
        np.average(
            (distance_from_centroid - radial_mean) ** 2,
            weights=rho_xy_slice,
        )
        ** 0.5
    )

    filtered_distances_from_centroid = distance_from_centroid <= 2 * radial_mean
    constrained_radial_mean = rho_xy_slice[filtered_distances_from_centroid].mean()

    return radial_mean, radial_std, constrained_radial_mean


def coord_to_position(x, nx, lx):
    """
    Translate from a 0-indexed integer coordinate
    to position on a grid of width lx and with nx points,
    centered on the origin.
    """
    return (x - nx / 2) * (lx / nx)


def process_file(h5file, plot_target=None):
    """
    Given a single HDF5 file,
    compute the quantities of interest and plot slices of interest.
    """
    step_index = get_step_index(h5file)
    rho = -h5file[f"data/{step_index}/fields/rho"][:] / constants.e
    nz, ny, nx = rho.shape

    middle_rho_xy_slice = get_middle_rho_slice(rho)
    centroid_y, centroid_x = get_centroid(middle_rho_xy_slice)
    max_rho_y, max_rho_x = np.unravel_index(
        middle_rho_xy_slice.argmax(),
        middle_rho_xy_slice.shape,
    )
    centroid_rho_xz_slice = rho[:, round(centroid_y), :]

    # Extract some interesting parameters - density
    # Probably better to do within e.g. 2r of centre
    # rather than entire volume
    max_planar_rho = middle_rho_xy_slice.max()
    mean_planar_rho = middle_rho_xy_slice.mean()

    if plot_target != "/dev/null":
        plot_slices(
            middle_rho_xy_slice,
            centroid_rho_xz_slice,
            step_index,
            centroid_y,
            plot_target,
        )

    radial_mean, radial_std, constrained_radial_mean = get_radial_means(
        middle_rho_xy_slice, (centroid_x, centroid_y)
    )

    return ResultTuple(
        step_index * STEP_INCREMENT_NS,
        coord_to_position(max_rho_x, nx, X_SIZE),
        coord_to_position(max_rho_y, ny, Y_SIZE),
        coord_to_position(centroid_x, nx, X_SIZE),
        coord_to_position(centroid_y, ny, Y_SIZE),
        radial_mean,
        radial_std,
        max_planar_rho,
        mean_planar_rho,
        constrained_radial_mean,
    )


def get_result(results, key):
    """
    Return a single key from a list of result tuples.
    """
    return [getattr(result, key) for result in results]


def plot_position(results, plot_filename):
    """
    Plot the distribution centre position for each time-step,
    for two estimation techniques.
    """
    fig, ax = plt.subplots(layout="constrained")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("Plasma positions")
    ax.set_xlim(-X_SIZE / 2, X_SIZE / 2)
    ax.set_ylim(-Y_SIZE / 2, Y_SIZE / 2)

    axis_colour = ax.spines["bottom"].get_edgecolor()
    ax.axhline(0, color=axis_colour)
    ax.axvline(0, color=axis_colour)

    ax.scatter(
        get_result(results, "max_rho_x"),
        get_result(results, "max_rho_y"),
        label="Maximal density",
    )
    ax.scatter(
        get_result(results, "centroid_x"),
        get_result(results, "centroid_y"),
        label="Centre of mass",
        marker="s",
    )

    ax.legend(loc="best", title="``Centre'' technique")

    save_or_show(fig, plot_filename)
    plt.close(fig)


def plot_radius(results, plot_filename):
    """
    Plot the estimated mean radius of the plasma for each time-step.
    """
    fig, ax = plt.subplots(layout="constrained")

    ax.set_xlabel("TOF (µs)")
    ax.set_ylabel("Plasma weighted mean radius (mm)")

    ax.scatter(get_result(results, "time"), get_result(results, "radial_mean"))
    _, ymax = ax.get_ylim()
    ax.set_ylim(0, ymax * 1.2)
    ax.set_xlim(0, None)

    save_or_show(fig, plot_filename)
    plt.close(fig)


def plot_density(results, plot_filename):
    """
    Plot the plasma density as a function of time-step.
    """
    fig, ax = plt.subplots(layout="constrained")

    ax.set_yscale("log")
    ax.set_xlabel("TOF (µs)")
    ax.set_ylabel(r"Plasma density ($10^{13}\mathrm{m}^{-3}$)")

    for key, label, marker in [
        ("max_planar_rho", "Maximum rho", "o"),
        ("mean_planar_rho", "Mean rho weighted by distance from centroid", "s"),
        ("constrained_radial_mean", "Mean rho within twice distribution width", "^"),
    ]:
        ax.scatter(
            get_result(results, "time"),
            [result / 1e13 for result in get_result(results, key)],
            label=label,
            marker=marker,
        )

    ax.legend(loc="best", title="``Technique''")
    save_or_show(fig, plot_filename)
    plt.close(fig)


def process_file_wrapper(filename, plot_directory=None, plot_styles=None):
    """
    Wrapper for `process_file` that additionally handles the setup,
    such that it may be parallelised with `ProcessPool.map()`.
    """
    basename = pathlib.Path(filename).name
    if plot_styles:
        plt.style.use(plot_styles)

    with h5py.File(filename, "r") as h5file:
        return process_file(
            h5file,
            str(f"{plot_directory}/{basename}.pdf") if plot_directory else "/dev/null",
        )


def process_files_serial(input_files, plots_filename, **kwargs):
    """
    Loop over a set of input files serially,
    allowing slice plots to be shown on screen or appended to a PdfPages object.
    """
    if plots_filename:
        plot_context = PdfPages(plots_filename)
    else:
        plot_context = contextlib.nullcontext()

    results = []
    with plot_context as plot_target:
        for filename in sorted(input_files):
            with h5py.File(filename, "r") as h5file:
                results.append(process_file(h5file, plot_target))

    return results


def merge_pdfs(input_directory, output_filename):
    """
    Given a directory containing one or more PDF files,
    concatenate them into a single PDF.
    """
    merger = pypdf.PdfWriter()
    for filename in sorted(glob.glob(f"{input_directory}/*.pdf")):
        merger.append(filename)
    merger.write(output_filename)


def process_files_parallel(input_files, plots_filename, plot_styles):
    """
    Loop over a set of input files using a ProcessPool.
    Slice plots will either be written to disk, or suppressed entirely.

    Since PdfPages can't be shared among a ProcessPool,
    each slice is output to a separate file and concatenated after the fact.
    """
    if plots_filename is not None and plots_filename != "/dev/null":
        target_directory_context = tempfile.TemporaryDirectory()
    else:
        target_directory_context = contextlib.nullcontext()

    with target_directory_context as target_directory:
        with multiprocessing.Pool() as pool:
            # While `target_directory` and `plot_styles`
            # are the same for ecah iteration,
            # they must be passed separately,
            # as `ProcessPool` cannot accept
            # lambdas, closures, or other functions const
            results = pool.starmap(
                process_file_wrapper,
                [
                    (filename, target_directory, plot_styles)
                    for filename in sorted(input_files)
                ],
            )

        if target_directory:
            merge_pdfs(target_directory, plots_filename)

    return results


def main():
    args = get_args()
    if args.plot_styles:
        plt.style.use(args.plot_styles)

    results = {True: process_files_parallel, False: process_files_serial}[
        args.parallel
    ](args.input_files, args.frame_plots_filename, plot_styles=args.plot_styles)

    plot_position(results, args.position_plot_filename)
    plot_radius(results, args.radius_plot_filename)
    plot_density(results, args.density_plot_filename)


if __name__ == "__main__":
    main()
