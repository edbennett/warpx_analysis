#!/usr/bin/env python3

from argparse import ArgumentParser
import contextlib
import functools
import logging

import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy import constants


STEP_INCREMENT_NS = 0.2
X_SIZE = 0.12
Y_SIZE = 0.12
Z_SIZE = 0.12
OUTER_CIRCLE_RADIUS = 0.051
INNER_CIRCLE_RADIUS = 0.01
ORIGIN = (0, 0)
INNER_CIRCLE_POSITION = (0.04, 0)


def get_args():
    parser = ArgumentParser(description="Analyse WarpX output without Mathematica")
    parser.add_argument("input_files", nargs="+", metavar="input_file", help="HDF5 filename")
    parser.add_argument("--plot_filename", default=None, help="Where to output plots of density slices")
    parser.add_argument("--plot_styles", default=None, help="Style sheet for plots")
    return parser.parse_args()


def get_step_index(h5file):
    step_index_str, = h5file["data"].keys()
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
            np.dot(dim_slice, np.arange(dimension_size))
            / sum_rho_slice
        )

    return tuple(centroid_position)


def get_middle_rho_slice(rho):
    """
    Extract XY central plane data.
    """
    middle_slice_index = rho.shape[0] // 2 + 1
    return rho[middle_slice_index]


def plot_rho_slice(ax, rho_slice, title, extent, **params):
    ax.set_xlabel("$x$")
    ax.set_title(title)
    return ax.imshow(
        rho_slice,
        origin="lower",
        extent=extent,
        interpolation="gaussian",
        cmap="magma",
        **params
    )


def save_or_show(fig, plot_target):
    if plot_target:
        if isinstance(plot_target, str):
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
    cmap_norm = get_cmap_norm(rho_xy_slice, rho_xz_slice)

    # xy plot
    fig, (xy_ax, xz_ax) = plt.subplots(
        ncols=2,
        layout="constrained",
        sharey=True,
        figsize=(4.5, 2),
    )
    xy_ax.set_ylabel("$y$")

    plot_rho_slice(
        xy_ax,
        rho_xy_slice,
        title=f"Step No. {step_index} – {STEP_INCREMENT_NS * step_index} ns",
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
    slice_position = (
        rounded_y - rho_xz_slice.shape[1] / 2
    ) * Y_SIZE / rho_xz_slice.shape[1]

    rho_xz_plot = plot_rho_slice(
        xz_ax,
        rho_xz_slice,
        title=f"$x = {slice_position} \\mathrm{{m}}$, cell = {rounded_y}",
        extent=(-Z_SIZE / 2, Z_SIZE / 2, -X_SIZE / 2, X_SIZE / 2),
        **cmap_norm,
    )

    fig.colorbar(rho_xz_plot, label=r"$\rho_{e^{-}}\;(\mathrm{m}^3)$", aspect=10)
    save_or_show(fig, plot_target)


def get_radial_means(rho_xy_slice, centroid):
    centroid_x, centroid_y = centroid
    ny, nx = rho_xy_slice.shape

    dx = X_SIZE / nx
    dy = Y_SIZE / ny
    distance_from_centroid = (
        ((dy * (np.arange(ny) - centroid_y)) ** 2)[:, np.newaxis]
        + (dx * (np.arange(nx) - centroid_x)) ** 2
    ) ** 0.5

    radial_mean = np.average(distance_from_centroid, weights=rho_xy_slice)
    radial_std = np.average(
        (distance_from_centroid - radial_mean) ** 2,
        weights=rho_xy_slice,
    ) ** 0.5

    filtered_distances_from_centroid = distance_from_centroid <= 2 * radial_mean
    constrained_radial_mean = np.average(
        distance_from_centroid[filtered_distances_from_centroid],
        weights=rho_xy_slice[filtered_distances_from_centroid],
    )

    return radial_mean, radial_std, constrained_radial_mean


def process_file(h5file, plot_target=None):
    step_index = get_step_index(h5file)
    rho = -h5file[f"data/{step_index}/fields/rho"][:] / constants.e
    nz, ny, nx = rho.shape
    dx = X_SIZE / nx
    dy = Y_SIZE / ny

    middle_rho_xy_slice = get_middle_rho_slice(rho)
    centroid_x, centroid_y = get_centroid(middle_rho_xy_slice)
    centroid_rho_xz_slice = rho[:, round(centroid_y), :]

    # Extract some interesting parameters - density
    # Probably better to do within e.g. 2r of centre
    # rather than entire volume
    max_planar_rho = middle_rho_xy_slice.max()
    mean_planar_rho = middle_rho_xy_slice.mean()

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

    return(
        step_index * STEP_INCREMENT_NS,
        (centroid_y - ny) * dy,
        (centroid_x - nx) * dx,
        (centroid_x, centroid_y),
        radial_mean,
        radial_std,
        max_planar_rho,
        mean_planar_rho,
        constrained_radial_mean,
    )


def main():
    args = get_args()
    if args.plot_styles:
        plt.style.use(args.plot_styles)

    if args.plot_filename:
        plot_context = PdfPages(args.plot_filename)
    else:
        plot_context = contextlib.nullcontext()

    with plot_context as plot_target:
        for filename in args.input_files:
            with h5py.File(filename, "r") as h5file:
                process_file(h5file, plot_target)


if __name__ == "__main__":
    main()
