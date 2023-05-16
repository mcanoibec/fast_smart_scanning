# ----------------------------------------------------------------------- #
# Copyright (c) 2023, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Software Name:    Fast Autonomous Scanning Toolkit (FAST)               #
# By: Argonne National Laboratory                                         #
# OPEN SOURCE LICENSE                                                     #
#                                                                         #
# Redistribution and use in source and binary forms, with or without      #
# modification, are permitted provided that the following conditions      #
# are met:                                                                #
#                                                                         #
# 1. Redistributions of source code must retain the above copyright       #
#    notice, this list of conditions and the following disclaimer.        #
#                                                                         #
# 2. Redistributions in binary form must reproduce the above copyright    #
#    notice, this list of conditions and the following disclaimer in      #
#    the documentation and/or other materials provided with the           #
#    distribution.                                                        #
#                                                                         #
# 3. Neither the name of the copyright holder nor the names of its        #
#    contributors may be used to endorse or promote products derived from #
#    this software without specific prior written permission.             #
#                                                                         #
# *********************************************************************** #
#                                                                         #
# DISCLAIMER                                                              #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE          #
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,    #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS   #
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED      #
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,  #
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF   #
# THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY            #
# OF SUCH DAMAGE.                                                         #
# *********************************************************************** #
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ..core.base import ExperimentalSample
from ..core.erd import SladsSklearnModel
from ..core.measurement_interface import ExternalMeasurementInterface
from ..input_params import ERDInputParams, GeneralInputParams, SampleParams
from ..utils.readMDA import readMDA

try:
    import hdf5plugin  # isort:skip
    import h5py  # isort:skip
except ImportError:
    print(
        "Cannot find hdf5plugin and h5py packages. Cannot run the read_mda_h5_chunks_custom_roi function. "
        "Since this function has a very specific use case, this error can generally be ignored."
    )


def create_experiment_sample(
    numx: int,
    numy: int,
    initial_idxs: list,
    inner_batch_size: int = 100,
    stop_ratio: float = 0.5,
    c_value: int = 2,
    full_erd_recalculation_frequency: int = 1,
    affected_neighbors_window_min: int = 5,
    affected_neighbors_window_max: int = 15,
    num_neighbors=None,
    erd_model_file_path: str = None,
) -> ExperimentalSample:
    if erd_model_file_path is None:
        erd_model_file_path = (
            Path(__file__).parent.parent.parent
            / "work_directory/training/cameraman/"
            / f"c_{c_value}/erd_model_relu.pkl"
        )
    erd_model = SladsSklearnModel(load_path=erd_model_file_path)

    params_erd = ERDInputParams(
        c_value=c_value,
        full_erd_recalculation_frequency=full_erd_recalculation_frequency,
        affected_neighbors_window_min=affected_neighbors_window_min,
        affected_neighbors_window_max=affected_neighbors_window_max,
    )
    if num_neighbors is not None:
        params_gen = GeneralInputParams(num_neighbors=num_neighbors, neighbor_weight_distance_norm=2)
    else:
        params_gen = GeneralInputParams()

    sample_params = SampleParams(
        image_shape=(numy, numx),
        inner_batch_size=inner_batch_size,
        initial_idxs=initial_idxs,
        stop_ratio=stop_ratio,
        random_seed=11,
    )

    measurement_interface = ExternalMeasurementInterface()

    sample = ExperimentalSample(
        sample_params=sample_params,
        general_params=params_gen,
        erd_params=params_erd,
        measurement_interface=measurement_interface,
        erd_model=erd_model,
    )
    return sample


def setdiff2d(A: npt.NDArray, B: npt.NDArray):
    nrows, ncols = A.shape
    dtype = {
        "names": ["f{}".format(i) for i in range(ncols)],
        "formats": ncols * [A.dtype],
    }
    C = np.setdiff1d(A.copy().view(dtype), B.copy().view(dtype))
    D = np.setdiff1d(B.copy().view(dtype), A.copy().view(dtype))
    return C, D


def read_mda_chunks(
    mdas: list, x_scaling: float, y_scaling: float, round_to_int: bool = True
) -> tuple[list, list, list, list]:
    xpoints = []
    ypoints = []
    intensities = []
    idxs_all = []
    xmin = 9999999
    ymin = 9999999

    for ix, mdaname in enumerate(mdas):
        mda = readMDA(mdaname, verbose=False)

        xvals0 = np.array(mda[1].p[0].data) / x_scaling
        yvals0 = np.array(mda[1].p[1].data) / y_scaling
        if round_to_int:
            xvals0 = np.round(xvals0, 0).astype("int")
            yvals0 = np.round(yvals0, 0).astype("int")

        ints0 = np.array(mda[1].d[3].data)
        # print('d3', ints0[0])
        # ints0 = np.array(mda[1].d[6].data)
        # print('d6',ints0[0])

        # need to do this shift to get the correct MDA positions
        xvals = xvals0[:-1]
        yvals = yvals0[:-1]
        # ints = ints0[1:]
        ints = ints0[1:]

        idxs = np.array((yvals, xvals)).T

        xmin = np.minimum(xmin, xvals[np.abs(xvals) > 0].min())
        ymin = np.minimum(ymin, yvals[np.abs(yvals) > 0].min())
        # print('xmin', xmin, 'ymin', ymin)

        xpoints.append(xvals)
        ypoints.append(yvals)
        intensities.append(ints)

        idxs_all.append(idxs)

    xpoints = [xp - xmin for xp in xpoints]
    ypoints = [yp - ymin for yp in ypoints]
    idxs_all = [idxs - [ymin, xmin] for idxs in idxs_all]
    return xpoints, ypoints, idxs_all, intensities


def blind_scale_positions(pos1d: npt.NDArray, expected_size: int):
    pos = (pos1d - pos1d.min()) / (pos1d.max() - pos1d.min()) * (expected_size - 1)
    return pos


def read_mda_h5_chunks_custom_roi(
    mdafiles: list,
    h5files: list,
    x_scaling: float,
    y_scaling: float,
    roi: list = None,
    round_to_int: bool = False,
):
    if roi is None:
        roi = [[810, 840], [388, 418]]
    xpoints = []
    ypoints = []
    intensities = []
    xpoints_min = 0
    ypoints_min = 0
    for mdafile, h5file in zip(mdafiles, h5files):
        mda = readMDA(mdafile, verbose=False)
        with h5py.File(h5file) as hf:
            h5data = hf["entry/data/data"][..., roi[0][0] : roi[0][1], roi[1][0] : roi[1][1]]
            h5data[h5data > 1e8] = 0

        intensities_this = h5data.sum(axis=(-1, -2))
        norm_counts = np.array(mda[1].d[0].data)
        intensities_normed = intensities_this.reshape(norm_counts.shape) / norm_counts

        xpoints.append(mda[1].p[0].data)
        ypoints.append(mda[1].p[1].data)
        intensities.append(intensities_normed)
        xpoints_min = np.minimum(xpoints_min, np.min(xpoints[-1]))
        ypoints_min = np.minimum(ypoints_min, np.min(ypoints[-1]))

    xpoints_fin = []
    ypoints_fin = []
    idxs_all = []
    for xvals, yvals in zip(xpoints, ypoints):
        xps = (np.array(xvals) - xpoints_min) / x_scaling
        yps = (np.array(yvals) - ypoints_min) / y_scaling
        if round_to_int:
            xps = np.round(xps).astype("int")
            yps = np.round(yps).astype("int")
        idxs = np.array([yps, xps]).T
        xpoints_fin.append(xps)
        ypoints_fin.append(yps)
        idxs_all.append(idxs)
    return xpoints_fin, ypoints_fin, idxs_all, intensities
