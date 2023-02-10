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

import argparse

import numpy as np

from ..input_params import SampleParams


def generate_scan_pattern(
    numx: int,
    numy: int,
    ratio_initial_scan: float = 0.1,
    num_scan_points: int = None,
    save: bool = False,
    verbose=True,
):
    if num_scan_points is not None:
        if verbose:
            print("Numer of scan points is provided. This overrides ratio_initial_scan.")
        ratio_initial_scan = None
    sample_params = SampleParams(
        image_shape=(numy, numx),
        initial_scan_points_num=num_scan_points,
        initial_scan_ratio=ratio_initial_scan,
        stop_ratio=0.3,
        random_seed=11,
    )
    num_scan_points = np.shape(sample_params.initial_idxs)[0]
    if verbose:
        print("Initial ratio is", num_scan_points / sample_params.image_size)
    if save:
        np.savetxt(
            f"initial_points_{numx}_{numy}_points_{num_scan_points}.csv",
            sample_params.initial_idxs,
            delimiter=",",
            fmt="%10d",
        )
    return sample_params.initial_idxs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate initial scanning structure")
    parser.add_argument("-x", "--numx", type=int, help="number of x points", required=True)
    parser.add_argument("-y", "--numy", type=int, help="number of y points", required=True)
    parser.add_argument("-s", "--nums", type=int, help="number of initial scan points", required=True)
    args = parser.parse_args()

    generate_scan_pattern(args.numx, args.numy, num_scan_points=args.nums, save=True)
