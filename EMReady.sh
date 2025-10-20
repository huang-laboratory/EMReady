#!/bin/bash

# Copyright (C) 2024  Hong Cao, Jiahua He, Tao Li, Sheng-You Huang and Huazhong University of Science and Technology

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.



# Users should properly set the following variables before running EMReady
#######################################################################
    EMReady_home=""
    activate=""
    EMReady_env=""
#######################################################################

if [ ! -d "$EMReady_home" ]; then
    echo "ERROR: Please set 'EMReady_home' to the absolute path of the directory where EMReady is installed"
    exit 1
fi

if [ ! -f "$activate" ]; then
    echo "ERROR: Please set 'activate' to the absolute path of conda's 'activate' executable"
    exit 1
fi

. $activate $EMReady_env 2>/dev/null

if [ "$CONDA_DEFAULT_ENV" != "$EMReady_env" ]; then
    echo "ERROR: Cannot activate the conda environment '$EMReady_env'"
    echo "Please set 'EMReady_env' to the name of EMReady's conda environment"
    exit 1
fi

VERSION="1.3"
for arg in "$@"
do
    case $arg in
        --version)
        echo "EMReady version $VERSION"
        exit 0
        ;;
    esac
done

if [ $# -lt 2 ];then
	echo ""
	echo "EMReady by Huang Lab @ HUST (http://huanglab.phys.hust.edu.cn/EMReady)"
	echo ""
	echo "USAGE: `basename $0` in_map.mrc out_map.mrc [options]"
	echo ""
	echo "Descriptions:"
	echo "    in_map.mrc    : Input EM density map in MRC2014 format"
	echo "    out_map.mrc   : Filename of the output processed density map"
	echo ""
	echo "    -g            : ID(s) of GPU devices to be used, e.g., 0 for GPU #0, and 2,3,6 for GPUs #2, #3, and #6"
	echo "                  default: 0"
	echo ""
	echo "    -s            : The stride of the sliding window to cut the input map into overlapping boxes. The value should be an integer within the range of [6, 48]. A smaller stride means a larger number of overlapping boxes"
	echo "                  default: 12"
	echo ""
	echo "    -b            : Number of input boxes in one batch. Users can adjust 'batch_size' according to the available VRAM of their GPU devices. Empirically, a GPU with 40 GB VRAM can afford a 'batch_size' of about 200"
	echo "                  default: 10"
	echo ""
	echo "    -m            : Input mask map in MRC2014 format"
	echo ""
	echo "    -c            : The contour threshold to binarize the mask map"
	echo "                  default: 0.0"
	echo ""
	echo "    -p            : Input structure in PDB or mmCIF format to be used as the mask"
	echo ""
	echo "    -r            : Zone radius of the mask around the input structure in Angstrom"
	echo "                  default: 4.0"
	echo ""
	echo "    -mo           : Filename of the output binary mask map"
	echo ""
	echo "    --use_cpu     : Run EMReady on CPU instead of GPU"
	echo ""
	echo "    --inverse     : Inverse the mask"
	echo ""

    	exit 1
fi

in_map=$1
out_map=$2
mask_map=""
mask_contour=0
mask_str=""
mask_str_radius=4.0
mask_out=""
inverse_mask=""
gpu_id="0"
stride=12
batch_size=10
use_cpu=""
model_state_dict_dir=$EMReady_home"/model_state_dicts"

while [ $# -gt 2 ];do
    case $3 in
    -m)
        shift
        mask_map="-m "$3;;
    -c)
        shift
        mask_contour=$3;;
    -p)
        shift
        mask_str="-p "$3;;
    -r)
        shift
        mask_str_radius=$3;;
    -mo)
        shift
        mask_out="-mo "$3;;
    --inverse)
        inverse_mask="--inverse_mask";;
    -g)
        shift
        gpu_id=$3;;
    -b)
        shift
        batch_size=$3;;
    -s)
        shift
        stride=$3;;
    --use_cpu)
        use_cpu="--use_cpu";;
    *)
    	echo " ERROR: wrong command argument \"$3\" !!"
    	echo " Type \"$0\" for help !!"
        exit 2;;
    esac
    shift
done

python ${EMReady_home}/pred.py -i $in_map -o $out_map -md $model_state_dict_dir $mask_map -c $mask_contour $mask_str -r $mask_str_radius $mask_out $inverse_mask -g $gpu_id -b $batch_size -s $stride $use_cpu 
