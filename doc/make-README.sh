#!/bin/bash

#******************************************************************************
#* File:   make-README.sh
#* Author: Jon Newman <jpnewman snail mit dot edu>
#*
#* Copyright (c) Jon Newman (jpnewman snail mit dot edu)
#* All right reserved.
#* This file is part of the Oat project.
#* This is free software: you can redistribute it and/or modify
#* it under the terms of the GNU General Public License as published by
#* the Free Software Foundation, either version 3 of the License, or
#* (at your option) any later version.
#* This software is distributed in the hope that it will be useful,
#* but WITHOUT ANY WARRANTY; without even the implied warranty of
#* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#* GNU General Public License for more details.
#* You should have received a copy of the GNU General Public License
#* along with this source code.  If not, see <http://www.gnu.org/licenses/>.
#******************************************************************************

pc() 
{
    pc_res=`echo "$1" | awk '/ mykey/{y=1;next}y'`
}

# oat-frameserve type configurations
pc "$(oat frameserve gige --help)" 
ofs_g="$pc_res"
pc "$(oat frameserve wcam --help)" 
ofs_w="$pc_res"
pc "$(oat frameserve file --help)" 
ofs_f="$pc_res"
pc "$(oat frameserve test --help)" 
ofs_t="$pc_res"

# oat-framefilt type configurations
pc "$(oat framefilt bsub --help)" 
off_b="$pc_res"
pc "$(oat framefilt mask --help)" 
off_ma="$pc_res"
pc "$(oat framefilt mog --help)" 
off_mo="$pc_res"
pc "$(oat framefilt undistort --help)" 
off_u="$pc_res"
pc "$(oat framefilt thresh --help)" 
off_t="$pc_res"

# oat-view type configurations
pc "$(oat view frame --help)" 
ovi_f="$pc_res"

# oat-posidet type configurations
pc "$(oat posidet board --help)" 
opd_b="$pc_res"
pc "$(oat posidet diff --help)" 
opd_d="$pc_res"
pc "$(oat posidet hsv --help)" 
opd_h="$pc_res"
pc "$(oat posidet thrsh --help)" 
opd_t="$pc_res"

# oat-posigen type configurations
pc "$(oat posigen rand2D --help)" 
opg_r2="$pc_res"

# oat-posifilt type configurations
pc "$(oat posifilt kalman --help)" 
opf_k="$pc_res"
pc "$(oat posifilt homography --help)" 
opf_h="$pc_res"
pc "$(oat posifilt region --help)" 
opf_r="$pc_res"

# oat-posicom configurations
pc "$(oat posicom mean --help)" 
opc_m="$pc_res"

# oat-posisck configurations
pc "$(oat posisock std --help)" 
ops_s="$pc_res"
pc "$(oat posisock pub --help)" 
ops_p="$pc_res"
pc "$(oat posisock rep --help)" 
ops_r="$pc_res"
pc "$(oat posisock udp --help)" 
ops_u="$pc_res"

# oat-calibrate configurations
pc "$(oat calibrate camera --help)" 
oca_c="$pc_res"
pc "$(oat calibrate homography --help)" 
oca_h="$pc_res"

# Semi-automated README.md and README.pdf construction
awk -v ofs="$(oat frameserve --help)" \
    -v ofs_g="$ofs_g" \
    -v ofs_w="$ofs_w" \
    -v ofs_f="$ofs_f" \
    -v ofs_t="$ofs_t" \
    -v off="$(oat framefilt --help)" \
    -v off_b="$off_b" \
    -v off_ma="$off_ma" \
    -v off_mo="$off_mo" \
    -v off_u="$off_u" \
    -v off_t="$off_t" \
    -v ovi="$(oat view --help)"      \
    -v ovi_f="$ovi_f" \
    -v opd="$(oat posidet --help)"   \
    -v opd_b="$opd_b" \
    -v opd_d="$opd_d" \
    -v opd_h="$opd_h" \
    -v opd_t="$opd_t" \
    -v opg="$(oat posigen --help)"   \
    -v opg_r2="$opg_r2" \
    -v opf="$(oat posifilt --help)"  \
    -v opf_k="$opf_k" \
    -v opf_h="$opf_h" \
    -v opf_r="$opf_r" \
    -v opc="$(oat posicom --help)"  \
    -v opc_m="$opc_m" \
    -v ode="$(oat decorate --help)"  \
    -v ore="$(oat record --help)"  \
    -v ops="$(oat posisock --help)"  \
    -v ops_s="$ops_s" \
    -v ops_p="$ops_p" \
    -v ops_r="$ops_r" \
    -v ops_u="$ops_u" \
    -v obu="$(oat buffer --help)"  \
    -v ocl="$(oat clean --help)"  \
    -v oca="$(oat calibrate --help)"  \
    -v oca_c="$oca_c" \
    -v oca_h="$oca_h" \
'{
    sub(/oat-frameserve-help/, ofs);
    sub(/oat-frameserve-gige-help/, ofs_g);
    sub(/oat-frameserve-wcam-help/, ofs_w);
    sub(/oat-frameserve-file-help/, ofs_f);
    sub(/oat-frameserve-test-help/, ofs_t);
    sub(/oat-framefilt-help/, off);
    sub(/oat-framefilt-bsub-help/, off_b);
    sub(/oat-framefilt-mask-help/, off_ma);
    sub(/oat-framefilt-mog-help/, off_mo);
    sub(/oat-framefilt-undistort-help/, off_u);
    sub(/oat-framefilt-thresh-help/, off_t);
    sub(/oat-view-help/, ovi);
    sub(/oat-view-frame-help/, ovi_f);
    sub(/oat-posidet-help/, opd);
    sub(/oat-posidet-board-help/, opd_b);
    sub(/oat-posidet-diff-help/, opd_d);
    sub(/oat-posidet-hsv-help/, opd_h);
    sub(/oat-posidet-thresh-help/, opd_t);
    sub(/oat-posigen-help/, opg);
    sub(/oat-posigen-rand2D-help/, opg_r2);
    sub(/oat-posifilt-help/, opf);
    sub(/oat-posifilt-kalman-help/, opf_k);
    sub(/oat-posifilt-homography-help/, opf_h);
    sub(/oat-posifilt-region-help/, opf_r);
    sub(/oat-posicom-help/, opc);
    sub(/oat-posicom-mean-help/, opc_m);
    sub(/oat-decorate-help/, ode);
    sub(/oat-record-help/, ore);
    sub(/oat-posisock-help/, ops);
    sub(/oat-posisock-std-help/, ops_s);
    sub(/oat-posisock-pub-help/, ops_p);
    sub(/oat-posisock-rep-help/, ops_r);
    sub(/oat-posisock-udp-help/, ops_u);
    sub(/oat-buffer-help/, obu);
    sub(/oat-clean-help/, ocl);
    sub(/oat-calibrate-help/, oca);
    sub(/oat-calibrate-camera-help/, oca_c);
    sub(/oat-calibrate-homography-help/, oca_h);
    print;
}' ./README-template.md > ./README.md

pandoc README.md -o README.pdf
cp README.md README.pdf ../
rm ./README.md
rm ./README.pdf
