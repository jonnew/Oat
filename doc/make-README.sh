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

# Semi-automated README.md and README.pdf construction

awk -v ofs="$(oat frameserve --help)" \
    -v off="$(oat framefilt --help)" \
    -v ovi="$(oat view --help)"      \
    -v opd="$(oat posidet --help)"   \
    -v opg="$(oat posigen --help)"   \
    -v opf="$(oat posifilt --help)"  \
    -v opc="$(oat posicom --help)"  \
    -v ode="$(oat decorate --help)"  \
    -v ore="$(oat record --help)"  \
    -v ops="$(oat posisock --help)"  \
    -v obu="$(oat buffer --help)"  \
    -v ocl="$(oat clean --help)"  \
    -v oca="$(oat calibrate --help)"  \
'{
    sub(/oat-frameserve-help/, ofs);
    sub(/oat-framefilt-help/, off);
    sub(/oat-view-help/, ovi);
    sub(/oat-posidet-help/, opd);
    sub(/oat-posigen-help/, opg);
    sub(/oat-posifilt-help/, opf);
    sub(/oat-posicom-help/, opc);
    sub(/oat-decorate-help/, ode);
    sub(/oat-record-help/, ore);
    sub(/oat-posisock-help/, ops);
    sub(/oat-buffer-help/, obu);
    sub(/oat-clean-help/, ocl);
    sub(/oat-calibrate-help/, oca);
    print;
}' ./README-template.md > ./README.md

pandoc README.md -o README.pdf
cp README.md README.pdf ../
rm ./README.md
rm ./README.pdf
