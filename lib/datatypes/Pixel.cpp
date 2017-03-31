//******************************************************************************
//* File:   Pixel.cpp
//* Author: Jon Newman <jpnewman snail mit dot edu>
//*
//* Copyright (c) Jon Newman (jpnewman snail mit dot edu)
//* All right reserved.
//* This file is part of the Oat project.
//* This is free software: you can redistribute it and/or modify
//* it under the terms of the GNU General Public License as published by
//* the Free Software Foundation, either version 3 of the License, or
//* (at your option) any later version.
//* This software is distributed in the hope that it will be useful,
//* but WITHOUT ANY WARRANTY; without even the implied warranty of
//* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//* GNU General Public License for more details.
//* You should have received a copy of the GNU General Public License
//* along with this source code.  If not, see <http://www.gnu.org/licenses/>.
//******************************************************************************

#include "Pixel.h"

namespace oat {

constexpr size_t Pixel::to_depth[4];
constexpr int Pixel::to_imread_code[4];
constexpr size_t Pixel::to_bytes[4];
constexpr int Pixel::to_cvtype[4];
constexpr int Pixel::color_conv_table[4][4];

} /* namespace oat */
