//******************************************************************************
//* File:   Format.cpp
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
//*****************************************************************************

#include <cassert>

#include "Format.h"

namespace oat {

/**
 * @brief Get a string containing a numpy file header for a given numpy dtype
 * specification. The number of data points that can be writen this file is
 * Format::MAX_NPY_ENTRIES.
 *
 * @param dtype_str Valid numpy.dtype.str.
 *
 * @return .npy file header.
 */
std::vector<char> getNumpyHeader(const std::string &dtype_str)
{
    // Header prefix: magic number, npy version, header length
    // position 8 and 9 reserved for header length
    std::vector<char> header;
    header.push_back((char) 0x93);
    append(header, std::string("NUMPY"));
    header.push_back((char) 0x01);
    header.push_back((char) 0x00);

    // Create numpy file header. Shape needs to be editted in destructor with
    // knowledge of the number of records written
    std::vector<char> dict;
    append(dict,
           std::string(
               "{"
               "'shape': (0000000000, ), " // 9.9 billion records max
               "'fortran_order': False, "
               "'descr': "));

    append(dict, dtype_str);
    dict.push_back('}');

    // Dict padding to 16-byte boundary
    int remainder = 16 - ((dict.size() + NPY_PREFIX_LEN) % 16);
    dict.insert(dict.end(), remainder,' ');
    dict.back() = '\n';

    // Append dictionary size including padding to header prefix
    auto dict_len = static_cast<uint16_t>(dict.size());
    header.push_back((dict_len >> 0) & 0xFF);
    header.push_back((dict_len >> 8) & 0xFF);

    // Append dict to header
    header.insert(std::end(header), std::begin(dict), std::end(dict));
    assert(header.size() % 16 == 0);

    return header;
}

void emplaceNumpyShape(FILE *fd, int64_t n)
{
    auto n_str = std::to_string(n);
    int pad_len = 10 - n_str.size();
    if (pad_len < 0) {
        std::cerr << "Maximum record size exceeded. Npy file header needs to "
                     "be mannually editied to recover data.";
        return;
    }

    std::string shape = "'shape': ";
    shape.append(pad_len, ' ');
    shape.append("(");
    shape.append(n_str);
    shape.append(", ), ");

    fseek(fd, NPY_DICT_START_BYTE, 0);
    fwrite(shape.data(), 1, shape.size(), fd);
}

} /* namespace oat */
