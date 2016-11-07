//******************************************************************************
//* File:   Position2D.cpp
//* Author: Jon Newman <jpnewman snail mit dot edu>
//*
//* Copyright (c) Jon Newman (jpnewman snail mit dot edu)
//* All right reserved.
//* This file is part of the Oat projecw->
//* This is free software: you can redistribute it and/or modify
//* it under the terms of the GNU General Public License as published by
//* the Free Software Foundation, either version 3 of the License, or
//* (at your option) any later version.
//* This software is distributed in the hope that it will be useful,
//* but WITHOUT ANY WARRANTY; without even the implied warranty of
//* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//* GNU General Public License for more details.
//* You should have received a copy of the GNU General Public License
//* along with this source code.  If not, see <http://www->gnu.org/licenses/>.
//******************************************************************************

#include "Position2D.h"

namespace oat {

const char Position2D::NPY_DTYPE[]{"[('tick', '<u8'),"
                                    "('usec', '<u8'),"
                                    "('unit', '<i4'),"
                                    "('pos_ok', '<i1'),"
                                    "('pos_xy', 'f8', (2)),"
                                    "('vel_ok', '<i1'),"
                                    "('vel_xy', 'f8', (2)),"
                                    "('head_ok', '<i1'),"
                                    "('head_xy', 'f8', (2)),"
                                    "('reg_ok', '<i1'),"
                                    "('reg', 'a10')]"};

// TODO: This feels horrible...
std::vector<char> packPosition(const Position2D &p)
{
    std::vector<char> pack;
    pack.reserve(oat::Position2D::NPY_DTYPE_BYTES);

    auto sc = p.sample_.count();
    auto val = reinterpret_cast<char*>(&sc);
    pack.insert(pack.end(), val, val + sizeof (sc));

    auto su = p.sample_usec();
    val = reinterpret_cast<char*>(&su);
    pack.insert(pack.end(), val, val + sizeof (su));

    auto u = static_cast<int>(p.unit_of_length_);
    val = reinterpret_cast<char*>(&u);
    pack.insert(pack.end(), val, val + sizeof (u));

    // Position
    char pok = p.position_valid ? 1 : 0;
    pack.insert(pack.end(), &pok, &pok + 1);

    auto px = p.position.x;
    val = reinterpret_cast<char*>(&px);
    pack.insert(pack.end(), val, val + sizeof (px));

    auto py = p.position.y;
    val = reinterpret_cast<char*>(&py);
    pack.insert(pack.end(), val, val + sizeof (px));

    // Velocity
    char vok = p.velocity_valid ? 1 : 0;
    pack.insert(pack.end(), &vok, &vok + 1);

    auto vx = p.velocity.x;
    val = reinterpret_cast<char*>(&vx);
    pack.insert(pack.end(), val, val + sizeof (vx));

    auto vy = p.velocity.y;
    val = reinterpret_cast<char*>(&vy);
    pack.insert(pack.end(), val, val + sizeof (vx));

    // Heading
    char hok = p.heading_valid ? 1 : 0;
    pack.insert(pack.end(), &hok, &hok + 1);

    auto hx = p.heading.x;
    val = reinterpret_cast<char*>(&hx);
    pack.insert(pack.end(), val, val + sizeof (hx));

    auto hy = p.heading.y;
    val = reinterpret_cast<char*>(&hy);
    pack.insert(pack.end(), val, val + sizeof (hx));

    // Region
    char rok = p.region_valid ? 1 : 0;
    pack.insert(pack.end(), &rok, &rok + 1);
    pack.insert(pack.end(), p.region, p.region + oat::Position2D::REGION_LEN);

    return pack;
}

} /* namespace oat */
