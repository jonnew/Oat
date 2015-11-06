//******************************************************************************
//* File:   Sink_test.cpp
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

#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main()
//#include "catch.hpp"
#include "/home/jon/Public/Oat/debug/catch/src/catch/include/catch.hpp"

#include "../src/Sink.h"

SCENARIO ("Sinks are un-copyable and can bind a single Node.", "[Sink]") {

    GIVEN ("Two fresh Sinks") {

        std::string addr = "test";
        oat::Sink<int> sink1;
        oat::Sink<int> sink2;

//        WHEN ("A copy of the sink is made") {
//
//            THEN ( "Compilation shall fail.") {
//                oat::Sink<int> sink2 = sink1;
//            }
//        }

        WHEN ("One sink binds a shmem segment") {

            sink1.bind(addr);

            THEN ("Attempting to bind that segment by the second sink shall throw") {
                REQUIRE_THROWS(
                    sink2.bind(addr);
                );
            }
        }

        WHEN ("A sink calls wait() or post() before binding a segment") {

            sink1.bind(addr);

            THEN ("The the sink shall throw") {
                REQUIRE_THROWS(
                    sink1.wait();
                );

                REQUIRE_THROWS(
                    sink1.post();
                );

            }
        }
    }
}
