//******************************************************************************
//* File:   Source_test.cpp
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
#include "catch.hpp"
#include "/home/jon/Public/Oat/debug/catch/src/catch/include/catch.hpp"

#include <string>

#include "../src/Source.h"
#include "../src/Sink.h"
#include "../src/SharedCVMat.h"

SCENARIO ("Up to 10 sources can connect a single Node.", "[Source]") {

    GIVEN ("11 sources and a bound sink with common node address") {

        std::string addr {"test"};
        oat::Sink<int> sink;

        INFO ("The sink binds a node");
        sink.bind(addr);
        oat::Source<int> s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10;

        WHEN ("sources 0-10 connect a node") {

            THEN ("The first 10 connections will succeed") {
                REQUIRE_NOTHROW(
                    s0.connect(addr);
                    s1.connect(addr);
                    s2.connect(addr);
                    s3.connect(addr);
                    s4.connect(addr);
                    s5.connect(addr);
                    s6.connect(addr);
                    s7.connect(addr);
                    s8.connect(addr);
                    s9.connect(addr);
                );
            }

            AND_THEN ("The 11th shall throw") {
                REQUIRE_THROWS(
                    s0.connect(addr);
                    s1.connect(addr);
                    s2.connect(addr);
                    s3.connect(addr);
                    s4.connect(addr);
                    s5.connect(addr);
                    s6.connect(addr);
                    s7.connect(addr);
                    s8.connect(addr);
                    s9.connect(addr);
                    s10.connect(addr);
                );
            }
        }
    }
}

SCENARIO ("Sources must connect() before waiting or posting.", "[Source]") {

    GIVEN ("A single, unconnected source ") {

        std::string addr {"test"};
        oat::Source<int> source;

        WHEN ("The source calls wait() before binding a segment") {
            THEN ("The source shall throw.") {
                REQUIRE_THROWS( source.wait(); );
            }
        }

        WHEN ("The source calls post() before binding a segment") {
            THEN ("The source shall throw.") {
            REQUIRE_THROWS( source.post(); );
            }
        }
    }
}

SCENARIO ("A Source<T> can only connect to a node bound by a Sink<T>.", "[Source]") {

    GIVEN ("A bound Sink<int> and an Source<float> with common node address") {

        std::string addr {"test"};
        oat::Sink<int> sink;
        oat::Source<float> source;

        INFO ("The sink binds a node");
        sink.bind(addr);

        WHEN ("The source attempts to connect()") {
            THEN ("The source shall throw.") {
                REQUIRE_THROWS( source.connect(addr); );
            }
        }
    }
}

SCENARIO ("Connected sources can retrieve shared objects to mutate them.", "[Source]") {

    GIVEN ("A bound Sink<int> and a connected Source<int> with common node address") {

        std::string addr {"test"};
        oat::Sink<int> sink;
        oat::Source<int> source;
        int * src_ptr, * snk_ptr;

        INFO ("The sink binds a node");
        sink.bind(addr);

        WHEN ("The source calls retrieve() before connecting") {
            THEN ("The source shall throw") {
                REQUIRE_THROWS( src_ptr = source.retrieve(); );
            }
        }

        WHEN ("The source calls retrieve() after connecting") {

            INFO ("The source connects to the node");
            source.connect(addr);

            THEN ("The source returns a pointer to mutate the int across source and sink") {

                INFO ("Start with uninitialized pointers to int, src_ptr and snk_ptr")
                CAPTURE(src_ptr);
                CAPTURE(snk_ptr);

                INFO ("Set pointers using src_ptr = source.retrieve() and snk_ptr = sink.retrieve()");
                REQUIRE_NOTHROW( src_ptr = source.retrieve(); );
                REQUIRE_NOTHROW( snk_ptr = sink.retrieve(); );

                INFO ("Use the src_ptr to set shared int to 42");
                *src_ptr = 42;
                CAPTURE(*src_ptr);

                REQUIRE( *src_ptr == *snk_ptr);
                REQUIRE( *src_ptr == *source.retrieve() );
                REQUIRE( *src_ptr == *sink.retrieve() );

                INFO ("Use the snk_ptr to set shared int to -42");
                *snk_ptr = -42;
                CAPTURE(*snk_ptr);

                REQUIRE( *snk_ptr == *src_ptr);
                REQUIRE( *snk_ptr == *source.retrieve() );
                REQUIRE( *snk_ptr == *sink.retrieve() );
            }
        }
    }
}
