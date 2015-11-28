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

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include <string>

#include "../../lib/shmemdf/Source.h"
#include "../../lib/shmemdf/Sink.h"
#include "../../lib/shmemdf/SharedFrameHeader.h"

const std::string node_addr = "test";

SCENARIO ("Up to 10 sources can connect a single Node.", "[Source]") {

    GIVEN ("11 sources and a bound sink with common node address") {

        oat::Sink<int> sink;

        INFO ("The sink binds a node");
        sink.bind(node_addr);
        oat::Source<int> s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10;

        WHEN ("sources 0 to Oat::Node:NUM_SLOTS connect a node") {

            THEN ("The first 10 connections will succeed") {
                REQUIRE_NOTHROW(
                    s0.touch(node_addr);
                    s1.touch(node_addr);
                    s2.touch(node_addr);
                    s3.touch(node_addr);
                    s4.touch(node_addr);
                    s5.touch(node_addr);
                    s6.touch(node_addr);
                    s7.touch(node_addr);
                    s8.touch(node_addr);
                    s9.touch(node_addr);

                    s0.connect();
                    s1.connect();
                    s2.connect();
                    s3.connect();
                    s4.connect();
                    s5.connect();
                    s6.connect();
                    s7.connect();
                    s8.connect();
                    s9.connect();
                );
            }

            AND_THEN ("The oat::Node:NUM_SLOTS+1 connection shall throw") {
                REQUIRE_THROWS(
                    s0.touch(node_addr);
                    s1.touch(node_addr);
                    s2.touch(node_addr);
                    s3.touch(node_addr);
                    s4.touch(node_addr);
                    s5.touch(node_addr);
                    s6.touch(node_addr);
                    s7.touch(node_addr);
                    s8.touch(node_addr);
                    s9.touch(node_addr);
                    s10.touch(node_addr);

                    s0.connect();
                    s1.connect();
                    s2.connect();
                    s3.connect();
                    s4.connect();
                    s5.connect();
                    s6.connect();
                    s7.connect();
                    s8.connect();
                    s9.connect();
                    s10.connect();
                );
            }
        }
    }
}

SCENARIO ("Sources must connect() before waiting or posting.", "[Source]") {

    GIVEN ("A single, unconnected source ") {

        oat::Source<int> source;

        WHEN ("The source calls wait()") {
            THEN ("The source shall throw.") {
                REQUIRE_THROWS( source.wait(); );
            }
        }

        WHEN ("The source calls post()") {
            THEN ("The source shall throw.") {
            REQUIRE_THROWS( source.post(); );
            }
        }
    }
}

SCENARIO ("Sources cannot connect() to the same node more than once.", "[Source]") {

        oat::Sink<int> sink;
        oat::Source<int> source;

        INFO ("The sink binds a node");
        sink.bind(node_addr);

        WHEN ("The source connects 2x to the same node") {
            THEN ("The source shall throw on the second connection.") {
                REQUIRE_NOTHROW( source.touch(node_addr) );
                REQUIRE_NOTHROW( source.connect() );

                REQUIRE_THROWS( source.touch(node_addr) );
                REQUIRE_THROWS( source.connect() );
            }
        }
}

SCENARIO ("A Source<T> can only connect to a node bound by a Sink<T>.", "[Source]") {

    GIVEN ("A bound Sink<int> and an Source<float> with common node address") {

        oat::Sink<int> sink;
        oat::Source<float> source;

        INFO ("The sink binds a node");
        sink.bind(node_addr);

        WHEN ("The source attempts to connect()") {
            THEN ("The source shall throw.") {
                REQUIRE_THROWS( source.touch(node_addr); );
            }
        }
    }
}

SCENARIO ("Connected sources can retrieve shared objects to mutate them.", "[Source]") {

    GIVEN ("A bound Sink<int> and a connected Source<int> with common node address") {

        oat::Sink<int> sink;
        oat::Source<int> source;
        int * src_ptr = static_cast<int *>(0);
        int * snk_ptr = static_cast<int *>(0);

        INFO ("The sink binds a node");
        sink.bind(node_addr);

        WHEN ("The source calls retrieve() before connecting") {
            THEN ("The source shall throw") {
                REQUIRE_THROWS( src_ptr = source.retrieve(); );
            }
        }

        WHEN ("The source calls retrieve() after connecting") {

            INFO ("The source connects to the node");
            source.touch(node_addr);
            source.connect();

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


// TODO: specialization tests
