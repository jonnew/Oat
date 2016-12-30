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

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include <string>

#include "../../lib/datatypes/Color.h"
#include "../../lib/shmemdf/SharedFrameHeader.h"
#include "../../lib/shmemdf/Sink.h"

const std::string node_addr = "test";

// Global via extern in Globals.h
namespace oat { volatile sig_atomic_t quit = 0; }

SCENARIO ("Sinks can bind a single Node.", "[Sink]") {

    GIVEN ("Two Sink<int>'s") {

        oat::Sink<int> sink1;
        oat::Sink<int> sink2;

        //TODO: Define stream operators for Nodes, Sources, and Sinks
        //CAPTURE(sink1);
        //CAPTURE(sink2);

        WHEN ("sink1 binds a shmem segment") {

            sink1.bind(node_addr);

            THEN ("An attempt to bind that segment by sink2 shall throw") {
                REQUIRE_THROWS( sink2.bind(node_addr); );
            }
        }
    }
}

SCENARIO ("Sinks must bind() before waiting or posting.", "[Sink]") {

    GIVEN ("A single Sink<int>") {

        oat::Sink<int> sink;

        WHEN ("When the sink calls wait() before binding a segment") {

            THEN ("The the sink shall throw") {
                REQUIRE_THROWS( sink.wait(); );
            }
        }

        WHEN ("When the sink calls post() before binding a segment") {

            THEN ("The the sink shall throw") {
                REQUIRE_THROWS( sink.post(); );
            }
        }

        WHEN ("When the sink calls wait() after binding a segment") {

            sink.bind(node_addr);

            THEN ("The the sink shall not throw") {
                REQUIRE_NOTHROW( sink.wait(); );
            }
        }

        WHEN ("When the sink calls post() after binding a segment "
               "before calling wait()") {

            sink.bind(node_addr);

            THEN ("The the sink shall throw") {
                REQUIRE_THROWS( sink.post(); );
            }

        }

        WHEN ("When the sink calls post() after binding a segment "
               "after calling wait()") {

            sink.bind(node_addr);
            sink.wait();

            THEN ("The the sink shall not throw") {
                REQUIRE_NOTHROW( sink.post(); );
            }
        }
    }
}

SCENARIO ("Sinks cannot bind() to the same node more than once.", "[Source]") {

        oat::Sink<int> sink;

        INFO ("The sink binds a node");
        REQUIRE_NOTHROW( sink.bind(node_addr); );
        REQUIRE_THROWS( sink.bind(node_addr); );
}

SCENARIO ("Bound sinks can retrieve shared objects to mutate them.", "[Sink]") {

    GIVEN ("A single Sink<int> and a shared *int=0") {

        int * shared = static_cast<int *>(0);
        oat::Sink<int> sink;

        WHEN ("When the sink calls retrieve() before binding a segment") {

            THEN ("The the sink shall throw") {
                REQUIRE_THROWS( shared = sink.retrieve(); );
            }
        }

        WHEN ("When the sink calls retrieve() after binding a segment") {

            sink.bind(node_addr);

            THEN ("The the sink returns a pointer to mutate the shared integer") {

                INFO ("Start with shared int uninitialized")
                CAPTURE(shared);

                REQUIRE_NOTHROW( shared = sink.retrieve(); );

                INFO ("Set shared int to 1")
                *shared = 1;
                CAPTURE(*shared);

                REQUIRE( *shared == *sink.retrieve() );
            }
        }
    }
}

SCENARIO ("Sink<SharedFrameHeader> must bind() before waiting, posting, or allocating.", "[Sink, SharedFrameHeader]") {

    GIVEN ("A single Sink<SharedFrameHeader>") {

        oat::Sink<oat::Frame> sink;
        cv::Mat mat;
        size_t cols {100};
        size_t rows {100};
        int type {1};
        oat::PixelColor color {oat::PIX_BGR};

        WHEN ("When the sink calls wait() before binding a segment") {

            THEN ("The the sink shall throw") {
                REQUIRE_THROWS( sink.wait(); );
            }
        }

        WHEN ("When the sink calls post() before binding a segment") {

            THEN ("The the sink shall throw") {
                REQUIRE_THROWS( sink.post(); );
            }
        }

        WHEN ("When the sink calls retrieve() before binding a segment") {

            THEN ("The the sink shall throw") {
                REQUIRE_THROWS( mat = sink.retrieve(cols, rows, type, color); );
            }
        }
    }
}
