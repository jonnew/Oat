//******************************************************************************
//* File:   Node_test.cpp
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

#include "../../lib/shmemdf/Node.h"

SCENARIO ("Nodes can accept up to Node::NUM_SLOTS sources.", "[Node]") {

    GIVEN ("A fresh Node") {

        oat::Node node;
        REQUIRE (node.source_ref_count() == 0);
        REQUIRE (node.sink_state() == oat::NodeState::UNDEFINED);

        WHEN ("Node::NUM_SLOTS+1 sources are added") {

            THEN ("The Node shall return normal exit codes until the 11th") {
                for (size_t i = 0; i <= oat::Node::NUM_SLOTS; i++) {
                    if (i < oat::Node::NUM_SLOTS)
                        REQUIRE (node.acquireSlot(i) == 0);
                    else
                        REQUIRE (node.acquireSlot(i) < 0);
                }
            }
        }

        WHEN ("a source is removed") {

            node.releaseSlot(0);

            THEN ("the source ref count remains 0") {
                REQUIRE(node.source_ref_count() == 0);
            }
        }

        WHEN ("a negatively indexed read-barrier is read") {

            THEN ("The Node shall throw") {
                REQUIRE_THROWS(
                    boost::interprocess::interprocess_semaphore &s = node.read_barrier(-1);
                );
            }
        }

        WHEN ("a single source is added") {

            size_t idx;
            node.acquireSlot(idx);

            THEN ("reading a greater indexed read-barrier shall throw") {
                REQUIRE_THROWS(
                boost::interprocess::interprocess_semaphore &s = node.read_barrier(idx+1);
                );
            }
        }
    }
}
