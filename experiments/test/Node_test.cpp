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

#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() 
#include "catch.hpp"

#include "../src/Node.h"

SCENARIO ("Nodes can accept up to 10 sources.", "[Node]") {

    GIVEN ("A fresh Node") {

        oat::Node node;
        REQUIRE(node.source_ref_count() == 0);
        REQUIRE(node.sink_state() == oat::SinkState::UNDEFINED);

        WHEN( "10 sources are added") {

            THEN ("The Node shall not throw until the 11th") {
                REQUIRE_NOTHROW(
                for (int i = 0; i < 10; i++) {
                    node.incrementSourceRefCount();
                }
                );
            }
        }

        WHEN ("11 sources are added") {

            THEN ("The Node shall throw") {
                REQUIRE_THROWS(
                for (int i = 0; i < 11; i++) {
                    node.incrementSourceRefCount();
                }
                );
            }
        }

        WHEN ("a source is removed") {

            node.decrementSourceRefCount();

            THEN ("the source ref count remains 0") {
                REQUIRE(node.source_ref_count() == 0);
            }
        }
        
        WHEN ("a negatively indexed read-barrier is read") {
            
            THEN ("The Node shall throw") {
                REQUIRE_THROWS(
                boost::interprocess::interprocess_semaphore &s = node.readBarrier(-1);
                );
            }
        }
        
        WHEN ("a single source is added") {
            
            size_t idx = node.incrementSourceRefCount();
            
            THEN ("reading a greater indexed read-barrier shall throw") {
                REQUIRE_THROWS(
                boost::interprocess::interprocess_semaphore &s = node.readBarrier(idx+1);
                );
            }
        }
    }
}