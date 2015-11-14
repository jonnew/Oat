//******************************************************************************
//* File:   concurrency_test.cpp
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
//#include <catch.hpp>
#include "/home/jon/Public/Oat/debug/catch/src/catch/include/catch.hpp"

#include <boost/thread.hpp>

#include "../../lib/shmemdf/Sink.h"
#include "../../lib/shmemdf/Source.h"

// Test outline
//
//- Given a fresh source
//    - It should not be possible for the source to enter the critical section until a sink posts
//
//- Given a set of fresh sources
//    - It should not be possible for the sources to enter the critical
//
//- Given several fresh sources
//    - It should not be possible for any of them to enter the critical section until a sink posts
//
//- Given a sink in the critical section
//    - It should not be possible for a source to enter
//
//- Given a source in the critical section
//    - It should not be possible for a sink to enter
//    - It should be possible for other sources to enter
//
//- Given a waiting sink
//    - A source should have to post for the sink to proceed
//
//- Given a waiting source
//    - A sink should have to post for the sink to proceed
//
//- Given a waiting source
//    - When a second source joins, both sources should proceed when a sink posts
//
//- Given two waiting sources
//    - When the first source leaves, the second source should proceed when a sink posts

int sinkBind(oat::Sink<int> sink) { sink.bind("test"); return 0; }
int sinkPost(oat::Sink<int> sink) { sink.post(); return 0; }
int sinkWait(oat::Sink<int> sink) { sink.post(); return 0; }
int sourceConnect(oat::Source<int> source) { source.connect("test"); return 0; }
int sourcePost(oat::Source<int> source) { source.post(); return 0; }
int sourceWait(oat::Source<int> source) { source.wait(); return 0; }

SCENARIO ("Sinks and Sources bound to a common Nodes must respect "
          "eachothers' locks.", "[Sink, Source, Concurrency]") {

    GIVEN ("A Sink and Source operating on separate threads") {

        oat::Sink<int> sink;
        oat::Source<int> source;

        WHEN ("When the sink is in the critical section") {

            // Sink binds to "test" and enters critical section
            sink.bind("test");
            sink.wait();


            
            std::thread sink_th(sourcePost(), source);
            std::thread sink_th(sinkWait(), sink);

            // Get ids of different threads
            std::thread::id sink_th_id = sink_th.get_id();
            std::thread::id sink_th_id = sink_th.get_id();

            THEN ("The the source shall wait until the sink post()'s to proceed") {
                REQUIRE_NOTHROW(
                for (size_t i = 0; i < oat::Node::NUM_SLOTS; i++) {
                    node.acquireSlot();
                }
                );
            }
        }
    }
}