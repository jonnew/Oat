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

#include <future>
#include <thread>
#include <chrono>

#include "../../lib/shmemdf/Sink.h"
#include "../../lib/shmemdf/Source.h"

// Test outline
//
//- Given a fresh source
//    - When source attempts to enter the critical section first, it 
//      should wait until the sink enters and leaves.
//    - When a sink is in the critical section, it should not be 
//      possible for the source to enter until a sink posts
//    - When the source enters the critical section, it should not be
//      possible for the sink to enter until the source posts
//    - When the source enters the critcal section, it should be posible 
//      for another source to connect(), and enter the critical section
//
//- Given several fresh sources
//    - It should not be possible for any of them to enter the critical section until a sink posts
//
//- Given a source in the critical section
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

using msec = std::chrono::milliseconds;
const std::string node_addr = "test";

int sinkBind(oat::Sink<int> sink) { sink.bind(node_addr); return 0; }
int sinkPost(oat::Sink<int> sink) { sink.post(); return 0; }
int sinkWait(oat::Sink<int> sink) { sink.post(); return 0; }
int sourceConnect(oat::Source<int> source) { source.connect(node_addr); return 0; }
int sourcePost(oat::Source<int> source) { source.post(); return 0; }
int sourceWait(oat::Source<int> source) { source.wait(); return 0; }

SCENARIO ("Sinks and Sources bound to a common Nodes must respect "
          "eachothers' locks.", "[Sink, Source, Concurrency]") {

    GIVEN ("A sink and source") {

        oat::Sink<int> sink;
        oat::Source<int> source;

        WHEN ("The source attempts to enter the critical section first.") {
           
            sink.bind(node_addr);

            // Source connects and then attempts to enter critical section on a
            // separate thread
            source.connect(node_addr);
            auto fut = std::async(std::launch::async, [&source]{ source.wait(); });

            THEN ("The the source shall block until the sink post()'s") {

                // Pause for 5 ms
                std::this_thread::sleep_for(msec(5));

                // Check to see that the source has not stopped waiting
                auto status = fut.wait_for(msec(0));
                REQUIRE(status != std::future_status::ready);

                // The enters and exits critical section
                sink.wait();
                /* Critical */
                sink.post();

                // Give sufficient time for wait to release
                std::this_thread::sleep_for(msec(1));
                status = fut.wait_for(msec(0));
                REQUIRE(status == std::future_status::ready);
            }
        }
        
        WHEN ("When the sink enters a critical section first") {

            // Sink binds to "test" and enters critical section
            sink.bind(node_addr);
            sink.wait();

            // Source connects and then wait()s on a separate thread
            source.connect(node_addr);
            auto fut = std::async(std::launch::async, [&source]{ source.wait(); });

            THEN ("The the source shall block until the sink post()'s") {

                // Pause for 5 ms
                std::this_thread::sleep_for(msec(5));

                // Check to see that the source has not stopped waiting
                auto status = fut.wait_for(msec(0));
                REQUIRE(status != std::future_status::ready);

                // The sink posts()
                sink.post();

                // Give sufficient time for wait to release
                std::this_thread::sleep_for(msec(1));
                status = fut.wait_for(msec(0));
                REQUIRE(status == std::future_status::ready);
            }
        }

        WHEN ("When the source enters a critical section") {

            // Sink binds and source connects to "test" 
            sink.bind(node_addr);
            source.connect(node_addr);

            // Sink enters critical section and leaves
            sink.wait();
            /* Critical */
            sink.post();

            // Source enters the critical section
            source.wait();

            // The sink attempts to enter the crtical section on
            // a separate thread
            auto fut = std::async(std::launch::async, [&sink]{ sink.wait(); });

            THEN ("The the sink shall block until the sink post()'s") {

                // Pause for 5 ms
                std::this_thread::sleep_for(msec(5));

                // Check to see that the source has not stopped waiting
                auto status = fut.wait_for(msec(0));
                REQUIRE(status != std::future_status::ready);

                // The sink posts()
                source.post();

                // Give sufficient time for wait to release
                std::this_thread::sleep_for(msec(1));
                status = fut.wait_for(msec(0));
                REQUIRE(status == std::future_status::ready);
            }
        }
//    - When the source enters the critcal section, it should be posible 
//      for another source to connect(), and enter the critical section
        WHEN ("When the source enters a critical section") {

            // Sink binds and source connects to "test" 
            sink.bind(node_addr);
            source.connect(node_addr);

            // Sink enters critical section and leaves
            sink.wait();
            /* Critical */
            sink.post();

            // Source enters the critical section
            source.wait();

            THEN ("A second source should not be ble to enter "
                  "the critical section until the sink posts") {

                // A second sink is generated, and attempts to enter
                // the critical section on different thread
                oat::Source<int> source1;
                source1.connect(node_addr);
                auto fut = std::async(std::launch::async, [&source1]{ source1.wait(); });

                // Check to see that the source has not entered critical secion
                auto status = fut.wait_for(msec(0));
                REQUIRE(status != std::future_status::ready);

                // DEADLOCK : Node has two sources now, so for sink to proceed, it needs two posts!
                // TODO: FIX
                // First source posts
                source.post();

                // Sink enters critical section and leaves
                sink.wait();
                /* Critical */

                // Check to see that the source1 has not entered critical section
                status = fut.wait_for(msec(0));
                REQUIRE(status != std::future_status::ready);

                sink.post();

                // Check to see that the source1 has entered critical section
                // Give sufficient time for wait to release
                std::this_thread::sleep_for(msec(1));
                status = fut.wait_for(msec(0));
                REQUIRE(status == std::future_status::ready);
            }
        }
    }
}
