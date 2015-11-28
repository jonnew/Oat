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

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include <future>
#include <thread>
#include <chrono>

#include "../../lib/shmemdf/Sink.h"
#include "../../lib/shmemdf/Source.h"

// Test outline
//
//### A Sink and Source bound to a common Node must respect eachothers' locks
//- Given a sink and source
//    - When source attempts to enter the critical section first
//        - Then, the the source shall block until the sink post()'s
//    - When the sink enters a critical section first
//        - Then, the source shall block until the sink post()'s
//    - When the source enters a critical section
//        - Then, the sink shall block until the source post()s
//        - Then, a second source should not be able to enter the critical section until the sink posts
//
//### The order of connecting and binding is irrelevant. The sink always enters the critical section first
//- Given a sink and two sources
//    - When, 1. The sources connect()
//            2. The sources attempt to enter the critical section
//            3. The sink bind()'s
//        - Then, the sources shall block until the sink enters/exits the critical section
//    - When, 1. The sink binds
//            2. the sources connect()
//            3. the sources try to enter critical section
//        - Then, the sources shall block until the sink enters/exits the critical section
//    - When, 1. The first source connect()'s
//            2. The first sources tries to enter the critical section
//            3. The sink binds,
//            4. The second source connect()'s
//            5. The second source tries to enter the critical section
//        - Then, the sources shall block until the sink enters/exits the critical section
//
//### Sources attaching or detaching during source or sink wait() periods should not cause deadlocks
//- Given a sink and two sources
//    - When, 1. The sink binds
//            2. the sources connect
//            3. The sink enters the critical section
//            4. One of the sources is destructed
//        - Then, the first source shall block until the sink enters/exits the critical section even if the second destructs
//- Given a sink and a source
//    - When, 1. The sink binds
//            2. The sink enters the critical section
//            3. A source connects
//            4. The source attempts to enter the critical section
//        - Then, the source shall block until the sink enters/exits the critical section

using msec = std::chrono::milliseconds;
const std::string node_addr = "test";

SCENARIO ("A Sink and Source bound to a common Nodes must respect "
          "eachothers' locks.", "[Sink, Source, Concurrency]") {

    GIVEN ("A sink and source") {

        oat::Sink<int> sink;
        oat::Source<int> source;

        WHEN ("The source attempts to enter the critical section first.") {

            sink.bind(node_addr);

            // Source connects and then attempts to enter critical section on a
            // separate thread
            source.touch(node_addr); 
            source.connect(); 
            auto fut = std::async(std::launch::async, [&source]{ source.wait(); });

            THEN ("The source shall block until the sink post()'s") {

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

        WHEN ("The sink enters a critical section first") {

            // Sink binds to "test" and enters critical section
            sink.bind(node_addr);
            sink.wait();

            // Source connects and then wait()s on a separate thread
            source.touch(node_addr); 
            source.connect(); 
            auto fut = std::async(std::launch::async, [&source]{ source.wait(); });

            THEN ("The source shall block until the sink post()'s") {

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

        WHEN ("The source enters a critical section") {

            // Sink binds and source connects to "test"
            sink.bind(node_addr);
            source.touch(node_addr);
            source.connect();

            // Sink tries to post without waiting first (bad lock order)
            REQUIRE_THROWS(sink.post());

            // The sink enters and exits critical section properly
            REQUIRE_NOTHROW(sink.wait());
             /* Critical */
            REQUIRE_NOTHROW(sink.post());

            // Source tries to post before waiting (Bad unlock order)
            REQUIRE_THROWS(source.post());

            // Source enters the critical section, correctly
            REQUIRE_NOTHROW(source.wait());

            THEN ("The the sink shall block until the sink post()'s") {

                // The sink attempts to enter the crtical section on
                // a separate thread
                auto fut = std::async(std::launch::async, [&sink]{ sink.wait(); });

                // Pause for 5 ms
                std::this_thread::sleep_for(msec(5));

                // Check to see that the sink has not stopped waiting
                auto status = fut.wait_for(msec(0));
                REQUIRE(status != std::future_status::ready);

                // Source tries to wait after just completeling wait (Bad lock order)
                REQUIRE_THROWS( source.wait());

                // Correct unlock order
                REQUIRE_NOTHROW(source.post());

                // Give sufficient time for wait to release
                std::this_thread::sleep_for(msec(1));
                status = fut.wait_for(msec(0));
                REQUIRE(status == std::future_status::ready);
            }

            THEN ("A second source should not be able to enter "
                  "the critical section until the sink posts") {

                // A second sink is generated, and attempts to enter
                // the critical section on different thread
                oat::Source<int> source1;
                source1.touch(node_addr); 
                source1.connect(); 
                auto fut = std::async(std::launch::async, [&source1]{ source1.wait(); });

                // Check to see that the second source has not entered critical section
                auto status = fut.wait_for(msec(0));
                REQUIRE(status != std::future_status::ready);

                // When the first source post's, such that the sink can proceed
                REQUIRE_NOTHROW(source.post());

                // The sink tries to post without waiting first (bad unlock order)
                REQUIRE_THROWS(sink.post());

                // The sink enters and exits critical section properly
                /* Start Critical */
                REQUIRE_NOTHROW(sink.wait());

                // Check to see that the source1 has not entered critical section
                status = fut.wait_for(msec(0));
                REQUIRE(status != std::future_status::ready);

                REQUIRE_NOTHROW(sink.post());
                /* End Critical */

                // Give sufficient time for wait to release
                std::this_thread::sleep_for(msec(1));

                // Check to see that the second source has entered critical section
                status = fut.wait_for(msec(0));
                REQUIRE(status == std::future_status::ready);

                // And confirm that the first source can also proceed
                REQUIRE_NOTHROW(source.wait());
            }
        }
    }
}

SCENARIO ("The order of connecting and binding is irrelevant. "
          "The sink always enters the critical section first",
          "[Sink, Source, Concurrency]") {

    GIVEN ("Two sources and a sink.") {

        oat::Sink<int> sink;
        oat::Source<int> s0, s1;

        WHEN ("1. The sources connect(), "
              "2. The sources attempt to enter the critical section, "
              "3. The sink bind()'s") {

            // Source0 tries to connect and enter critical section
            s0.touch(node_addr); 
            auto s0_fut = std::async(std::launch::async, [&s0]{ s0.connect(); s0.wait(); });

            // Source1 tries to connect and enter critical section
            s1.touch(node_addr); 
            auto s1_fut = std::async(std::launch::async, [&s1]{ s1.connect(); s1.wait(); });

            THEN ("The sources shall block until the sink enters/exits the critical section.") {

                // Pause for 5 ms
                std::this_thread::sleep_for(msec(5));

                // Check to see that the source has not stopped waiting
                auto status_0 = s0_fut.wait_for(msec(0));
                REQUIRE(status_0 != std::future_status::ready);

                auto status_1 = s1_fut.wait_for(msec(0));
                REQUIRE(status_1 != std::future_status::ready);

                // Sink binds
                sink.bind(node_addr);

                // Tries to post without waiting first (bad lock order)
                REQUIRE_THROWS(sink.post());

                /* Start Critical */
                REQUIRE_NOTHROW(sink.wait());

                // Check to see that the sources have not stopped waiting
                status_0 = s0_fut.wait_for(msec(0));
                REQUIRE(status_0 != std::future_status::ready);

                status_1 = s1_fut.wait_for(msec(0));
                REQUIRE(status_1 != std::future_status::ready);

                REQUIRE_NOTHROW(sink.post());
                /* End Critical */

                // Give sufficient time for wait to release
                std::this_thread::sleep_for(msec(1));

                // Check to see that the sources have connected and entered
                // critical section
                status_0 = s0_fut.wait_for(msec(0));
                REQUIRE(status_0 == std::future_status::ready);

                status_1 = s1_fut.wait_for(msec(0));
                REQUIRE(status_1 == std::future_status::ready);

            }
        }

        WHEN ("1. The sink binds, "
              "2. the sources connect(), "
              "3. the sources try to enter critical section") {

            // Sink binds
            sink.bind(node_addr);

            // Source0 tries to connect and enter critical section
            s0.touch(node_addr);
            auto s0_fut = std::async(std::launch::async, [&s0]{ s0.connect(); s0.wait(); });

            // Source1 tries to connect and enter critical section
            s1.touch(node_addr);
            auto s1_fut = std::async(std::launch::async, [&s1]{ s1.connect(); s1.wait(); });

            THEN ("The sources shall block until the sink enters/exits the critical section.") {

                // Pause for 5 ms
                std::this_thread::sleep_for(msec(5));

                // Check to see that the source has not stopped waiting
                auto status_0 = s0_fut.wait_for(msec(0));
                REQUIRE(status_0 != std::future_status::ready);

                auto status_1 = s1_fut.wait_for(msec(0));
                REQUIRE(status_1 != std::future_status::ready);

                // Tries to post without waiting first (bad lock order)
                REQUIRE_THROWS(sink.post());

                /* Start Critical */
                // Enters and exits properly
                REQUIRE_NOTHROW(sink.wait());

                // Check to see that the sources have not stopped waiting
                status_0 = s0_fut.wait_for(msec(0));
                REQUIRE(status_0 != std::future_status::ready);

                status_1 = s1_fut.wait_for(msec(0));
                REQUIRE(status_1 != std::future_status::ready);

                REQUIRE_NOTHROW(sink.post());
                /* End Critical */

                // Give sufficient time for wait to release
                std::this_thread::sleep_for(msec(1));

                // Check to see that the source has entered the critical
                // section
                status_0 = s0_fut.wait_for(msec(0));
                REQUIRE(status_0 == std::future_status::ready);

                status_1 = s1_fut.wait_for(msec(0));
                REQUIRE(status_1 == std::future_status::ready);
            }
        }

        WHEN ("1. The first source connect()'s, "
              "2. The first sources tries to enter the critical section, "
              "3. The sink binds, "
              "4. The second source connect()'s, "
              "5. The second source tries to enter the critical section.") {

            // Source0 tries to connect and wait
            s0.touch(node_addr);
            auto s0_fut = std::async(std::launch::async, [&s0]{ s0.connect(); s0.wait(); });

            // Sink binds
            sink.bind(node_addr);

            // Source1 tries to connect and wait
            s1.touch(node_addr);
            auto s1_fut = std::async(std::launch::async, [&s1]{ s1.connect(); s1.wait(); });

            THEN ("The sources shall block until the sink enters/exits the critical section.") {

                // Pause for 5 ms
                std::this_thread::sleep_for(msec(5));

                // Check to see that the sources have not stopped waiting
                auto status_0 = s0_fut.wait_for(msec(0));
                REQUIRE(status_0 != std::future_status::ready);

                auto status_1 = s1_fut.wait_for(msec(0));
                REQUIRE(status_1 != std::future_status::ready);

                // Tries to post without waiting first (bad lock order)
                REQUIRE_THROWS(sink.post());

                /* Start Critical */
                // Enters and exits properly
                REQUIRE_NOTHROW(sink.wait());

                // Check to see that the sources have not stopped waiting
                status_0 = s0_fut.wait_for(msec(0));
                REQUIRE(status_0 != std::future_status::ready);

                status_1 = s1_fut.wait_for(msec(0));
                REQUIRE(status_1 != std::future_status::ready);

                REQUIRE_NOTHROW(sink.post());
                /* End Critical */

                // Give sufficient time for wait to release
                std::this_thread::sleep_for(msec(1));

                // Check to see that the sources have entered the critical
                // section
                status_0 = s0_fut.wait_for(msec(0));
                REQUIRE(status_0 == std::future_status::ready);

                status_1 = s1_fut.wait_for(msec(0));
                REQUIRE(status_1 == std::future_status::ready);
            }
        }
    }
}

SCENARIO ("Sources attaching or detaching during source or sink wait() periods "
          "should not cause deadlocks", "[Sink, Source, Concurrency]") {

    GIVEN ("Two sources and a sink.") {

        oat::Sink<int> sink;
        oat::Source<int> s0;
        auto s1 = new oat::Source<int>();

        WHEN ("1. The sink bind()'s, "
              "2. The sources connect()'s, "
              "3. The sink enters the critical section") {

            sink.bind(node_addr);
            s0.touch(node_addr);
            s0.connect();
            s1->touch(node_addr);
            s1->connect();

            sink.wait();

            THEN ("The first source shall block until the sink enters/exits the "
                  "critical section even if the second destructs") {

                // Source0 tries enter critical section
                auto s0_fut = std::async(std::launch::async, [&s0]{ s0.wait(); });

                // Source1 destructs
                delete s1;

                // Pause for 5 ms
                std::this_thread::sleep_for(msec(5));

                // Check to see that the source0 has not stopped waiting
                auto status_0 = s0_fut.wait_for(msec(0));
                REQUIRE(status_0 != std::future_status::ready);

                REQUIRE_NOTHROW(sink.post());
                /* End Critical */

                // Give sufficient time for wait to release
                std::this_thread::sleep_for(msec(1));

                // Check to see that source0 have connected and entered
                // critical section
                status_0 = s0_fut.wait_for(msec(0));
                REQUIRE(status_0 == std::future_status::ready);

            }
        }
    }

    GIVEN ("Two sources and a sink.") {

        oat::Sink<int> sink;
        oat::Source<int> s0, s1;

        WHEN ("1. The sink binds, "
              "2. The sink enters the critical section, "
              "3. A sources connect, "
              "4. The sources attempt to enter the critical section.") {

            REQUIRE_NOTHROW( sink.bind(node_addr));
            /* Start Critical */
            REQUIRE_NOTHROW( sink.wait());

            // Source0 tries to connect and enter critical section
            s0.touch(node_addr); 
            auto s0_fut = std::async(std::launch::async, [&s0]{ s0.connect(); s0.wait(); });

            // Source1 tries to connect and enter critical section
            s1.touch(node_addr); 
            auto s1_fut = std::async(std::launch::async, [&s1]{ s1.connect(); s1.wait(); });

            THEN ("The source shall block until the sink enters/exits the "
                  "critical section") {

                // Pause for 5 ms
                std::this_thread::sleep_for(msec(5));

                // Check to see that the source0 has not stopped waiting
                auto status_0 = s0_fut.wait_for(msec(0));
                REQUIRE(status_0 != std::future_status::ready);

                auto status_1 = s1_fut.wait_for(msec(0));
                REQUIRE(status_1 != std::future_status::ready);

                REQUIRE_NOTHROW(sink.post());
                /* End Critical */

                // Give sufficient time for wait to release
                std::this_thread::sleep_for(msec(1));

                // Check to see that the sources have entered the
                // critical section
                status_0 = s0_fut.wait_for(msec(0));
                REQUIRE(status_0 == std::future_status::ready);

                status_1 = s1_fut.wait_for(msec(0));
                REQUIRE(status_1 == std::future_status::ready);

            }
        }
    }
}
