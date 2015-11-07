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
#include "catch.hpp"
//#include "/home/jon/Public/Oat/debug/catch/src/catch/include/catch.hpp"

#include <string>

#include "../src/Sink.h"
#include "../src/SharedCVMat.h"

SCENARIO ("Sinks can bind a single Node.", "[Sink]") {

    GIVEN ("Two sinks") {

        std::string addr {"test"};
        oat::Sink<int> sink1;
        oat::Sink<int> sink2;
        
        //TODO: Define stream operators for Nodes, Sources, and Sinks
        //CAPTURE(sink1);
        //CAPTURE(sink2);

        WHEN ("sink1 binds a shmem segment") {

            sink1.bind(addr);

            THEN ("An attempt to bind that segment by sink2 shall throw") {
                REQUIRE_THROWS(
                    sink2.bind(addr);
                );
            }
        }
    }
}

SCENARIO ("Sinks must bind() before waiting or posting.", "[Sink]") {
    
    GIVEN ("A single sink") {

        oat::Sink<int> sink;
        std::string addr {"test"};

        WHEN ("When the sink calls wait() before binding a segment") {

            THEN ("The the sink shall throw") {
                REQUIRE_THROWS(
                    sink.wait();
                );
            }
        }
        
        WHEN ("When the sink calls post() before binding a segment") {

            THEN ("The the sink shall throw") {
                REQUIRE_THROWS(
                    sink.post();
                );
            }
        }
        
        WHEN ("When the sink calls wait() after binding a segment") {
            
            sink.bind(addr);

            THEN ("The the sink shall not throw") {
                REQUIRE_NOTHROW(
                    sink.wait();
                );
            }
        }
        
        WHEN ("When the sink calls post() after binding a segment") {

            sink.bind(addr);
            
            THEN ("The the sink shall not throw") {
                REQUIRE_NOTHROW(
                    sink.post();
                );
            }
        }
    }
}

SCENARIO ("Bound sinks can retrieve and clone shared objects.", "[Sink]") {
    
    GIVEN ("A single sink and an integer to be shared") {

        int shared_value;
        oat::Sink<int> sink;
        std::string addr {"test"};
        
        WHEN ("When the sink calls retrieve() before binding a segment") {

            THEN ("The the sink shall throw") {
                REQUIRE_THROWS( shared_value = sink.retrieve(); );
            }
        }

        WHEN ("When the sink calls retrieve() after binding a segment") {

            sink.bind(addr); 
            
            THEN ("The the sink returns a mutable reference to the shared integer") {
                
                INFO ("Start with shared_value uninitialized")
                CAPTURE(shared_value);

                REQUIRE_NOTHROW( shared_value = sink.retrieve(); );

                INFO ("Set shared_value to 1") 
                shared_value = 1;
                CAPTURE(shared_value);
                
                REQUIRE( shared_value == sink.retrieve() );
            }
        }
    }
}


SCENARIO ("Sink<SharedCVMat> must bind() before waiting, posting, or allocating.", "[Sink, SharedCVMat]") {
    
    GIVEN ("A single Sink<SharedCVMat>") {

        oat::Sink<oat::SharedCVMat> sink;
        std::string addr {"test"};
        cv::Mat mat; 
        cv::Size dims {100, 100};
        int type {1};

        WHEN ("When the sink calls wait() before binding a segment") {

            THEN ("The the sink shall throw") {
                REQUIRE_THROWS(
                    sink.wait();
                );
            }
        }
        
        WHEN ("When the sink calls post() before binding a segment") {

            THEN ("The the sink shall throw") {
                REQUIRE_THROWS(
                    sink.post();
                );
            }
        }
        
        WHEN ("When the sink calls retrieve() before binding a segment") {

            THEN ("The the sink shall throw") {
                REQUIRE_THROWS(
                    mat = sink.retrieve(dims, type);
                );
            }
        }
    }
}
