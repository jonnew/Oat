//******************************************************************************
//* Copyright (c) Jon Newman (jpnewman at mit snail edu) 
//* All right reserved.
//* This file is part of the Simple Tracker project.
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

#include "HSVDetector.h"

#include <signal.h>
#include <opencv2/highgui.hpp>

volatile sig_atomic_t done = 0;
bool running = true;

void term(int) {
    done = 1;
}

void run(std::string source, std::string pos_sink, std::string config_file, std::string config_key, std::string frame_sink=NULL) {

    HSVDetector detector(source, pos_sink);
    if(frame_sink){
        // TODO
    }
    detector.configure(config_file, config_key);
    
    
    std::cout << "HSV detector \"" + detector.get_detector_name() + "\" has begun listening to source \"" + source + "\"." << std::endl;
    std::cout << "HSV detector \"" + detector.get_detector_name() + "\" has begun steaming to sink \"" + sink + "\"." << std::endl;

    while (!done) { // !done
        if (running) {
            detector.applyFilterAndServe();
        }
    }
    
    std::cout << "HSV detector \"" + detector.get_detector_name() + "\" is exiting." << std::endl;
}

int main(int argc, char *argv[]) {

    signal(SIGINT, term);

    if (argc < 5) {
        std::cout << "Usage: " << argv[0] << " SOURCE-NAME POSITION-SINK-NAME CONFIG-FILE CONFIG-KEY [FRAME-SINK-NAME]" << std::endl;
        std::cout << "HSV object detector for cv::Mat data servers." << std::endl;
        return 1;
    }
    
//    // TODO: more robust parsing scheme...
//    const std::string source = static_cast<std::string> (argv[1]);
//    const std::string pos_sink = static_cast<std::string> (argv[2]);
//    const std::string config_file = static_cast<std::string> (argv[3]);
//    const std::string config_key = static_cast<std::string> (argv[4]);
//    if( argc == 5) {
//        
//    else
//    if (argc > 5) {
//        const std::string frame_sink = static_cast<std::string> (argv[5]);
//    }

    // Exit
    return 0;
}


