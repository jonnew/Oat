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

void term(int) {
    done = 1;
}

int main(int argc, char *argv[]) {

    signal(SIGINT, term);

    if (argc != 5) {
        std::cout << "Usage: " << argv[0] << " SOURCE-NAME SINK-NAME CONFIG-FILE CONFIG-KEY" << std::endl; 
        std::cout << "HSV object detector for cv::Mat data servers." << std::endl;
        return 1;
    }

    HSVDetector detector(argv[1], argv[2]);
    detector.configure(argv[3], argv[4]);
    std::cout << "HSV detector \"" + detector.get_detector_name() + "\" has begun listening to source \"" + detector.get_cli_name() + "\"." << std::endl;
    std::cout << "HSV detector \"" + detector.get_detector_name() + "\" has begun steaming to sink \"" + detector.get_srv_name() + "\"." << std::endl;

    // Execute infinite, thread-safe loop with function calls governed by
    // underlying condition variable system.
    while (!done) {
        detector.applyFilter();
        cv::waitKey(1);
    }

    // Exit
    std::cout << "HSV detector \"" + detector.get_detector_name() + "\" is exiting." << std::endl;
    return 0;
}


