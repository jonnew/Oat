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

#include "Viewer.h"

#include <string>
#include <signal.h>

volatile sig_atomic_t done = 0;

void term(int) {
    done = 1;
}

int main(int argc, char *argv[]) {

    signal(SIGINT, term);

    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " SOURCE-NAME" << std::endl;
        std::cout << "Viewer for cv::Mat data servers" << std::endl;
        return 1;
    }

    const std::string source = static_cast<std::string>(argv[1]);
    
    Viewer viewer(source);
    std::cout << "A viewer has begun listening to source \"" + source + "\"." << std::endl;

    // Execute infinite, thread-safe loop with showImage calls governed by
    // underlying condition variable system.
    while (!done) {
        viewer.showImage();
    }

    // Exit
    std::cout << "Viewer listening to source \"" + source + "\" is exiting." << std::endl;
    return 0;
}
