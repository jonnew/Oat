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

#include "CameraControl.h"

#include <signal.h>

volatile sig_atomic_t done = 0;

void term(int) {
    done = 1;
}

int main(int argc, char *argv[]) {

    signal(SIGINT, term);

    if (argc != 4) {
        std::cout << "Usage: " << argv[0] << "SINK-NAME CONFIG-FILE KEY" << std::endl;
        std::cout << "Data server for point-grey GigE cameras." << std::endl;
        return 1;
    }

    const std::string sink = static_cast<std::string>(argv[1]);
    
    CameraControl cc(sink);
    cc.configure(argv[2], argv[3]);
    
    std::cout << "GigECamera server named \"" + sink + "\" has started." << std::endl;

    // TODO: exit signal
     while (!done) {
        cc.serveMat();
    }

    // Exit
    std::cout << "GigECamera server named\"" + sink + "\" is exiting." << std::endl;
    return 0;
}
