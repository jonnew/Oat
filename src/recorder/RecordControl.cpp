//******************************************************************************
//* File:   RecordControl.cpp
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

#include <unordered_map>

#include "RecordControl.h"

namespace oat {

int controlRecorder(std::istream &in, oat::Recorder &recorder, bool print_cmd) {

    // Command map
    std::unordered_map<std::string, char> cmd_map;
    cmd_map["exit"] = 'e';
    cmd_map["help"] = 's';
    cmd_map["start"] = 's';
    cmd_map["stop"] = 'S';
    cmd_map["new"] = 'n';
    cmd_map["move"] = 'm';

    // User control loop
    std::string cmd;

    bool quit = false;

    while (!quit) {

        std::cout << ">>> ";
        std::getline(in, cmd);

        if (print_cmd)
            std::cout << cmd << "\n";

        switch (cmd_map[cmd]) {
            case 's' :
            {
                recorder.set_record_on(true);
                std::cout << "Recording ON.\n";
                break;
            }
            case 'S' :
            {
                recorder.set_record_on(false);
                std::cout << "Recording OFF.\n";
                break;
            }
            case 'e' :
            {
                quit = true;
                break;
            }
            default :
            {
                std::cerr << "Invalid command \'" << cmd << "\'\n";
                break;
            }
        }
    }

    return 0;
}

void printInteractiveUsage(std::ostream &out) {

    out << "COMMANDS\n"
        << "CMD         FUNCTION\n"
        << " help       Print this information.\n"
        << " start      Start recording. This will append and file if it\n"
        << "            already exists.\n"
        << " pause      Pause recording. This will pause\n"
        << "            recording and will not start a new file.\n"
        << " new        Start new file. Start time will be used to create\n"
        << "            unique file name.\n"
        << " rename     Specify a new file location. User will be prompted\n"
        << "            to select a new save location.\n"
        << " exit       Exit the program.\n";
}

void printRemoteUsage(std::ostream &out) {

    out << "Recorder is under remote control.\n"
        << "Commands provided through STDIN have no effect.\n";
}

} /* namespace oat */

