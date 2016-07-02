//******************************************************************************
//* File:   RecordControl.h
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

#ifndef OAT_RECORDCONTROL_H
#define OAT_RECORDCONTROL_H

#include <iosfwd>
#include <string>

#include "Recorder.h"

namespace oat {

const char interactive_record_control_usage_string[] =
    "COMMANDS\n"
    "CMD         FUNCTION\n"
    " help       Print this information.\n"
    " start      Start recording. This will append the file if it\n"
    "            already exists. It will create a new one if it doesn't.\n"
    " pause      Pause recording. This will pause the recording\n"
    "            without creating a new file.\n"
    " new        Start a new file using folder location and file name\n"
    "            options as provided in command line arguements.\n"
    " quit       Exit the program.\n";

const char remote_record_control_usage_string[] =
    "Recorder is under remote control.\n"
    "Commands provided through STDIN have no effect.\n"
    "except Ctrl+C to quit.\n";

int controlRecorder(std::istream &in,
                    std::ostream &out,
                    oat::Recorder &recorder,
                    bool pretty_cmd = false);

void printInteractiveUsage(std::ostream &out);
void printRemoteUsage(std::ostream &out);

}      /* namespace oat */
#endif /* OAT_RECORDCONTROL_H */

