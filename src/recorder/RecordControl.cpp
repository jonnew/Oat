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

#include <csignal>
#include <unordered_map>

#include "RecordControl.h"

extern volatile sig_atomic_t quit;

namespace oat {

/**
 * @brief Control a recorder.
 *
 * Allows user to control a recorder object via a simple message interface.
 *
 * @param in Input command stream.
 * @param out Output stream for command confirmations.
 * @param recorder Recorder to control.
 * @param pretty_cmd Format IO for console usage.
 *
 * @return Code:
 * 0. Normal exit.
 * 1. Create new recorder (with new file name) and re-enter.
 */
int controlRecorder(std::istream &in,
                    std::ostream &out,
                    oat::Recorder &recorder,
                    bool pretty_cmd) {

    // Command map
    std::unordered_map<std::string, char> cmd_map;
    cmd_map["quit"] = 'q';
    cmd_map["help"] = 'h';
    cmd_map["start"] = 's';
    cmd_map["pause"] = 'p';
    cmd_map["new"] = 'n';
    cmd_map["ping"] = 'i';

    // User control loop
    std::string cmd;

    // Means to exit via user input
    bool interactive_quit = false;

    while (!interactive_quit && !::quit && !recorder.source_eof()) {

        if (pretty_cmd)
            out << ">>> " << std::flush;

        std::getline(in, cmd);
        if (in.eof()) {
            out << "Input command stream closed." << std::endl;
            break;
        }

        if (cmd.empty()) {
            out << "No command...\n";
            out << "source_eof: " << recorder.source_eof() << std::endl;
            continue;
        }

        switch (cmd_map[cmd]) {
            case 'q' :
            {
                interactive_quit = true;
                out << "Received quit signal." << std::endl;
                break;
            }
            case 'h' :
            {
                printInteractiveUsage(out);
                out << std::endl;
                break;
            }
            case 's' :
            {
                if (!recorder.recording_initialized())
                    recorder.initializeRecording();
                recorder.set_record_on(true);
                out << "Recording started." << std::endl;
                break;
            }
            case 'p' :
            {
                recorder.set_record_on(false);
                out << "Recording paused." << std::endl;
                break;
            }
            case 'n' :
            {
                recorder.set_record_on(false);
                out << "Creating new file." << std::endl;
                return 1;
            }
            case 'i' :
            {
                out << "Ping received." << std::endl;
                break;
            }
            default :
            {
                out << "Invalid command \'" << cmd << "\'" << std::endl;
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
        << " start      Start recording. This will append the file if it\n"
        << "            already exists. It will create a new one if it doesn't.\n"
        << " pause      Pause recording. This will pause the recording\n"
        << "            without creating a new file.\n"
        << " new ARG    Start new file. User will be prompted for new file\n"
        << "            name.\n"
        << " quit       Exit the program.\n";
}

void printRemoteUsage(std::ostream &out) {

    out << "Recorder is under remote control.\n"
        << "Commands provided through STDIN have no effect\n"
        << "except Ctrl+C to quit.\n";
}

} /* namespace oat */
