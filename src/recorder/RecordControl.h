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
//*****************************************************************************

#ifndef OAT_RECORDCONTROL_H
#define OAT_RECORDCONTROL_H

//#include "Recorder.h"

// TODO: Implementations to separate file
namespace oat {
   
    class recorder;

    void startRecording(oat::recorder &recorder) {
        recorder.set_record_on(true);
    }

    void pauseRecordering(oat::recorder &recorder) {
        recorder.set_record_on(false);
    }
   
    void createNewRecordingFile(oat::recorder &recorder) {

    }

    // TODO: Not that I care right now, but I note that this is POSIX specific
    // and would need some preprocessors stuff for other platforms
    int interact(std::istream &in, oat::recorder &recorder, pthread_t process_thread) {
   
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
            std::getline(std::cin, cmd);
            //if (!(std::cin >> cmd))
            //    oat::ignoreLine(std::cin);
    
            switch (cmd_map[cmd]) {
                case 's' :
                {
                    startRecording();
                    std::cout << "Recording ON.\n";
                    break;
                }
                case 'S' :
                {
                    pauseRecordering();
                    std::cout << "Recording OFF.\n";
                    break;
                }
                case 'e' :
                {
                    quit = true;
                    pthread_kill(process_thread, SIGINT);
                    break;
                }
                default :
                {
                    // Flush cin in case the uer just inserted crap
                    //oat::ignoreLine(std::cin);
                    std::cerr << "Invalid command \'" << cmd << "\'\n";
                    break;
                }
            }
        }
    
        return 0;
    }

}      /* namespace oat */
#endif /* OAT_RECORDCONTROL_H */

