//******************************************************************************
//* File:   IOFormat.h
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

#ifndef OAT_IOFORMAT_H
#define OAT_IOFORMAT_H

#include <iomanip>
#include <iostream>
#include <string>
#include <unistd.h>

#define RST  "\x1B[0m"

#define FRED(x) "\x1B[31m" + x + RST
#define FGRN(x) "\x1B[32m" + x + RST
#define FYEL(x) "\x1B[33m" + x + RST
#define FBLU(x) "\x1B[34m" + x + RST
#define FMAG(x) "\x1B[35m" + x + RST
#define FCYN(x) "\x1B[36m" + x + RST
#define FWHT(x) "\x1B[37m" + x + RST

#define BOLD(x) "\x1B[1m" + x + RST
#define UNDL(x) "\x1B[4m" + x + RST

#define HAZ_COLOR isatty(fileno(stdout)) 

namespace oat {

inline std::string bold(const std::string &message) {

    return HAZ_COLOR ? BOLD(message) : message; 
}

inline std::string sourceText(const std::string &source_name) {

    return HAZ_COLOR ? FGRN(source_name) : source_name; 
}

inline std::string sinkText(const std::string& sink_name) {

    return HAZ_COLOR ? FRED(sink_name) : sink_name; 
}

inline std::string whoMessage(const std::string& source, const std::string& message) {

    return (HAZ_COLOR ? BOLD(source) : source) + ": " + message; 
}

inline std::string Warn(const std::string& message) {

    return HAZ_COLOR ? FYEL(message) : message; 
}

inline std::string Error(const std::string& message) {

    return HAZ_COLOR ? FRED(message) : message; 
}

inline std::string dbgMessage(const std::string& message) {

    return HAZ_COLOR ? FMAG(message) : message; 
}

inline std::string whoWarn(const std::string& source, const std::string& message) {

    return HAZ_COLOR ? 
        BOLD(source) + ": " + FYEL(message) 
        : 
        source +  ": " + message; 
}

inline std::string whoError(const std::string& source, const std::string& message) {

    return HAZ_COLOR ? 
        BOLD(source) + ": " + FRED(message) 
        : 
        source +  ": " + message; 
}

inline std::string configNoTableError(const std::string& table_name,
                                      const std::string& config_file) {

    return  "No configuration table named '" + table_name +
            "' was provided in the configuration file '" + config_file + "'";
}

inline std::string configValueError(const std::string& entry_name,
                                    const std::string& table_name,
                                    const std::string& config_file,
                                    const std::string& message) {

    return "'" + entry_name + "' in '" + table_name + "' in '" + config_file + "' " + message;
}

}      /* namespace oat */
#endif /* OAT_IOFORMAT_H */
