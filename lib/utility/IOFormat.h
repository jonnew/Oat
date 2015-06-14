//******************************************************************************
//* Copyright (c) Jon Newman (jpnewman snail mit dot edu) 
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

#ifndef IOFORMAT_H
#define	IOFORMAT_H

#include <string>

namespace oat {

    inline std::string bold(std::string message) {

        return "\e[1m" + message + "\e[0m";
    }

    inline std::string boldSource(std::string source_name) {

        return "\e[34;1m" + source_name + "\e[0m";
    }

    inline std::string boldSink(std::string sink_name) {

        return "\e[35;1m" + sink_name + "\e[0m";
    }

    inline std::string whoMessage(std::string source, std::string message) {

        return "\e[1m" + source + ": \e[0m" + message;
    }

    inline std::string whoWarn(std::string source, std::string message) {

        return "\e[1m" + source + ": \e[0m\e[33m" + message + "\e[0m";
    }

    inline std::string whoError(std::string source, std::string message) {

        return "\e[1m" + source + ": \e[0m\e[31m" + message + "\e[0m";
    }
    
    inline std::string dbgMessage(std::string message) {

        return "\e[35m" + message + "\e[0m";
    }
}

#endif	/* IOFORMAT_H */
