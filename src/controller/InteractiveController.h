//******************************************************************************
//* File:   InteractiveController.h
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
//****************************************************************************

#ifndef OAT_INTERACTIVECONTROLLER_H
#define OAT_INTERACTIVECONTROLLER_H

#include "Controller.h"

#include <atomic>
#include <mutex>
#include <thread>

namespace oat {

class InteractiveController : public Controller {

public:
    using Controller::Controller;
    void execute();

private:
    std::atomic<bool> break_scan_{false};
    std::mutex socket_mtx_;
    std::thread scan_thread_;

    void scanLoop();
};

}      /* namespace oat */
#endif /* OAT_CONTROLLER_H */

