//******************************************************************************
//* File:   ViewerBase.h
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

#ifndef OAT_VIEWERBASE_H
#define OAT_VIEWERBASE_H

#include <memory>
#include <boost/program_options.hpp>

#include "../../lib/utility/in_place.h"

namespace po = boost::program_options;

namespace oat {

class ViewerBase {

public:

    template<typename T, typename... Args> 
    ViewerBase(std::in_place<T>, Args&&... args) :
      viewer(new ViewerModel<T>(std::forward<Args>(args)...)) { }
    void appendOptions(po::options_description &opts) const 
        { viewer->appendOptions(opts); }
    void configure(const po::variables_map &vm) const 
        { viewer->configure(vm); }
    void connectToNode(void) const { viewer->connectToNode(); }
    bool process(void) const { return viewer->process(); }
    std::string name() const { return viewer->name(); }
    
private:

    struct ViewerConcept {
        virtual ~ViewerConcept() { }
        virtual void appendOptions(po::options_description &opts) = 0;
        virtual void configure(const po::variables_map &vm) = 0;
        virtual void connectToNode(void) = 0;
        virtual bool process(void) = 0;
        virtual std::string name(void) const = 0;
    };

    template<typename T>
    struct ViewerModel : ViewerConcept {
        template <typename... Args>
        ViewerModel(Args&&... args) : viewer(std::forward<Args>(args)...) { }
        void appendOptions(po::options_description &opts) override 
            { viewer.appendOptions(opts); }
        void configure(const po::variables_map &vm) override 
            { viewer.configure(vm); }
        void connectToNode(void) override { viewer.connectToNode();}
        bool process(void) override { return viewer.process(); }
        std::string name() const override { return viewer.name(); }
    private:
        T viewer;
    };

    std::unique_ptr<ViewerConcept> viewer;
};
}      /* namespace oat */
#endif /* OAT_VIEWERBASE_H */
