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

#include "../../lib/base/Component.h"
#include "../../lib/utility/in_place.h"

namespace po = boost::program_options;

namespace oat {

// Viewer<T> type erasure
class ViewerBase {

public:
    template <typename T, typename... Args>
    ViewerBase(oat::in_place<T>, Args&&... args) :
      viewer(std::make_shared<ViewerModel<T>>(std::forward<Args>(args)...)) { }
    void appendOptions(po::options_description &opts) const
        { viewer->appendOptions(opts); }
    void configure(const po::variables_map &vm) const
        { viewer->configure(vm); }
    void run() { viewer->run(); }
    std::string name() const { return viewer->name(); }
    oat::ComponentType type() const { return viewer->type(); }

private:
    struct ViewerConcept {
        virtual ~ViewerConcept() { }
        virtual void appendOptions(po::options_description &opts) = 0;
        virtual void configure(const po::variables_map &vm) = 0;
        virtual void run(void) = 0;
        virtual std::string name(void) const = 0;
        virtual oat::ComponentType type(void) const = 0;
    };

    template <typename T>
    struct ViewerModel : ViewerConcept {
        template <typename... Args>
        ViewerModel(Args&&... args) : viewer(std::forward<Args>(args)...) { }
        void appendOptions(po::options_description &opts) override
            { viewer.appendOptions(opts); }
        void configure(const po::variables_map &vm) override
            { viewer.configure(vm); }
        void run() override { viewer.run(); }
        std::string name() const override { return viewer.name(); }
        oat::ComponentType type() const override { return viewer.type(); }
    private:
        T viewer;
    };

    // Destructor of shared_ptr is type erased
    std::shared_ptr<ViewerConcept> viewer;
};

}      /* namespace oat */
#endif /* OAT_VIEWERBASE_H */
