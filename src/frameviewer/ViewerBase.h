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

namespace po = boost::program_options;

namespace oat {

//template<typename T>
//struct FakeCopyable
//{
//    FakeCopyable(T&& t) : target(std::forward<T>(t)) { }
//
//    FakeCopyable(FakeCopyable&&) = default;
//    FakeCopyable(const FakeCopyable&) { throw std::logic_error("Cannot copy."); }
//
//    template<typename... Args>
//    T operator()(Args&&... a) { return target(std::forward<Args>(a)...); }
//
//    T target;
//};
//
//template<typename T>
//FakeCopyable<T> fake_copyable(T&& t) { return { std::forward<T>(t) }; }

template <typename T> 
struct in_place {};

class ViewerBase {

    struct ViewerConcept {
        virtual ~ViewerConcept() { }
        virtual void appendOptions(po::options_description &opts) const = 0;
        virtual void configure(const po::variables_map &vm) = 0;
        virtual void connectToNode(void) = 0;
        virtual bool process(void) = 0;
        virtual std::string name(void) const = 0;
    };

    template<typename T>
    struct ViewerModel : ViewerConcept {
        template <typename... Args>
        ViewerModel(Args&&... args) : viewer(std::forward<Args>(args)...) { }
        void appendOptions(po::options_description &opts) const 
            { viewer.appendOptions(opts); }
        void configure(const po::variables_map &vm) { viewer.configure(vm); }
        void connectToNode(void) { viewer.connectToNode();}
        bool process(void) { return viewer.process(); }
        std::string name() const { return viewer.name(); }
    private:
        T viewer;
    };

    //TODO: Unique ptr?
    std::shared_ptr<ViewerConcept> viewer;

public:

    template<typename T, typename... Args> ViewerBase(in_place<T>, Args&&... args) :
        viewer(new ViewerModel<T>(std::forward<Args>(args)...)) { }

    std::string name() const { return viewer->name(); }

    void appendOptions(po::options_description &opts) const {
        viewer->appendOptions(opts);
    }

    void configure(const po::variables_map &vm) const {
        viewer->configure(vm);
    }

    void connectToNode(void) { viewer->connectToNode(); }

    bool process(void) { return viewer->process(); }
};

}      /* namespace oat */
#endif /* OAT_VIEWERBASE_H */
