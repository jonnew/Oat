//******************************************************************************
//* Document number: N3656
//* Date: 2013-04-18
//* Project: Programming Language C++, Library Working Group
//* Reply-to: Stephan T. Lavavej <stl@microsoft.com>
//*****************************************************************************

// make_unique<T>(args...)
// make_unique<T[]>(n)
// make_unique<T[N]>(args...) = delete

#ifndef OAT_MAKEUNIQUE_H
#define OAT_MAKEUNIQUE_H

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

namespace oat {

    using namespace std;

    template<class T> struct _Unique_if {
        typedef unique_ptr<T> _Single_object;
    };

    template<class T> struct _Unique_if<T[]> {
        typedef unique_ptr<T[]> _Unknown_bound;
    };

    template<class T, size_t N> struct _Unique_if<T[N]> {
        typedef void _Known_bound;
    };

    template<class T, class... Args>
        typename _Unique_if<T>::_Single_object
        make_unique(Args&&... args) {
            return unique_ptr<T>(new T(std::forward<Args>(args)...));
        }

    template<class T>
        typename _Unique_if<T>::_Unknown_bound
        make_unique(size_t n) {
            typedef typename remove_extent<T>::type U;
            return unique_ptr<T>(new U[n]());
        }

    template<class T, class... Args>
        typename _Unique_if<T>::_Known_bound
        make_unique(Args&&...) = delete;
}
#endif /* OAT_MAKEUNIQUE_H */
