#include <boost/asio/steady_timer.hpp>
#include <boost/asio/io_service.hpp>
#include <iostream>
#include <chrono>

using namespace boost::asio;

int main() {
    
    io_service io;

    steady_timer timer0 {io, std::chrono::seconds {3} };
    steady_timer timer1 {io, std::chrono::seconds {4} };

    timer0.async_wait([](const boost::system::error_code& ec) {
        std::cout << "3 seconds.\n";
    });

    timer1.async_wait([](const boost::system::error_code& ec) {
        std::cout << "4 seconds.\n";
    });

    std::cout << "async_wait returned.\n";

    io.run();

    return 0;
}

