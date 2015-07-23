#include <boost/asio/write.hpp>
#include <boost/asio/buffer.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/io_service.hpp>
#include <iostream>
#include <chrono>
#include <array>
#include <string>

using namespace boost::asio;
using namespace boost::asio::ip;
    
io_service io;
tcp::resolver res {io};
tcp::socket soc {io};
std::array<char, 4096> bytes;


void read_handler(const boost::system::error_code& ec, std::size_t bytes_tx) {

    if (!ec) {
        std::cout.write(bytes.data(), bytes_tx);
        soc.async_read_some(buffer(bytes), read_handler);
    }
}

void connect_handler(const boost::system::error_code& ec) {
    if (!ec) {
        std::string r = "GET / HTTP/1.1\r\nHost: theboostcpplibraries.com\r\n\r\n";
        write(soc, buffer(r));
        soc.async_read_some(buffer(bytes), read_handler);
    }
}

void resolve_handler(const boost::system::error_code& ec, tcp::resolver::iterator it) {

    if (!ec) {
        soc.async_connect(*it, connect_handler);
    }
}

int main() {

    tcp::resolver::query q {"theboostcpplibraries.com", "80"};
    res.async_resolve(q, resolve_handler);
    std::cout << "Running...\n";

    io.run();

    return 0;
}

