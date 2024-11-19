#ifndef TIMESTAMPS_H
#define TIMESTAMPS_H

#include <chrono>
#include <string>
#include <vector>

class Timestamps {
public:
    std::vector<std::pair<std::string, std::chrono::time_point<std::chrono::system_clock>>> inner;
    void mark(const std::string& name) {
        inner.emplace_back(name, std::chrono::high_resolution_clock::now());
    }
};

#endif //TIMESTAMPS_H
