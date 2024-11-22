#ifndef TIMESTAMPS_H
#define TIMESTAMPS_H

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

class Timestamps {
public:
    std::vector<std::pair<std::string, std::chrono::time_point<std::chrono::system_clock>>> inner;
    void mark(const std::string& name) {
        inner.emplace_back(name, std::chrono::high_resolution_clock::now());
    }
    void print(
        const std::chrono::time_point<std::chrono::system_clock> start,
        const std::chrono::time_point<std::chrono::system_clock> end
    ) {
        auto execTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        int longestName = 0;
        for (const auto& [name, _time] : inner) {
            longestName = std::max(longestName, static_cast<int>(name.length()));
        }

        int timePad = static_cast<int>(std::ceil(std::log10(execTime.count()))) + 2;

        for (const auto& [name, time] : inner) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(time - start);
            std::cout << std::setw(longestName) << name << ": " << std::setw(timePad) << duration << std::endl;
        }

        std::cout << std::setw(longestName) << "TOTAL" << ": " << std::setw(timePad) << execTime << std::endl;
    }
};

#endif //TIMESTAMPS_H
