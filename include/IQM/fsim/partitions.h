/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#ifndef IQM_FSIM_PARTITIONS_H
#define IQM_FSIM_PARTITIONS_H

#include <cstdint>

namespace IQM {
    /**
     * Fft buffer gets heavy reuse.
     * Store offsets here for easier reasoning.
     */
    struct FftBufferPartitions {
        uint64_t sort;
        uint64_t sortTemp;
        uint64_t sortHist;
        uint64_t noiseLevels;
        uint64_t noisePowers;
        uint64_t end;
    };
}

#endif //IQM_FSIM_PARTITIONS_H
