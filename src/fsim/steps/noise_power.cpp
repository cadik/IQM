/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include <IQM/fsim/noise_power.h>
#include <IQM/fsim.h>

static std::vector<uint32_t> src =
#include <fsim/fsim_pack_for_median.inc>
;

static std::vector<uint32_t> srcSort =
#include <lib/multi_radixsort.inc>
;

static std::vector<uint32_t> srcSortHistogram =
#include <lib/multi_radixsort_histograms.inc>
;

static std::vector<uint32_t> srcOut =
#include <fsim/fsim_noise_power.inc>
;

using IQM::GPU::VulkanRuntime;

IQM::FSIMNoisePower::FSIMNoisePower(const vk::raii::Device &device, const vk::raii::DescriptorPool& descPool) {
    const auto smPack = VulkanRuntime::createShaderModule(device, src);
    const auto smSort = VulkanRuntime::createShaderModule(device, srcSort);
    const auto smSortHistogram = VulkanRuntime::createShaderModule(device, srcSortHistogram);
    const auto smNoisePower = VulkanRuntime::createShaderModule(device, srcOut);

    this->descSetLayout = VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageBuffer, 1},
        {vk::DescriptorType::eStorageBuffer, 1},
    });

    this->descSetLayoutSort = VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageBuffer, 1},
        {vk::DescriptorType::eStorageBuffer, 1},
        {vk::DescriptorType::eStorageBuffer, 1},
    });

    const std::vector layouts = {
        *this->descSetLayout,
        *this->descSetLayoutSort,
        *this->descSetLayoutSort,
        *this->descSetLayout,
        *this->descSetLayout,
        *this->descSetLayoutSort,
    };

    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo = {
        .descriptorPool = descPool,
        .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data()
    };

    auto createdLayouts = vk::raii::DescriptorSets{device, descriptorSetAllocateInfo};
    this->descSet = std::move(createdLayouts[0]);
    this->descSetSortEven = std::move(createdLayouts[1]);
    this->descSetSortOdd = std::move(createdLayouts[2]);
    this->descSetSortHistogramEven = std::move(createdLayouts[3]);
    this->descSetSortHistogramOdd = std::move(createdLayouts[4]);
    this->descSetNoisePower = std::move(createdLayouts[5]);

    // 2x uint - buffer size, index
    const auto ranges = VulkanRuntime::createPushConstantRange(2 * sizeof(uint32_t));
    const auto rangesSort = VulkanRuntime::createPushConstantRange(4 * sizeof(uint32_t));

    this->layout = VulkanRuntime::createPipelineLayout(device, {this->descSetLayout}, {ranges});
    this->layoutSort = VulkanRuntime::createPipelineLayout(device, {this->descSetLayoutSort}, {rangesSort});
    this->layoutSortHistogram = VulkanRuntime::createPipelineLayout(device, {this->descSetLayout}, {rangesSort});
    this->layoutNoisePower = VulkanRuntime::createPipelineLayout(device, {this->descSetLayoutSort}, {ranges});

    this->pipeline = VulkanRuntime::createComputePipeline(device, smPack, this->layout);
    this->pipelineSort = VulkanRuntime::createComputePipeline(device, smSort, this->layoutSort);
    this->pipelineSortHistogram = VulkanRuntime::createComputePipeline(device, smSortHistogram, this->layoutSortHistogram);
    this->pipelineNoisePower = VulkanRuntime::createComputePipeline(device, smNoisePower, this->layoutNoisePower);
}

void IQM::FSIMNoisePower::computeNoisePower(const FSIMInput &input, const unsigned width, const unsigned height) {
    uint32_t sortGlobalInvocationSize = (width * height) / 32;
    uint32_t remainder = (width * height) % 32;
    sortGlobalInvocationSize += remainder > 0 ? 1 : 0;

    uint32_t nSortWorkgroups = (sortGlobalInvocationSize + 256 - 1) / 256;
    uint32_t nBlocksPerWorkgroup = 32;

    vk::MemoryBarrier barrier = {
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
    };

    for (int i = 0; i < FSIM_ORIENTATIONS * 2; i++) {
        input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipeline);
        input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layout, 0, {this->descSet}, {});

        input.cmdBuf->pushConstants<unsigned>(this->layout, vk::ShaderStageFlagBits::eCompute, 0, width * height);
        input.cmdBuf->pushConstants<unsigned>(this->layout, vk::ShaderStageFlagBits::eCompute, sizeof(uint32_t),i);

        auto groups = (width * height) / 256 + 1;

        input.cmdBuf->dispatch(groups, 1, 1);

        input.cmdBuf->pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            vk::DependencyFlagBits::eDeviceGroup,
            {barrier},
            {},
            {}
        );

        for (unsigned j = 0; j < 4; j++) {
            auto activeSet = j % 2 == 0 ? &this->descSetSortHistogramEven : &this->descSetSortHistogramOdd;
            input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelineSortHistogram);
            input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutSortHistogram, 0, {*activeSet}, {});

            std::array values = {
                static_cast<unsigned>(width * height),
                j * 8,
                nSortWorkgroups,
                nBlocksPerWorkgroup,
            };
            input.cmdBuf->pushConstants<unsigned>(this->layoutSortHistogram, vk::ShaderStageFlagBits::eCompute, 0, values);

            input.cmdBuf->dispatch(sortGlobalInvocationSize, 1, 1);

            input.cmdBuf->pipelineBarrier(
                vk::PipelineStageFlagBits::eComputeShader,
                vk::PipelineStageFlagBits::eComputeShader,
                vk::DependencyFlagBits::eDeviceGroup,
                {barrier},
                {},
                {}
            );

            activeSet = j % 2 == 0 ? &this->descSetSortEven : &this->descSetSortOdd;
            input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelineSort);
            input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutSort, 0, {*activeSet}, {});

            input.cmdBuf->pushConstants<unsigned>(this->layoutSort, vk::ShaderStageFlagBits::eCompute, 0, values);

            input.cmdBuf->dispatch(sortGlobalInvocationSize, 1, 1);

            input.cmdBuf->pipelineBarrier(
                vk::PipelineStageFlagBits::eComputeShader,
                vk::PipelineStageFlagBits::eComputeShader,
                vk::DependencyFlagBits::eDeviceGroup,
                {barrier},
                {},
                {}
            );
        }

        input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelineNoisePower);
        input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutNoisePower, 0, {this->descSetNoisePower}, {});

        input.cmdBuf->pushConstants<unsigned>(this->layoutNoisePower, vk::ShaderStageFlagBits::eCompute, 0, width * height);
        input.cmdBuf->pushConstants<unsigned>(this->layoutNoisePower, vk::ShaderStageFlagBits::eCompute, sizeof(uint32_t),i);

        input.cmdBuf->dispatch(1, 1, 1);

        input.cmdBuf->pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            vk::DependencyFlagBits::eDeviceGroup,
            {barrier},
            {},
            {}
        );
    }
}

void IQM::FSIMNoisePower::setUpDescriptors(const FSIMInput &input, const unsigned width, const unsigned height, const FftBufferPartitions& partitions) const {
    auto bufInfoIn = std::vector {
        vk::DescriptorBufferInfo {
            .buffer = *input.bufIfft,
            .offset = 0,
            .range = 2 * width * height * sizeof(float) * FSIM_ORIENTATIONS * FSIM_SCALES * 3,
        }
    };

    // sorting can be safely done in FFT buffer, since it must be big enough
    auto bufInfoOut = std::vector {
        vk::DescriptorBufferInfo {
            .buffer = *input.bufFft,
            .offset = 0,
            .range = partitions.sortTemp - partitions.sort,
        }
    };

    auto bufInfoSortTemp = std::vector {
        vk::DescriptorBufferInfo {
            .buffer = *input.bufFft,
            .offset = partitions.sortTemp,
            .range = partitions.sortHist - partitions.sortTemp,
        }
    };

    auto bufInfoSortHist = std::vector {
        vk::DescriptorBufferInfo {
            .buffer = *input.bufFft,
            .offset = partitions.sortHist,
            .range = partitions.noiseLevels - partitions.sortHist,
        }
    };

    // this is saved just after the space needed for sort
    auto bufInfoFilterSums = std::vector {
        vk::DescriptorBufferInfo {
            .buffer = *input.bufFft,
            .offset = partitions.noiseLevels,
            .range = partitions.noisePowers - partitions.noiseLevels,
        }
    };

    // store even further
    auto bufInfoNoisePower = std::vector {
        vk::DescriptorBufferInfo {
            .buffer = **input.bufFft,
            .offset = partitions.noisePowers,
            .range = partitions.end - partitions.noisePowers,
        }
    };

    const auto writes = {
        // copy pass
        VulkanRuntime::createWriteSet(this->descSet, 0, bufInfoIn),
        VulkanRuntime::createWriteSet(this->descSet, 1, bufInfoOut),
        // sort even
        VulkanRuntime::createWriteSet(this->descSetSortEven, 0, bufInfoOut),
        VulkanRuntime::createWriteSet(this->descSetSortEven, 1, bufInfoSortTemp),
        VulkanRuntime::createWriteSet(this->descSetSortEven,2, bufInfoSortHist),
        // sort odd
        VulkanRuntime::createWriteSet(this->descSetSortOdd, 0, bufInfoSortTemp),
        VulkanRuntime::createWriteSet(this->descSetSortOdd, 1, bufInfoOut),
        VulkanRuntime::createWriteSet(this->descSetSortOdd, 2, bufInfoSortHist),
        // sort histogram even
        VulkanRuntime::createWriteSet(this->descSetSortHistogramEven, 0, bufInfoOut),
        VulkanRuntime::createWriteSet(this->descSetSortHistogramEven, 1, bufInfoSortHist),
        // sort histogram odd
        VulkanRuntime::createWriteSet(this->descSetSortHistogramOdd, 0, bufInfoSortTemp),
        VulkanRuntime::createWriteSet(this->descSetSortHistogramOdd, 1, bufInfoSortHist),
        // noise power
        VulkanRuntime::createWriteSet(this->descSetNoisePower, 0, bufInfoOut),
        VulkanRuntime::createWriteSet(this->descSetNoisePower, 1, bufInfoFilterSums),
        VulkanRuntime::createWriteSet(this->descSetNoisePower, 2, bufInfoNoisePower),
    };

    input.device->updateDescriptorSets(writes, nullptr);
}