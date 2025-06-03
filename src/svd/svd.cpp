/*
 * Image Quality Metrics
 * Petr Volf - 2025
 */

#include <IQM/svd.h>

static std::vector<uint32_t> srcConvert =
#include <svd/convert.inc>
;

static std::vector<uint32_t> srcSvd =
#include <svd/compute.inc>
;

static std::vector<uint32_t> srcReduce =
#include <svd/svd_reduce.inc>
;

static std::vector<uint32_t> srcSort =
#include <lib/multi_radixsort.inc>
;

static std::vector<uint32_t> srcSortHistogram =
#include <lib/multi_radixsort_histograms.inc>
;

static std::vector<uint32_t> srcSum =
#include <svd/msvd_sum.inc>
;

using IQM::GPU::VulkanRuntime;

IQM::SVD::SVD(const vk::raii::Device &device) {
    const auto smConvert = VulkanRuntime::createShaderModule(device, srcConvert);
    const auto smSvd = VulkanRuntime::createShaderModule(device, srcSvd);
    const auto smReduce = VulkanRuntime::createShaderModule(device, srcReduce);
    const auto smSort = VulkanRuntime::createShaderModule(device, srcSort);
    const auto smSortHistogram = VulkanRuntime::createShaderModule(device, srcSortHistogram);
    const auto smSum = VulkanRuntime::createShaderModule(device, srcSum);

    this->descPool = VulkanRuntime::createDescPool(device, 8, {
        vk::DescriptorPoolSize{.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = 24},
        vk::DescriptorPoolSize{.type = vk::DescriptorType::eStorageImage, .descriptorCount = 4},
    });

    this->descSetLayoutConvert = std::move(VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageImage, 2},
        {vk::DescriptorType::eStorageImage, 2},
    }));

    this->descSetLayoutSvd = std::move(VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageImage, 2},
        {vk::DescriptorType::eStorageBuffer, 1},
    }));

    this->descSetLayoutReduce = std::move(VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageBuffer, 1},
        {vk::DescriptorType::eStorageBuffer, 1},
    }));

    this->descSetLayoutSort = std::move(VulkanRuntime::createDescLayout(device, {
        {vk::DescriptorType::eStorageBuffer, 1},
        {vk::DescriptorType::eStorageBuffer, 1},
        {vk::DescriptorType::eStorageBuffer, 1},
    }));

    const std::vector layouts = {
        *this->descSetLayoutConvert,
        *this->descSetLayoutSvd,
        *this->descSetLayoutReduce,
        *this->descSetLayoutSort,
        *this->descSetLayoutSort,
        *this->descSetLayoutReduce,
        *this->descSetLayoutReduce,
        *this->descSetLayoutReduce,
    };

    vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo = {
        .descriptorPool = this->descPool,
        .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data()
    };

    auto createdLayouts = vk::raii::DescriptorSets{device, descriptorSetAllocateInfo};
    this->descSetConvert = std::move(createdLayouts[0]);
    this->descSetSvd = std::move(createdLayouts[1]);
    this->descSetReduce = std::move(createdLayouts[2]);
    this->descSetSortEven = std::move(createdLayouts[3]);
    this->descSetSortOdd = std::move(createdLayouts[4]);
    this->descSetSortHistogramEven = std::move(createdLayouts[5]);
    this->descSetSortHistogramOdd = std::move(createdLayouts[6]);
    this->descSetSum = std::move(createdLayouts[7]);

    // 1x int - buffer size
    const auto ranges = VulkanRuntime::createPushConstantRange(sizeof(int) * 1);
    const auto rangesSum = VulkanRuntime::createPushConstantRange(2 * sizeof(uint32_t));
    const auto rangesSort = VulkanRuntime::createPushConstantRange(4 * sizeof(uint32_t));

    this->layoutConvert = VulkanRuntime::createPipelineLayout(device, {this->descSetLayoutConvert}, {});
    this->layoutSvd = VulkanRuntime::createPipelineLayout(device, {this->descSetLayoutSvd}, {});
    this->layoutReduce = VulkanRuntime::createPipelineLayout(device, {this->descSetLayoutReduce}, ranges);
    this->layoutSort = VulkanRuntime::createPipelineLayout(device, {this->descSetLayoutSort}, {rangesSort});
    this->layoutSortHistogram = VulkanRuntime::createPipelineLayout(device, {this->descSetLayoutReduce}, {rangesSort});
    this->layoutSum = VulkanRuntime::createPipelineLayout(device, {this->descSetLayoutReduce}, {rangesSum});

    this->pipelineConvert = VulkanRuntime::createComputePipeline(device, smConvert, this->layoutConvert);
    this->pipelineSvd = VulkanRuntime::createComputePipeline(device, smSvd, this->layoutSvd);
    this->pipelineReduce = VulkanRuntime::createComputePipeline(device, smReduce, this->layoutReduce);
    this->pipelineSort = VulkanRuntime::createComputePipeline(device, smSort, this->layoutSort);
    this->pipelineSortHistogram = VulkanRuntime::createComputePipeline(device, smSortHistogram, this->layoutSortHistogram);
    this->pipelineSum = VulkanRuntime::createComputePipeline(device, smSum, this->layoutSum);
}

void IQM::SVD::computeMetric(const SVDInput &input) {
    this->initDescriptors(input);

    uint32_t outBufSize = (input.width / 8) * (input.height / 8);

    this->convertColorSpace(input);

    this->computeSvd(input);

    this->reduceSingularValues(input);

    auto bufCopy = vk::BufferCopy{
        .srcOffset = 0,
        .dstOffset = 0,
        .size = outBufSize * sizeof(float),
    };
    input.cmdBuf->copyBuffer(*input.bufReduce, *input.bufSort, {bufCopy});

    vk::MemoryBarrier barrier = {
        .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
    };

    input.cmdBuf->pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup,
        {barrier},
        {},
        {}
    );

    // need max and median for later steps
    this->sortBlocks(input);

    // after the sorting is done, reuse it's temp buffer to compute M-SVD
    bufCopy = vk::BufferCopy{
        .srcOffset = 0,
        .dstOffset = 0,
        .size = outBufSize * sizeof(float),
    };
    input.cmdBuf->copyBuffer(*input.bufReduce, *input.bufSortTemp, {bufCopy});

    barrier = {
        .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
    };

    input.cmdBuf->pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup,
        {barrier},
        {},
        {}
    );

    // now parallel sum
    this->computeMsvd(input);
}

void IQM::SVD::convertColorSpace(const SVDInput &input) {
    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelineConvert);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutConvert, 0, {this->descSetConvert}, {});

    //shaders work in 16x16 tiles
    auto [groupsX, groupsY] = VulkanRuntime::compute2DGroupCounts(input.width, input.height, 16);

    input.cmdBuf->dispatch(groupsX, groupsY, 1);

    vk::MemoryBarrier memoryBarrier = {
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
    };

    input.cmdBuf->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup, {memoryBarrier}, {}, {}
    );
}

void IQM::SVD::computeSvd(const SVDInput &input) {
    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelineSvd);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutSvd, 0, {this->descSetSvd}, {});

    input.cmdBuf->dispatch(input.width/8, input.height/8, 2);

    vk::MemoryBarrier memoryBarrier = {
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
    };

    input.cmdBuf->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup, {memoryBarrier}, {}, {}
    );
}

void IQM::SVD::reduceSingularValues(const SVDInput &input) {
    auto valueCount = (input.width / 8) * (input.height / 8);

    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelineReduce);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutReduce, 0, {this->descSetReduce}, {});
    input.cmdBuf->pushConstants<unsigned>(this->layoutReduce, vk::ShaderStageFlagBits::eCompute, 0, valueCount);

    // group takes 128 values, reduces to 8 values
    auto groupsX = (valueCount / 128) + 1;
    input.cmdBuf->dispatch(groupsX, 1, 1);

    vk::MemoryBarrier barrier = {
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eTransferRead,
    };

    input.cmdBuf->pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eTransfer,
        vk::DependencyFlagBits::eDeviceGroup,
        {barrier},
        {},
        {}
    );
}

void IQM::SVD::sortBlocks(const SVDInput &input) {
    auto valueCount = (input.width / 8) * (input.height / 8);
    uint32_t sortGlobalInvocationSize = valueCount / 32;
    uint32_t remainder = valueCount % 32;
    sortGlobalInvocationSize += remainder > 0 ? 1 : 0;

    uint32_t nSortWorkgroups = (sortGlobalInvocationSize + 256 - 1) / 256;
    uint32_t nBlocksPerWorkgroup = 32;

    vk::MemoryBarrier barrier = {
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
    };

    for (unsigned j = 0; j < 4; j++) {
        auto activeSet = j % 2 == 0 ? &this->descSetSortHistogramEven : &this->descSetSortHistogramOdd;
        input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelineSortHistogram);
        input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutSortHistogram, 0, {*activeSet}, {});

        std::array values = {
            valueCount,
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
}

void IQM::SVD::computeMsvd(const SVDInput &input) {
    auto valueCount = (input.width / 8) * (input.height / 8);

    input.cmdBuf->bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelineSum);
    input.cmdBuf->bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->layoutSum, 0, {this->descSetSum}, {});

    const auto sumSize = 1024;

    uint32_t bufferSize = valueCount;
    uint64_t groups = (bufferSize / sumSize) + 1;
    uint32_t size = bufferSize;
    uint32_t doDiff = 1;

    for (;;) {
        input.cmdBuf->pushConstants<unsigned>(this->layoutSum, vk::ShaderStageFlagBits::eCompute, 0, size);
        input.cmdBuf->pushConstants<unsigned>(this->layoutSum, vk::ShaderStageFlagBits::eCompute, sizeof(uint32_t), doDiff);
        input.cmdBuf->dispatch(groups, 1, 1);

        vk::BufferMemoryBarrier barrier = {
            .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead,
            .buffer = *input.bufSortTemp,
            .offset = 0,
            .size = bufferSize * sizeof(float),
        };
        input.cmdBuf->pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            vk::DependencyFlagBits::eDeviceGroup,
            {},
            {barrier},
            {}
        );
        if (groups == 1) {
            break;
        }
        size = groups;
        groups = (groups / sumSize) + 1;
        doDiff = 0;
    }
}

void IQM::SVD::initDescriptors(const SVDInput &input) {
    auto bufSize = 2 * 8 * (input.width / 8) * (input.height / 8);
    uint32_t outBufSize = (input.width / 8) * (input.height / 8);

    uint32_t sortGlobalInvocationSize = outBufSize / 32;
    uint32_t remainder = outBufSize % 32;
    sortGlobalInvocationSize += remainder > 0 ? 1 : 0;

    uint32_t nSortWorkgroups = (sortGlobalInvocationSize + 256 - 1) / 256;
    auto histBufSize = nSortWorkgroups * 256 * sizeof(uint32_t);

    std::vector bufInfos = {
        vk::DescriptorBufferInfo {
            .buffer = *input.bufSvd,
            .offset = 0,
            .range = bufSize * sizeof(float),
        },
        vk::DescriptorBufferInfo {
            .buffer = *input.bufReduce,
            .offset = 0,
            .range = outBufSize * sizeof(float),
        },
    };

    std::vector bufInfoSvd = {
        vk::DescriptorBufferInfo {
            .buffer = *input.bufSvd,
            .offset = 0,
            .range = bufSize * sizeof(float),
        }
    };

    auto bufInfoSort = std::vector {
        vk::DescriptorBufferInfo {
            .buffer = *input.bufSort,
            .offset = 0,
            .range = outBufSize * sizeof(float),
        }
    };

    auto bufInfoSortTemp = std::vector {
        vk::DescriptorBufferInfo {
            .buffer = *input.bufSortTemp,
            .offset = 0,
            .range = outBufSize * sizeof(float),
        }
    };

    auto bufInfoSortHist = std::vector {
        vk::DescriptorBufferInfo {
            .buffer = *input.bufSvd,
            .offset = 0,
            .range = histBufSize,
        }
    };

    const auto inputConvertImageInfos = VulkanRuntime::createImageInfos({input.ivTest, input.ivRef});
    const auto outConvertImageInfos = VulkanRuntime::createImageInfos({input.ivConvTest, input.ivConvRef});

    const auto writes = {
        // convert pass
        VulkanRuntime::createWriteSet(
            this->descSetConvert,
            0,
            inputConvertImageInfos
        ),
        VulkanRuntime::createWriteSet(
            this->descSetConvert,
            1,
            outConvertImageInfos
        ),
        // compute pass
        VulkanRuntime::createWriteSet(
            this->descSetSvd,
            0,
            outConvertImageInfos
        ),
        VulkanRuntime::createWriteSet(
            this->descSetSvd,
            1,
            bufInfoSvd
        ),
        // reduce pass
        VulkanRuntime::createWriteSet(
            this->descSetReduce,
            0,
            bufInfos
        ),
        // sort even
        VulkanRuntime::createWriteSet(
            this->descSetSortEven,
            0,
            bufInfoSort
        ),
        VulkanRuntime::createWriteSet(
            this->descSetSortEven,
            1,
            bufInfoSortTemp
        ),
        VulkanRuntime::createWriteSet(
            this->descSetSortEven,
            2,
            bufInfoSortHist
        ),
        // sort odd
        VulkanRuntime::createWriteSet(
            this->descSetSortOdd,
            0,
            bufInfoSortTemp
        ),
        VulkanRuntime::createWriteSet(
            this->descSetSortOdd,
            1,
            bufInfoSort
        ),
        VulkanRuntime::createWriteSet(
            this->descSetSortOdd,
            2,
            bufInfoSortHist
        ),
        // sort histogram even
        VulkanRuntime::createWriteSet(
            this->descSetSortHistogramEven,
            0,
            bufInfoSort
        ),
        VulkanRuntime::createWriteSet(
            this->descSetSortHistogramEven,
            1,
            bufInfoSortHist
        ),
        // sort histogram odd
        VulkanRuntime::createWriteSet(
            this->descSetSortHistogramOdd,
            0,
            bufInfoSortTemp
        ),
        VulkanRuntime::createWriteSet(
            this->descSetSortHistogramOdd,
            1,
            bufInfoSortHist
        ),
        // sum
        VulkanRuntime::createWriteSet(
            this->descSetSum,
            0,
            bufInfoSortTemp
        ),
        VulkanRuntime::createWriteSet(
            this->descSetSum,
            1,
            bufInfoSort
        ),
    };

    input.device->updateDescriptorSets(writes, nullptr);
}