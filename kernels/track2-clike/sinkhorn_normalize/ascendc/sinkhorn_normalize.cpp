#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t kBlockCount = 32;
constexpr uint32_t kMatricesPerBlock = 32;
constexpr uint32_t kMatrixElements = 16;
constexpr uint32_t kElementsPerBlock = kMatricesPerBlock * kMatrixElements;
constexpr float kEps = 1.0e-6f;

class SinkhornNormalizeKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y)
    {
        const uint32_t block = GetBlockIdx();
        const uint32_t offset = block * kElementsPerBlock;
        xGm_.SetGlobalBuffer((__gm__ float*)x + offset, kElementsPerBlock);
        yGm_.SetGlobalBuffer((__gm__ float*)y + offset, kElementsPerBlock);
        pipe_.InitBuffer(stateBuf_, kElementsPerBlock * sizeof(float));
        state_ = stateBuf_.Get<float>();
    }

    __aicore__ inline void Process()
    {
        DataCopy(state_, xGm_, kElementsPerBlock);
        SoftmaxRows();
        NormalizeColumns();
        for (uint32_t iteration = 1; iteration < 10; ++iteration) {
            NormalizeRows();
            NormalizeColumns();
        }
        DataCopy(yGm_, state_, kElementsPerBlock);
    }

private:
    __aicore__ inline void SoftmaxRows()
    {
        for (uint32_t matrix = 0; matrix < kMatricesPerBlock; ++matrix) {
            const uint32_t matrixBase = matrix * kMatrixElements;
            for (uint32_t row = 0; row < 4; ++row) {
                const uint32_t rowBase = matrixBase + row * 4;
                float maxValue = state_.GetValue(rowBase);
                for (uint32_t col = 1; col < 4; ++col) {
                    const float value = state_.GetValue(rowBase + col);
                    maxValue = value > maxValue ? value : maxValue;
                }
                for (uint32_t col = 0; col < 4; ++col) {
                    state_.SetValue(rowBase + col, state_.GetValue(rowBase + col) - maxValue);
                }
            }
        }
        PipeBarrier<PIPE_V>();
        Exp(state_, state_, kElementsPerBlock);
        PipeBarrier<PIPE_V>();

        for (uint32_t matrix = 0; matrix < kMatricesPerBlock; ++matrix) {
            const uint32_t matrixBase = matrix * kMatrixElements;
            for (uint32_t row = 0; row < 4; ++row) {
                const uint32_t rowBase = matrixBase + row * 4;
                float sum = 0.0f;
                for (uint32_t col = 0; col < 4; ++col) {
                    sum += state_.GetValue(rowBase + col);
                }
                const float inv = 1.0f / sum;
                for (uint32_t col = 0; col < 4; ++col) {
                    state_.SetValue(rowBase + col, state_.GetValue(rowBase + col) * inv + kEps);
                }
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void NormalizeRows()
    {
        for (uint32_t matrix = 0; matrix < kMatricesPerBlock; ++matrix) {
            const uint32_t matrixBase = matrix * kMatrixElements;
            for (uint32_t row = 0; row < 4; ++row) {
                const uint32_t rowBase = matrixBase + row * 4;
                float sum = 0.0f;
                for (uint32_t col = 0; col < 4; ++col) {
                    sum += state_.GetValue(rowBase + col);
                }
                const float inv = 1.0f / (sum + kEps);
                for (uint32_t col = 0; col < 4; ++col) {
                    state_.SetValue(rowBase + col, state_.GetValue(rowBase + col) * inv);
                }
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void NormalizeColumns()
    {
        for (uint32_t matrix = 0; matrix < kMatricesPerBlock; ++matrix) {
            const uint32_t matrixBase = matrix * kMatrixElements;
            for (uint32_t col = 0; col < 4; ++col) {
                float sum = 0.0f;
                for (uint32_t row = 0; row < 4; ++row) {
                    sum += state_.GetValue(matrixBase + row * 4 + col);
                }
                const float inv = 1.0f / (sum + kEps);
                for (uint32_t row = 0; row < 4; ++row) {
                    const uint32_t index = matrixBase + row * 4 + col;
                    state_.SetValue(index, state_.GetValue(index) * inv);
                }
            }
        }
        PipeBarrier<PIPE_V>();
    }

    TPipe pipe_;
    TBuf<QuePosition::VECCALC> stateBuf_;
    LocalTensor<float> state_;
    GlobalTensor<float> xGm_;
    GlobalTensor<float> yGm_;
};
}

extern "C" __global__ __aicore__ void sinkhorn_normalize(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (GetBlockIdx() >= kBlockCount) {
        return;
    }
    SinkhornNormalizeKernel kernel;
    kernel.Init(x, y);
    kernel.Process();
}
