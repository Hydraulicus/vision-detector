#include "preprocessor.h"

#include <cmath>
#include <algorithm>

namespace vision_detector {

Preprocessor::Preprocessor() = default;

Preprocessor::~Preprocessor() = default;

void Preprocessor::setTargetSize(int width, int height) {
    target_width_ = width;
    target_height_ = height;
    resize_buffer_.resize(width * height * 3);
}

void Preprocessor::process(
    const uint8_t* input_rgb,
    int input_width,
    int input_height,
    int input_stride,
    std::vector<float>& output
) {
    // Resize if needed
    const uint8_t* data_to_normalize = input_rgb;
    int width = input_width;
    int height = input_height;

    if (input_width != target_width_ || input_height != target_height_) {
        resize(input_rgb, input_width, input_height, input_stride,
               resize_buffer_.data(), target_width_, target_height_);
        data_to_normalize = resize_buffer_.data();
        width = target_width_;
        height = target_height_;
    }

    // Normalize
    normalize(data_to_normalize, width, height, output);
}

void Preprocessor::resize(
    const uint8_t* input,
    int in_w, int in_h, int in_stride,
    uint8_t* output,
    int out_w, int out_h
) {
    // Bilinear interpolation
    float x_ratio = static_cast<float>(in_w) / out_w;
    float y_ratio = static_cast<float>(in_h) / out_h;

    for (int y = 0; y < out_h; ++y) {
        float src_y = y * y_ratio;
        int y0 = static_cast<int>(src_y);
        int y1 = std::min(y0 + 1, in_h - 1);
        float y_lerp = src_y - y0;

        for (int x = 0; x < out_w; ++x) {
            float src_x = x * x_ratio;
            int x0 = static_cast<int>(src_x);
            int x1 = std::min(x0 + 1, in_w - 1);
            float x_lerp = src_x - x0;

            for (int c = 0; c < 3; ++c) {
                // Get 4 neighboring pixels
                float p00 = input[y0 * in_stride + x0 * 3 + c];
                float p01 = input[y0 * in_stride + x1 * 3 + c];
                float p10 = input[y1 * in_stride + x0 * 3 + c];
                float p11 = input[y1 * in_stride + x1 * 3 + c];

                // Bilinear interpolation
                float top = p00 + x_lerp * (p01 - p00);
                float bottom = p10 + x_lerp * (p11 - p10);
                float value = top + y_lerp * (bottom - top);

                output[y * out_w * 3 + x * 3 + c] =
                    static_cast<uint8_t>(std::clamp(value, 0.0f, 255.0f));
            }
        }
    }
}

void Preprocessor::normalize(
    const uint8_t* input,
    int width, int height,
    std::vector<float>& output
) {
    output.resize(width * height * 3);

    // Normalization for MobileNet SSD and similar models: [0, 255] -> [-1, 1]
    // Formula: (pixel - 127.5) / 127.5
    constexpr float mean = 127.5f;
    constexpr float std = 127.5f;

    for (int i = 0; i < width * height * 3; ++i) {
        output[i] = (static_cast<float>(input[i]) - mean) / std;
    }
}

}  // namespace vision_detector