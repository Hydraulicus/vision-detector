#ifndef VISION_DETECTOR_PREPROCESSOR_H
#define VISION_DETECTOR_PREPROCESSOR_H

#include <cstdint>
#include <vector>

namespace vision_detector {

class Preprocessor {
public:
    Preprocessor();
    ~Preprocessor();

    // Configure target size for model input
    void setTargetSize(int width, int height);

    // Configure normalization range
    // SSD MobileNet: min=-1, max=1 (formula: (pixel - 127.5) / 127.5)
    // YOLOv8: min=0, max=1 (formula: pixel / 255.0)
    void setNormalization(float min, float max);

    // Preprocess raw RGB frame for model input
    // Input: raw RGB data (uint8_t, 3 channels)
    // Output: normalized float data ready for TFLite
    void process(
        const uint8_t* input_rgb,
        int input_width,
        int input_height,
        int input_stride,
        std::vector<float>& output
    );

    // Get configured target size
    int getTargetWidth() const { return target_width_; }
    int getTargetHeight() const { return target_height_; }

private:
    int target_width_ = 320;
    int target_height_ = 320;

    // Normalization range (default: SSD MobileNet [-1, 1])
    float norm_min_ = -1.0f;
    float norm_max_ = 1.0f;

    // Resize with bilinear interpolation
    void resize(
        const uint8_t* input,
        int in_w, int in_h, int in_stride,
        uint8_t* output,
        int out_w, int out_h
    );

    // Normalize pixel values
    // Typical: (pixel / 255.0) or (pixel - 127.5) / 127.5
    void normalize(
        const uint8_t* input,
        int width, int height,
        std::vector<float>& output
    );

    std::vector<uint8_t> resize_buffer_;
};

}  // namespace vision_detector

#endif  // VISION_DETECTOR_PREPROCESSOR_H