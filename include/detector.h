#ifndef VISION_DETECTOR_DETECTOR_H
#define VISION_DETECTOR_DETECTOR_H

#include <string>
#include <vector>
#include <memory>
#include <detector_protocol/protocol.h>

namespace vision_detector {

struct DetectorConfig {
    std::string model_path;
    float confidence_threshold = 0.5f;
    float nms_threshold = 0.4f;
    int num_threads = 2;
    bool use_gpu = false;
};

class Detector {
public:
    Detector();
    ~Detector();

    // Non-copyable
    Detector(const Detector&) = delete;
    Detector& operator=(const Detector&) = delete;

    // Initialize with model file
    bool initialize(const DetectorConfig& config);

    // Run inference on preprocessed input
    // Input: float array of size [1, height, width, 3]
    // Returns: vector of detections
    std::vector<detector_protocol::Detection> detect(
        const float* input_data,
        int input_width,
        int input_height
    );

    // Get model info
    int getInputWidth() const { return input_width_; }
    int getInputHeight() const { return input_height_; }
    int getNumClasses() const { return num_classes_; }
    const std::string& getModelName() const { return model_name_; }
    float getLastInferenceTimeMs() const { return last_inference_time_ms_; }

    // Get class label
    std::string getClassLabel(int class_id) const;

    // Cleanup
    void cleanup();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    int input_width_ = 0;
    int input_height_ = 0;
    int num_classes_ = 0;
    std::string model_name_;
    float last_inference_time_ms_ = 0.0f;

    DetectorConfig config_;
    std::vector<std::string> class_labels_;

    bool loadLabels(const std::string& labels_path);
    void applyNMS(std::vector<detector_protocol::Detection>& detections);
};

}  // namespace vision_detector

#endif  // VISION_DETECTOR_DETECTOR_H