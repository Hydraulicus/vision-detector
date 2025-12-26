#include "detector.h"

#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <iostream>

#ifdef USE_TFLITE
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#endif

namespace vision_detector {

struct Detector::Impl {
#ifdef USE_TFLITE
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;
#endif
};

Detector::Detector() : impl_(std::make_unique<Impl>()) {}

Detector::~Detector() {
    cleanup();
}

bool Detector::initialize(const DetectorConfig& config) {
    config_ = config;

#ifdef USE_TFLITE
    // Load TFLite model
    impl_->model = tflite::FlatBufferModel::BuildFromFile(config.model_path.c_str());
    if (!impl_->model) {
        std::cerr << "Failed to load model: " << config.model_path << std::endl;
        return false;
    }

    tflite::InterpreterBuilder builder(*impl_->model, impl_->resolver);
    builder(&impl_->interpreter);
    if (!impl_->interpreter) {
        std::cerr << "Failed to create interpreter" << std::endl;
        return false;
    }

    impl_->interpreter->SetNumThreads(config.num_threads);

    if (impl_->interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors" << std::endl;
        return false;
    }

    // Get input dimensions
    auto* input = impl_->interpreter->input_tensor(0);
    if (input->dims->size >= 3) {
        input_height_ = input->dims->data[1];
        input_width_ = input->dims->data[2];
    } else {
        input_width_ = 320;
        input_height_ = 320;
    }

    // Extract model name from path
    auto pos = config.model_path.rfind('/');
    if (pos != std::string::npos) {
        model_name_ = config.model_path.substr(pos + 1);
    } else {
        model_name_ = config.model_path;
    }

    std::cout << "TFLite model loaded: " << model_name_ << std::endl;
    std::cout << "  Input size: " << input_width_ << "x" << input_height_ << std::endl;

#else
    // Placeholder values when TFLite is disabled
    input_width_ = 320;
    input_height_ = 320;
    num_classes_ = 10;
    model_name_ = "placeholder_model";
#endif

    // Load labels if provided via config or adjacent to model
    if (!config.labels_path.empty()) {
        loadLabels(config.labels_path);
    } else {
        std::string labels_path = config.model_path;
        auto pos = labels_path.rfind('.');
        if (pos != std::string::npos) {
            labels_path = labels_path.substr(0, pos) + ".txt";
            loadLabels(labels_path);
        }
    }

    if (class_labels_.empty()) {
        num_classes_ = 10;  // Default
    }

    return true;
}

std::vector<detector_protocol::Detection> Detector::detect(
    const float* input_data,
    int input_width,
    int input_height
) {
    std::vector<detector_protocol::Detection> detections;

    auto start = std::chrono::high_resolution_clock::now();

#ifdef USE_TFLITE
    // Copy input data to interpreter
    float* input = impl_->interpreter->typed_input_tensor<float>(0);
    if (!input) {
        std::cerr << "Failed to get input tensor" << std::endl;
        return detections;
    }

    size_t input_size = input_width * input_height * 3;
    std::memcpy(input, input_data, input_size * sizeof(float));

    // Run inference
    if (impl_->interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "Inference failed" << std::endl;
        return detections;
    }

    // Parse output tensors
    // SSD MobileNet output format: boxes, classes, scores, num_detections
    // Output tensor indices may vary by model
    int num_outputs = impl_->interpreter->outputs().size();

    if (num_outputs >= 4) {
        // Standard SSD output format
        float* boxes = impl_->interpreter->typed_output_tensor<float>(0);
        float* classes = impl_->interpreter->typed_output_tensor<float>(1);
        float* scores = impl_->interpreter->typed_output_tensor<float>(2);
        float* num_det = impl_->interpreter->typed_output_tensor<float>(3);

        if (boxes && classes && scores && num_det) {
            int count = static_cast<int>(*num_det);
            count = std::min(count, 100);  // Limit to reasonable number

            for (int i = 0; i < count; ++i) {
                if (scores[i] < config_.confidence_threshold) continue;

                detector_protocol::Detection det;
                // SSD format: [ymin, xmin, ymax, xmax] normalized
                det.y = boxes[i * 4 + 0];
                det.x = boxes[i * 4 + 1];
                det.height = boxes[i * 4 + 2] - det.y;
                det.width = boxes[i * 4 + 3] - det.x;
                det.confidence = scores[i];
                det.class_id = static_cast<uint32_t>(classes[i]);

                std::string label = getClassLabel(det.class_id);
                std::strncpy(det.label, label.c_str(), sizeof(det.label) - 1);
                det.label[sizeof(det.label) - 1] = '\0';

                detections.push_back(det);
            }
        }
    }
#endif

    auto end = std::chrono::high_resolution_clock::now();
    last_inference_time_ms_ = std::chrono::duration<float, std::milli>(end - start).count();

    // Apply NMS
    applyNMS(detections);

    return detections;
}

std::string Detector::getClassLabel(int class_id) const {
    if (class_id >= 0 && class_id < static_cast<int>(class_labels_.size())) {
        return class_labels_[class_id];
    }
    return "class_" + std::to_string(class_id);
}

bool Detector::loadLabels(const std::string& labels_path) {
    std::ifstream file(labels_path);
    if (!file.is_open()) {
        std::cout << "No labels file found: " << labels_path << std::endl;
        return false;
    }

    class_labels_.clear();
    std::string line;
    while (std::getline(file, line)) {
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        if (!line.empty()) {
            class_labels_.push_back(line);
        }
    }

    num_classes_ = static_cast<int>(class_labels_.size());
    std::cout << "Loaded " << num_classes_ << " class labels" << std::endl;
    return true;
}

void Detector::applyNMS(std::vector<detector_protocol::Detection>& detections) {
    if (detections.empty()) return;

    // Sort by confidence (descending)
    std::sort(detections.begin(), detections.end(),
        [](const auto& a, const auto& b) {
            return a.confidence > b.confidence;
        });

    std::vector<bool> suppressed(detections.size(), false);

    auto iou = [](const detector_protocol::Detection& a,
                  const detector_protocol::Detection& b) -> float {
        float x1 = std::max(a.x, b.x);
        float y1 = std::max(a.y, b.y);
        float x2 = std::min(a.x + a.width, b.x + b.width);
        float y2 = std::min(a.y + a.height, b.y + b.height);

        float inter_w = std::max(0.0f, x2 - x1);
        float inter_h = std::max(0.0f, y2 - y1);
        float inter_area = inter_w * inter_h;

        float area_a = a.width * a.height;
        float area_b = b.width * b.height;
        float union_area = area_a + area_b - inter_area;

        return union_area > 0 ? inter_area / union_area : 0.0f;
    };

    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) continue;

        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j]) continue;

            if (detections[i].class_id == detections[j].class_id) {
                if (iou(detections[i], detections[j]) > config_.nms_threshold) {
                    suppressed[j] = true;
                }
            }
        }
    }

    // Remove suppressed detections
    auto it = detections.begin();
    size_t idx = 0;
    while (it != detections.end()) {
        if (suppressed[idx++]) {
            it = detections.erase(it);
        } else {
            ++it;
        }
    }
}

void Detector::cleanup() {
#ifdef USE_TFLITE
    impl_->interpreter.reset();
    impl_->model.reset();
#endif
    class_labels_.clear();
}

}  // namespace vision_detector