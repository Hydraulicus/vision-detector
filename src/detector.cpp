#include "detector.h"

#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <cmath>

// TFLite headers - uncomment when TFLite is available
// #include "tensorflow/lite/interpreter.h"
// #include "tensorflow/lite/kernels/register.h"
// #include "tensorflow/lite/model.h"

namespace vision_detector {

struct Detector::Impl {
    // TFLite objects - uncomment when TFLite is available
    // std::unique_ptr<tflite::FlatBufferModel> model;
    // std::unique_ptr<tflite::Interpreter> interpreter;
    // tflite::ops::builtin::BuiltinOpResolver resolver;
};

Detector::Detector() : impl_(std::make_unique<Impl>()) {}

Detector::~Detector() {
    cleanup();
}

bool Detector::initialize(const DetectorConfig& config) {
    config_ = config;

    // TODO: Load TFLite model
    // impl_->model = tflite::FlatBufferModel::BuildFromFile(config.model_path.c_str());
    // if (!impl_->model) {
    //     return false;
    // }
    //
    // tflite::InterpreterBuilder builder(*impl_->model, impl_->resolver);
    // builder(&impl_->interpreter);
    // if (!impl_->interpreter) {
    //     return false;
    // }
    //
    // impl_->interpreter->SetNumThreads(config.num_threads);
    //
    // if (config.use_gpu) {
    //     // Setup GPU delegate
    // }
    //
    // if (impl_->interpreter->AllocateTensors() != kTfLiteOk) {
    //     return false;
    // }
    //
    // // Get input dimensions
    // auto* input = impl_->interpreter->input_tensor(0);
    // input_height_ = input->dims->data[1];
    // input_width_ = input->dims->data[2];
    //
    // // Get output info for num_classes
    // auto* output = impl_->interpreter->output_tensor(0);
    // num_classes_ = output->dims->data[...];

    // Placeholder values until TFLite is integrated
    input_width_ = 320;
    input_height_ = 320;
    num_classes_ = 10;
    model_name_ = "placeholder_model";

    // Load labels if available
    std::string labels_path = config.model_path;
    auto pos = labels_path.rfind('.');
    if (pos != std::string::npos) {
        labels_path = labels_path.substr(0, pos) + ".txt";
        loadLabels(labels_path);
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

    // TODO: Run TFLite inference
    // float* input = impl_->interpreter->typed_input_tensor<float>(0);
    // std::memcpy(input, input_data, input_width * input_height * 3 * sizeof(float));
    //
    // if (impl_->interpreter->Invoke() != kTfLiteOk) {
    //     return detections;
    // }
    //
    // // Parse output tensors (format depends on model)
    // float* boxes = impl_->interpreter->typed_output_tensor<float>(0);
    // float* classes = impl_->interpreter->typed_output_tensor<float>(1);
    // float* scores = impl_->interpreter->typed_output_tensor<float>(2);
    // float* num_det = impl_->interpreter->typed_output_tensor<float>(3);
    //
    // int count = static_cast<int>(*num_det);
    // for (int i = 0; i < count; ++i) {
    //     if (scores[i] < config_.confidence_threshold) continue;
    //
    //     detector_protocol::Detection det;
    //     det.y = boxes[i * 4 + 0];
    //     det.x = boxes[i * 4 + 1];
    //     det.height = boxes[i * 4 + 2] - det.y;
    //     det.width = boxes[i * 4 + 3] - det.x;
    //     det.confidence = scores[i];
    //     det.class_id = static_cast<uint32_t>(classes[i]);
    //
    //     std::string label = getClassLabel(det.class_id);
    //     std::strncpy(det.label, label.c_str(), sizeof(det.label) - 1);
    //
    //     detections.push_back(det);
    // }

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
        return false;
    }

    class_labels_.clear();
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            class_labels_.push_back(line);
        }
    }

    num_classes_ = static_cast<int>(class_labels_.size());
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
    // impl_->interpreter.reset();
    // impl_->model.reset();
    class_labels_.clear();
}

}  // namespace vision_detector