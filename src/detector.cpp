#include "detector.h"

#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <filesystem>
#include <sys/utsname.h>

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

    // Set model name (from config or extract from path)
    if (!config.model_name.empty()) {
        model_name_ = config.model_name;
    } else {
        auto pos = config.model_path.rfind('/');
        if (pos != std::string::npos) {
            model_name_ = config.model_path.substr(pos + 1);
        } else {
            model_name_ = config.model_path;
        }
    }

    // Set model description
    model_description_ = config.model_description;

    // Get model file size
    try {
        model_size_bytes_ = std::filesystem::file_size(config.model_path);
    } catch (...) {
        model_size_bytes_ = 0;
    }

    std::cout << "TFLite model loaded: " << model_name_ << std::endl;
    std::cout << "  Input size: " << input_width_ << "x" << input_height_ << std::endl;
    std::cout << "  Output type: " << config_.output_type << std::endl;

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

    // Debug: print output tensor info once
    static bool debug_printed = false;
    if (!debug_printed) {
        int num_outputs = impl_->interpreter->outputs().size();
        std::cout << "Output tensors (" << num_outputs << "):" << std::endl;
        for (int i = 0; i < num_outputs; ++i) {
            auto* tensor = impl_->interpreter->output_tensor(i);
            std::cout << "  [" << i << "] " << impl_->interpreter->GetOutputName(i)
                      << " shape: ";
            for (int d = 0; d < tensor->dims->size; ++d) {
                std::cout << tensor->dims->data[d] << " ";
            }
            std::cout << std::endl;
        }
        debug_printed = true;
    }

    // Choose parser based on output type
    if (config_.output_type == "yolov8") {
        parseYoloV8Output(detections);
    } else {
        // Default: SSD MobileNet
        parseSSDOutput(detections);
    }
#endif

    auto end = std::chrono::high_resolution_clock::now();
    last_inference_time_ms_ = std::chrono::duration<float, std::milli>(end - start).count();

    // Apply NMS
    applyNMS(detections);

    return detections;
}

#ifdef USE_TFLITE
void Detector::parseSSDOutput(std::vector<detector_protocol::Detection>& detections) {
    int num_outputs = impl_->interpreter->outputs().size();

    if (num_outputs >= 4) {
        // Detect TF1 vs TF2 model by checking output tensor name
        std::string out_name = impl_->interpreter->GetOutputName(0);
        bool is_tf2 = (out_name.find("StatefulPartitionedCall") != std::string::npos);

        int boxes_idx, classes_idx, scores_idx;
        if (is_tf2) {
            boxes_idx = 1;
            classes_idx = 3;
            scores_idx = 0;
        } else {
            boxes_idx = 0;
            classes_idx = 1;
            scores_idx = 2;
        }

        float* boxes = impl_->interpreter->typed_output_tensor<float>(boxes_idx);
        float* classes = impl_->interpreter->typed_output_tensor<float>(classes_idx);
        float* scores = impl_->interpreter->typed_output_tensor<float>(scores_idx);

        if (boxes && classes && scores) {
            auto* scores_tensor = impl_->interpreter->output_tensor(scores_idx);
            int num_scores = scores_tensor->dims->data[1];
            num_scores = std::min(num_scores, 100);

            for (int i = 0; i < num_scores; ++i) {
                if (scores[i] < config_.confidence_threshold || scores[i] > 1.0f) continue;

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
}

void Detector::parseYoloV8Output(std::vector<detector_protocol::Detection>& detections) {
    // YOLOv8 output format: [1, num_classes+4, num_boxes]
    // For single class: [1, 5, 2100] where 5 = 4 (box coords) + 1 (class score)
    // Box format: [cx, cy, w, h] in pixels (need to normalize by input size)

    auto* output_tensor = impl_->interpreter->output_tensor(0);
    float* output = impl_->interpreter->typed_output_tensor<float>(0);

    if (!output || output_tensor->dims->size < 3) {
        std::cerr << "Invalid YOLOv8 output tensor" << std::endl;
        return;
    }

    // Output shape: [batch, channels, num_boxes]
    // channels = 4 (box) + num_classes
    int num_channels = output_tensor->dims->data[1];
    int num_boxes = output_tensor->dims->data[2];
    int num_classes_in_output = num_channels - 4;

    // YOLOv8 TFLite output is transposed: [1, 5, 2100]
    // Data is already normalized to [0, 1] by ultralytics export
    // Data layout: all cx values first, then all cy, w, h, scores

    for (int i = 0; i < num_boxes; ++i) {
        // Find best class score for this box
        float max_score = 0.0f;
        int max_class = 0;

        for (int c = 0; c < num_classes_in_output; ++c) {
            float score = output[(4 + c) * num_boxes + i];
            if (score > max_score) {
                max_score = score;
                max_class = c;
            }
        }

        // Skip low confidence
        if (max_score < config_.confidence_threshold) continue;

        // Get box coordinates (already normalized to [0, 1])
        float cx = output[0 * num_boxes + i];  // center x
        float cy = output[1 * num_boxes + i];  // center y
        float bw = output[2 * num_boxes + i];  // width
        float bh = output[3 * num_boxes + i];  // height

        // Convert from center format to corner format
        float x = cx - bw / 2.0f;
        float y = cy - bh / 2.0f;

        // Clamp to valid range
        x = std::max(0.0f, std::min(1.0f, x));
        y = std::max(0.0f, std::min(1.0f, y));
        bw = std::max(0.0f, std::min(1.0f - x, bw));
        bh = std::max(0.0f, std::min(1.0f - y, bh));

        // Skip invalid boxes
        if (bw <= 0.001f || bh <= 0.001f) continue;

        detector_protocol::Detection det;
        det.x = x;
        det.y = y;
        det.width = bw;
        det.height = bh;
        det.confidence = max_score;
        det.class_id = static_cast<uint32_t>(max_class);

        std::string label = getClassLabel(det.class_id);
        std::strncpy(det.label, label.c_str(), sizeof(det.label) - 1);
        det.label[sizeof(det.label) - 1] = '\0';

        detections.push_back(det);
    }
}
#endif

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

detector_protocol::ModelInfo Detector::getModelInfo() const {
    using namespace detector_protocol;

    ModelInfo info{};

    // Name
    std::strncpy(info.name, model_name_.c_str(), sizeof(info.name) - 1);
    info.name[sizeof(info.name) - 1] = '\0';

    // Description
    std::strncpy(info.description, model_description_.c_str(), sizeof(info.description) - 1);
    info.description[sizeof(info.description) - 1] = '\0';

    // Model type
    if (config_.output_type == "yolov8") {
        info.type = ModelType::YOLOV8;
    } else if (config_.output_type == "yolov5") {
        info.type = ModelType::YOLOV5;
    } else if (config_.output_type == "efficientdet") {
        info.type = ModelType::EFFICIENTDET;
    } else {
        info.type = ModelType::SSD_MOBILENET;
    }

    // Dimensions
    info.input_width = static_cast<uint32_t>(input_width_);
    info.input_height = static_cast<uint32_t>(input_height_);
    info.num_classes = static_cast<uint32_t>(num_classes_);
    info.model_size_bytes = model_size_bytes_;

    // Device info
    struct utsname sys_info;
    if (uname(&sys_info) == 0) {
        std::string device = std::string(sys_info.sysname) + "-" + sys_info.machine;
        std::strncpy(info.device, device.c_str(), sizeof(info.device) - 1);
        info.device[sizeof(info.device) - 1] = '\0';
    } else {
        std::strncpy(info.device, "unknown", sizeof(info.device) - 1);
    }

    // Zero reserved
    std::memset(info.reserved, 0, sizeof(info.reserved));

    return info;
}

}  // namespace vision_detector