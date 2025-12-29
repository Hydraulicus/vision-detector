#include "evaluator.h"
#include "json_utils.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <filesystem>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace fs = std::filesystem;

namespace vision_detector {

// Color palette for different classes
static const uint8_t CLASS_COLORS[][3] = {
    {255, 0, 0},     // Red
    {0, 255, 0},     // Green
    {0, 0, 255},     // Blue
    {255, 255, 0},   // Yellow
    {255, 0, 255},   // Magenta
    {0, 255, 255},   // Cyan
    {255, 128, 0},   // Orange
    {128, 0, 255},   // Purple
    {0, 255, 128},   // Spring Green
    {255, 0, 128},   // Rose
};
static const int NUM_COLORS = sizeof(CLASS_COLORS) / sizeof(CLASS_COLORS[0]);

Evaluator::Evaluator() = default;
Evaluator::~Evaluator() = default;

bool Evaluator::initialize(const EvaluationConfig& config) {
    config_ = config;

    // Initialize model manager
    if (!model_manager_.initialize(config.models_path)) {
        std::cerr << "Failed to initialize model manager" << std::endl;
        return false;
    }

    // Get model configuration
    if (!model_manager_.getModelConfig(config.model_name, model_config_)) {
        std::cerr << "Failed to get model config for: " << config.model_name << std::endl;
        return false;
    }

    // Create output directory if needed
    if (!fs::exists(config.output_path)) {
        fs::create_directories(config.output_path);
    }

    // Initialize detector
    DetectorConfig det_config;
    det_config.model_path = model_config_.model_path;
    det_config.labels_path = model_config_.labels_path;
    det_config.confidence_threshold = config.confidence_threshold;
    det_config.nms_threshold = config.nms_threshold;
    det_config.use_gpu = config.use_gpu;

    if (!detector_.initialize(det_config)) {
        std::cerr << "Failed to initialize detector" << std::endl;
        return false;
    }

    // Initialize preprocessor
    preprocessor_.setTargetSize(detector_.getInputWidth(), detector_.getInputHeight());

    std::cout << "Evaluator initialized:" << std::endl;
    std::cout << "  Model: " << model_config_.name << std::endl;
    std::cout << "  Input: " << config.input_path << std::endl;
    std::cout << "  Output: " << config.output_path << std::endl;

    return true;
}

bool Evaluator::run() {
    results_.clear();

    std::vector<std::string> image_files;

    if (isDirectory(config_.input_path)) {
        image_files = getImageFiles(config_.input_path);
        std::cout << "Found " << image_files.size() << " image(s) to process" << std::endl;
    } else {
        image_files.push_back(config_.input_path);
    }

    if (image_files.empty()) {
        std::cerr << "No images found to process" << std::endl;
        return false;
    }

    int processed = 0;
    for (const auto& image_path : image_files) {
        std::cout << "Processing: " << image_path << std::endl;
        if (processImage(image_path)) {
            ++processed;
        }
    }

    std::cout << "\nProcessed " << processed << "/" << image_files.size() << " images" << std::endl;

    // Print summary
    int total_detections = 0;
    float total_time = 0.0f;
    for (const auto& result : results_) {
        total_detections += result.detections.size();
        total_time += result.inference_time_ms;
    }

    if (!results_.empty()) {
        std::cout << "Total detections: " << total_detections << std::endl;
        std::cout << "Average inference time: " << (total_time / results_.size()) << " ms" << std::endl;
    }

    return processed > 0;
}

bool Evaluator::processImage(const std::string& image_path) {
    // Load image
    std::vector<uint8_t> image_data;
    int width, height;

    if (!loadImage(image_path, image_data, width, height)) {
        std::cerr << "  Failed to load image" << std::endl;
        return false;
    }

    // Preprocess
    std::vector<float> preprocessed;
    preprocessor_.process(image_data.data(), width, height, width * 3, preprocessed);

    // Run detection
    auto detections = detector_.detect(
        preprocessed.data(),
        detector_.getInputWidth(),
        detector_.getInputHeight()
    );

    // Create result
    EvaluationResult result;
    result.source_file = image_path;
    result.model_name = model_config_.name;
    result.image_width = width;
    result.image_height = height;
    result.inference_time_ms = detector_.getLastInferenceTimeMs();
    result.detections = detections;

    std::cout << "  Found " << detections.size() << " detection(s) in "
              << result.inference_time_ms << " ms" << std::endl;

    // Generate output paths
    std::string basename = getBasename(image_path);
    std::string output_image = config_.output_path + "/" + basename + "_detected.jpg";
    std::string output_json = config_.output_path + "/" + basename + "_results.json";

    // Save annotated image
    if (!saveAnnotatedImage(output_image, image_data.data(), width, height, detections)) {
        std::cerr << "  Failed to save annotated image" << std::endl;
    }

    // Save JSON results
    if (!saveJsonResults(output_json, result)) {
        std::cerr << "  Failed to save JSON results" << std::endl;
    }

    results_.push_back(result);
    return true;
}

bool Evaluator::loadImage(const std::string& path,
                          std::vector<uint8_t>& data,
                          int& width, int& height) {
    int channels;
    uint8_t* img = stbi_load(path.c_str(), &width, &height, &channels, 3);
    if (!img) {
        return false;
    }

    data.assign(img, img + width * height * 3);
    stbi_image_free(img);
    return true;
}

bool Evaluator::saveAnnotatedImage(const std::string& path,
                                   const uint8_t* image_data,
                                   int width, int height,
                                   const std::vector<detector_protocol::Detection>& detections) {
    // Copy image data to modify
    std::vector<uint8_t> annotated(image_data, image_data + width * height * 3);

    for (const auto& det : detections) {
        // Convert normalized coordinates to pixel coordinates
        int x = static_cast<int>(det.x * width);
        int y = static_cast<int>(det.y * height);
        int w = static_cast<int>(det.width * width);
        int h = static_cast<int>(det.height * height);

        // Clamp to image bounds
        x = std::max(0, std::min(x, width - 1));
        y = std::max(0, std::min(y, height - 1));
        w = std::min(w, width - x);
        h = std::min(h, height - y);

        // Get color for class
        int color_idx = det.class_id % NUM_COLORS;
        uint8_t r = CLASS_COLORS[color_idx][0];
        uint8_t g = CLASS_COLORS[color_idx][1];
        uint8_t b = CLASS_COLORS[color_idx][2];

        // Draw bounding box
        drawBox(annotated.data(), width, height, x, y, w, h, r, g, b);

        // Draw label
        std::stringstream label;
        label << det.label << " " << std::fixed << std::setprecision(0) << (det.confidence * 100) << "%";
        drawLabel(annotated.data(), width, height, x, y - 12, label.str(), r, g, b);
    }

    // Save as JPEG
    int quality = 95;
    return stbi_write_jpg(path.c_str(), width, height, 3, annotated.data(), quality) != 0;
}

bool Evaluator::saveJsonResults(const std::string& path, const EvaluationResult& result) {
    std::ofstream file(path);
    if (!file.is_open()) {
        return false;
    }

    // Get current timestamp
    auto now = std::time(nullptr);
    auto tm = *std::localtime(&now);
    std::ostringstream timestamp;
    timestamp << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");

    file << "{\n";
    file << "  \"source\": \"" << fs::path(result.source_file).filename().string() << "\",\n";
    file << "  \"model\": \"" << result.model_name << "\",\n";
    file << "  \"timestamp\": \"" << timestamp.str() << "\",\n";
    file << "  \"image_size\": {\"width\": " << result.image_width
         << ", \"height\": " << result.image_height << "},\n";
    file << "  \"inference_time_ms\": " << std::fixed << std::setprecision(2)
         << result.inference_time_ms << ",\n";
    file << "  \"detections\": [\n";

    for (size_t i = 0; i < result.detections.size(); ++i) {
        const auto& det = result.detections[i];

        // Convert to pixel coordinates for JSON output
        int x = static_cast<int>(det.x * result.image_width);
        int y = static_cast<int>(det.y * result.image_height);
        int w = static_cast<int>(det.width * result.image_width);
        int h = static_cast<int>(det.height * result.image_height);

        file << "    {\n";
        file << "      \"class_id\": " << det.class_id << ",\n";
        file << "      \"label\": \"" << det.label << "\",\n";
        file << "      \"confidence\": " << std::fixed << std::setprecision(4)
             << det.confidence << ",\n";
        file << "      \"bbox\": {\"x\": " << x << ", \"y\": " << y
             << ", \"width\": " << w << ", \"height\": " << h << "}\n";
        file << "    }";
        if (i < result.detections.size() - 1) file << ",";
        file << "\n";
    }

    file << "  ]\n";
    file << "}\n";

    return true;
}

void Evaluator::drawBox(uint8_t* image, int img_width, int img_height,
                        int x, int y, int w, int h,
                        uint8_t r, uint8_t g, uint8_t b) {
    int thickness = 2;

    // Draw horizontal lines (top and bottom)
    for (int t = 0; t < thickness; ++t) {
        int yt = y + t;
        int yb = y + h - 1 - t;
        for (int px = x; px < x + w && px < img_width; ++px) {
            if (yt >= 0 && yt < img_height && px >= 0) {
                int idx = (yt * img_width + px) * 3;
                image[idx] = r; image[idx + 1] = g; image[idx + 2] = b;
            }
            if (yb >= 0 && yb < img_height && px >= 0) {
                int idx = (yb * img_width + px) * 3;
                image[idx] = r; image[idx + 1] = g; image[idx + 2] = b;
            }
        }
    }

    // Draw vertical lines (left and right)
    for (int t = 0; t < thickness; ++t) {
        int xl = x + t;
        int xr = x + w - 1 - t;
        for (int py = y; py < y + h && py < img_height; ++py) {
            if (xl >= 0 && xl < img_width && py >= 0) {
                int idx = (py * img_width + xl) * 3;
                image[idx] = r; image[idx + 1] = g; image[idx + 2] = b;
            }
            if (xr >= 0 && xr < img_width && py >= 0) {
                int idx = (py * img_width + xr) * 3;
                image[idx] = r; image[idx + 1] = g; image[idx + 2] = b;
            }
        }
    }
}

void Evaluator::drawLabel(uint8_t* image, int img_width, int img_height,
                          int x, int y, const std::string& text,
                          uint8_t r, uint8_t g, uint8_t b) {
    // Simple 5x7 pixel font - just draw a colored background rectangle with text placeholder
    // For production, you'd want a proper bitmap font
    int char_width = 6;
    int char_height = 10;
    int padding = 2;

    int label_width = static_cast<int>(text.length()) * char_width + padding * 2;
    int label_height = char_height + padding * 2;

    // Draw background rectangle
    int bg_y = std::max(0, y - label_height);
    for (int py = bg_y; py < bg_y + label_height && py < img_height; ++py) {
        for (int px = x; px < x + label_width && px < img_width; ++px) {
            if (px >= 0 && py >= 0) {
                int idx = (py * img_width + px) * 3;
                image[idx] = r; image[idx + 1] = g; image[idx + 2] = b;
            }
        }
    }

    // Draw text in white (simple approach - just light pixels for each character position)
    int text_y = bg_y + padding + 1;
    int text_x = x + padding;
    for (size_t i = 0; i < text.length(); ++i) {
        // Draw a simple representation - white block for each char
        for (int dy = 0; dy < char_height - 2; ++dy) {
            for (int dx = 0; dx < char_width - 2; ++dx) {
                int px = text_x + static_cast<int>(i) * char_width + dx;
                int py = text_y + dy;
                if (px >= 0 && px < img_width && py >= 0 && py < img_height) {
                    // Create a simple pattern for the character
                    bool draw = false;
                    char c = text[i];
                    // Simple character outlines (very basic)
                    if (c != ' ') {
                        if (dy == 0 || dy == char_height - 3) draw = true;  // top/bottom
                        if (dx == 0 || dx == char_width - 3) draw = true;   // left/right
                    }
                    if (draw) {
                        int idx = (py * img_width + px) * 3;
                        image[idx] = 255; image[idx + 1] = 255; image[idx + 2] = 255;
                    }
                }
            }
        }
    }
}

std::vector<std::string> Evaluator::getImageFiles(const std::string& dir_path) {
    std::vector<std::string> files;

    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (entry.is_regular_file()) {
            std::string ext = getExtension(entry.path().string());
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" ||
                ext == ".bmp" || ext == ".tga") {
                files.push_back(entry.path().string());
            }
        }
    }

    std::sort(files.begin(), files.end());
    return files;
}

bool Evaluator::isDirectory(const std::string& path) {
    return fs::is_directory(path);
}

std::string Evaluator::getExtension(const std::string& path) {
    auto pos = path.rfind('.');
    if (pos != std::string::npos) {
        return path.substr(pos);
    }
    return "";
}

std::string Evaluator::getBasename(const std::string& path) {
    fs::path p(path);
    return p.stem().string();
}

}  // namespace vision_detector