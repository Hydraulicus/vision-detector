#ifndef VISION_DETECTOR_EVALUATOR_H
#define VISION_DETECTOR_EVALUATOR_H

#include <string>
#include <vector>

#include "detector.h"
#include "preprocessor.h"
#include "model_manager.h"

namespace vision_detector {

struct EvaluationConfig {
    std::string model_name;
    std::string models_path;
    std::string input_path;    // Image file or directory
    std::string output_path;   // Output directory
    float confidence_threshold = 0.5f;
    float nms_threshold = 0.4f;
    bool use_gpu = false;
};

struct EvaluationResult {
    std::string source_file;
    std::string model_name;
    int image_width = 0;
    int image_height = 0;
    float inference_time_ms = 0.0f;
    std::vector<detector_protocol::Detection> detections;
};

class Evaluator {
public:
    Evaluator();
    ~Evaluator();

    // Initialize with configuration
    bool initialize(const EvaluationConfig& config);

    // Run evaluation
    bool run();

    // Get results
    const std::vector<EvaluationResult>& getResults() const { return results_; }

private:
    EvaluationConfig config_;
    ModelManager model_manager_;
    ModelConfig model_config_;
    Detector detector_;
    Preprocessor preprocessor_;
    std::vector<EvaluationResult> results_;

    // Process single image
    bool processImage(const std::string& image_path);

    // Load image from file
    bool loadImage(const std::string& path,
                   std::vector<uint8_t>& data,
                   int& width, int& height);

    // Save annotated image
    bool saveAnnotatedImage(const std::string& path,
                            const uint8_t* image_data,
                            int width, int height,
                            const std::vector<detector_protocol::Detection>& detections);

    // Save JSON results
    bool saveJsonResults(const std::string& path, const EvaluationResult& result);

    // Draw bounding box on image
    void drawBox(uint8_t* image, int img_width, int img_height,
                 int x, int y, int w, int h,
                 uint8_t r, uint8_t g, uint8_t b);

    // Draw text label (simple pixel font)
    void drawLabel(uint8_t* image, int img_width, int img_height,
                   int x, int y, const std::string& text,
                   uint8_t r, uint8_t g, uint8_t b);

    // Get list of image files in directory
    std::vector<std::string> getImageFiles(const std::string& dir_path);

    // Check if path is a directory
    bool isDirectory(const std::string& path);

    // Get file extension
    std::string getExtension(const std::string& path);

    // Get filename without path and extension
    std::string getBasename(const std::string& path);
};

}  // namespace vision_detector

#endif  // VISION_DETECTOR_EVALUATOR_H