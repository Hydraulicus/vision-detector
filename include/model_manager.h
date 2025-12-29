#ifndef VISION_DETECTOR_MODEL_MANAGER_H
#define VISION_DETECTOR_MODEL_MANAGER_H

#include <string>
#include <vector>
#include <map>
#include <memory>

namespace vision_detector {

// Model configuration loaded from config.json
struct ModelConfig {
    std::string name;
    std::string description;
    std::string model_file;
    std::string labels_file;

    // Input configuration
    int input_width = 320;
    int input_height = 320;
    std::string input_format = "RGB";
    float normalize_min = -1.0f;
    float normalize_max = 1.0f;

    // Output configuration
    std::string output_type = "ssd_mobilenet";
    std::string tf_version = "auto";

    // Computed paths (set after loading)
    std::string model_path;      // Full path to model file
    std::string labels_path;     // Full path to labels file
    std::string base_path;       // Base directory of model
    std::string eval_input_path; // Path to eval/input
    std::string eval_output_path;// Path to eval/output
};

// Model registry entry from models.json
struct ModelEntry {
    std::string id;          // Model identifier (e.g., "tanks")
    std::string path;        // Relative path to model folder
    std::string description;
};

class ModelManager {
public:
    ModelManager();
    ~ModelManager();

    // Initialize with models directory path
    bool initialize(const std::string& models_path);

    // List available models
    std::vector<ModelEntry> listModels() const;

    // Get model configuration by name
    bool getModelConfig(const std::string& model_name, ModelConfig& config) const;

    // Get default model name
    std::string getDefaultModel() const { return default_model_; }

    // Check if model exists
    bool hasModel(const std::string& model_name) const;

    // Get models directory path
    const std::string& getModelsPath() const { return models_path_; }

private:
    std::string models_path_;
    std::string default_model_;
    std::map<std::string, ModelEntry> models_;

    bool loadRegistry(const std::string& registry_path);
    bool loadModelConfig(const std::string& model_path, ModelConfig& config) const;
};

}  // namespace vision_detector

#endif  // VISION_DETECTOR_MODEL_MANAGER_H