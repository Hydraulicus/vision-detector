#include "model_manager.h"
#include "json_utils.h"

#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

namespace vision_detector {

ModelManager::ModelManager() = default;
ModelManager::~ModelManager() = default;

bool ModelManager::initialize(const std::string& models_path) {
    models_path_ = models_path;

    // Normalize path
    if (!models_path_.empty() && models_path_.back() == '/') {
        models_path_.pop_back();
    }

    // Check if models directory exists
    if (!fs::exists(models_path_)) {
        std::cerr << "Models directory not found: " << models_path_ << std::endl;
        return false;
    }

    // Load registry
    std::string registry_path = models_path_ + "/models.json";
    if (!loadRegistry(registry_path)) {
        std::cerr << "Failed to load models registry: " << registry_path << std::endl;
        return false;
    }

    std::cout << "Loaded " << models_.size() << " model(s) from registry" << std::endl;
    return true;
}

bool ModelManager::loadRegistry(const std::string& registry_path) {
    try {
        JsonValue root = JsonParser::parseFile(registry_path);

        // Get default model
        if (root.has("default")) {
            default_model_ = root["default"].asString();
        }

        // Load models
        if (root.has("models")) {
            const auto& models = root["models"];
            for (const auto& [id, entry] : models.items()) {
                ModelEntry model_entry;
                model_entry.id = id;
                model_entry.path = entry["path"].asString();
                if (entry.has("description")) {
                    model_entry.description = entry["description"].asString();
                }
                models_[id] = model_entry;
            }
        }

        return !models_.empty();
    } catch (const std::exception& e) {
        std::cerr << "Error parsing registry: " << e.what() << std::endl;
        return false;
    }
}

std::vector<ModelEntry> ModelManager::listModels() const {
    std::vector<ModelEntry> result;
    for (const auto& [id, entry] : models_) {
        result.push_back(entry);
    }
    return result;
}

bool ModelManager::hasModel(const std::string& model_name) const {
    return models_.find(model_name) != models_.end();
}

bool ModelManager::getModelConfig(const std::string& model_name, ModelConfig& config) const {
    auto it = models_.find(model_name);
    if (it == models_.end()) {
        std::cerr << "Model not found: " << model_name << std::endl;
        return false;
    }

    std::string model_dir = models_path_ + "/" + it->second.path;
    return loadModelConfig(model_dir, config);
}

bool ModelManager::loadModelConfig(const std::string& model_dir, ModelConfig& config) const {
    std::string config_path = model_dir + "/config.json";

    try {
        JsonValue root = JsonParser::parseFile(config_path);

        config.base_path = model_dir;
        config.name = root["name"].asString();

        if (root.has("description")) {
            config.description = root["description"].asString();
        }

        config.model_file = root["model_file"].asString();
        config.labels_file = root["labels_file"].asString();

        // Build full paths
        config.model_path = model_dir + "/" + config.model_file;
        config.labels_path = model_dir + "/" + config.labels_file;
        config.eval_input_path = model_dir + "/eval/input";
        config.eval_output_path = model_dir + "/eval/output";

        // Input configuration
        if (root.has("input")) {
            const auto& input = root["input"];
            if (input.has("width")) config.input_width = input["width"].asInt();
            if (input.has("height")) config.input_height = input["height"].asInt();
            if (input.has("format")) config.input_format = input["format"].asString();
            if (input.has("normalize")) {
                const auto& norm = input["normalize"];
                if (norm.size() >= 2) {
                    config.normalize_min = norm[0].asFloat();
                    config.normalize_max = norm[1].asFloat();
                }
            }
        }

        // Output configuration
        if (root.has("output")) {
            const auto& output = root["output"];
            if (output.has("type")) config.output_type = output["type"].asString();
            if (output.has("tf_version")) config.tf_version = output["tf_version"].asString();
        }

        // Verify model file exists
        if (!fs::exists(config.model_path)) {
            std::cerr << "Model file not found: " << config.model_path << std::endl;
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading model config: " << e.what() << std::endl;
        return false;
    }
}

}  // namespace vision_detector