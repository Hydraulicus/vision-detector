#include <iostream>
#include <string>
#include <csignal>

#include "detector.h"
#include "preprocessor.h"
#include "server.h"
#include "model_manager.h"
#include "evaluator.h"

using namespace vision_detector;

// Global server pointer for signal handling
static DetectionServer* g_server = nullptr;

void signalHandler(int signum) {
    std::cout << "\nReceived signal " << signum << ", shutting down..." << std::endl;
    if (g_server) {
        g_server->stop();
    }
}

void printUsage(const char* program) {
    std::cout << "Vision Detector Service v1.0\n"
              << "\nUsage: " << program << " [options]\n"
              << "\nServer Mode (default):\n"
              << "  -m, --model <name|path>  Model name from registry or path to .tflite file\n"
              << "  --models-path <path>     Path to models directory (default: ./private/models)\n"
              << "  -t, --threshold <val>    Confidence threshold (default: 0.5)\n"
              << "  -n, --nms <val>          NMS threshold (default: 0.4)\n"
              << "  --gpu                    Use GPU delegate\n"
              << "  --list-models            List available models and exit\n"
              << "\nEvaluation Mode:\n"
              << "  --eval                   Enable evaluation mode (no IPC server)\n"
              << "  -m, --model <name>       Model name from registry (required)\n"
              << "  --models-path <path>     Path to models directory\n"
              << "  -i, --input <path>       Input image or directory (required)\n"
              << "  -o, --output <path>      Output directory (required)\n"
              << "  -t, --threshold <val>    Confidence threshold (default: 0.5)\n"
              << "  -n, --nms <val>          NMS threshold (default: 0.4)\n"
              << "  --gpu                    Use GPU delegate\n"
              << "\nExamples:\n"
              << "  # Server mode with model from registry\n"
              << "  " << program << " -m tanks --models-path ./private/models\n"
              << "\n  # Server mode with direct model path\n"
              << "  " << program << " -m ./models/detect.tflite -l ./models/labels.txt\n"
              << "\n  # List available models\n"
              << "  " << program << " --list-models --models-path ./private/models\n"
              << "\n  # Evaluation mode - single image\n"
              << "  " << program << " --eval -m tanks -i test.jpg -o results/\n"
              << "\n  # Evaluation mode - batch directory\n"
              << "  " << program << " --eval -m coins -i images/ -o results/\n"
              << std::endl;
}

int runServerMode(const std::string& model_arg,
                  const std::string& labels_path,
                  const std::string& models_path,
                  float threshold, float nms_threshold, bool use_gpu) {

    DetectorConfig config;
    config.confidence_threshold = threshold;
    config.nms_threshold = nms_threshold;
    config.use_gpu = use_gpu;

    // Check if model_arg is a path or a model name
    bool is_path = (model_arg.find('/') != std::string::npos) ||
                   (model_arg.find(".tflite") != std::string::npos);

    // Model config (only used for registry mode)
    ModelConfig model_config;
    bool has_model_config = false;

    if (is_path) {
        // Direct path mode
        config.model_path = model_arg;
        config.labels_path = labels_path;
    } else {
        // Model registry mode
        ModelManager model_manager;
        if (!model_manager.initialize(models_path)) {
            std::cerr << "Failed to initialize model manager" << std::endl;
            return 1;
        }

        if (!model_manager.getModelConfig(model_arg, model_config)) {
            std::cerr << "Model not found: " << model_arg << std::endl;
            std::cerr << "Use --list-models to see available models" << std::endl;
            return 1;
        }

        config.model_path = model_config.model_path;
        config.labels_path = model_config.labels_path;
        config.output_type = model_config.output_type;
        config.model_name = model_config.name;
        config.model_description = model_config.description;
        has_model_config = true;

        std::cout << "Using model from registry: " << model_config.name << std::endl;
    }

    // Setup signal handlers
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    // Initialize components
    Detector detector;
    Preprocessor preprocessor;
    DetectionServer server;

    std::cout << "Loading model: " << config.model_path << std::endl;

    if (!detector.initialize(config)) {
        std::cerr << "Failed to initialize detector" << std::endl;
        return 1;
    }

    std::cout << "Model loaded successfully:" << std::endl;
    std::cout << "  Input size: " << detector.getInputWidth() << "x"
              << detector.getInputHeight() << std::endl;
    std::cout << "  Classes: " << detector.getNumClasses() << std::endl;

    preprocessor.setTargetSize(detector.getInputWidth(), detector.getInputHeight());

    // Set normalization from model config (if available)
    if (has_model_config) {
        preprocessor.setNormalization(model_config.normalize_min, model_config.normalize_max);
    }

    if (!server.initialize(&detector, &preprocessor)) {
        std::cerr << "Failed to initialize server" << std::endl;
        return 1;
    }

    g_server = &server;

    std::cout << "\nStarting detection server..." << std::endl;
    std::cout << "Listening for connections..." << std::endl;

    if (!server.start(true)) {  // Blocking
        std::cerr << "Server error" << std::endl;
        return 1;
    }

    std::cout << "\nServer stopped." << std::endl;
    std::cout << "Frames processed: " << server.getFramesProcessed() << std::endl;
    std::cout << "Average inference time: " << server.getAverageInferenceTimeMs()
              << " ms" << std::endl;

    return 0;
}

int runEvalMode(const EvaluationConfig& eval_config) {
    Evaluator evaluator;

    if (!evaluator.initialize(eval_config)) {
        std::cerr << "Failed to initialize evaluator" << std::endl;
        return 1;
    }

    if (!evaluator.run()) {
        std::cerr << "Evaluation failed" << std::endl;
        return 1;
    }

    return 0;
}

int listModels(const std::string& models_path) {
    ModelManager model_manager;

    if (!model_manager.initialize(models_path)) {
        std::cerr << "Failed to initialize model manager" << std::endl;
        std::cerr << "Models path: " << models_path << std::endl;
        return 1;
    }

    auto models = model_manager.listModels();

    std::cout << "Available models (" << models.size() << "):\n" << std::endl;

    for (const auto& model : models) {
        std::cout << "  " << model.id;
        if (model.id == model_manager.getDefaultModel()) {
            std::cout << " (default)";
        }
        std::cout << std::endl;
        if (!model.description.empty()) {
            std::cout << "    " << model.description << std::endl;
        }
        std::cout << "    Path: " << model.path << std::endl;
        std::cout << std::endl;
    }

    return 0;
}

int main(int argc, char* argv[]) {
    std::cout << "Vision Detector Service v1.0\n" << std::endl;

    // Parse arguments
    std::string model_arg;
    std::string labels_path;
    std::string models_path = "./private/models";
    std::string input_path;
    std::string output_path;
    float threshold = 0.5f;
    float nms_threshold = 0.4f;
    bool use_gpu = false;
    bool eval_mode = false;
    bool list_models_flag = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            model_arg = argv[++i];
        } else if ((arg == "-l" || arg == "--labels") && i + 1 < argc) {
            labels_path = argv[++i];
        } else if (arg == "--models-path" && i + 1 < argc) {
            models_path = argv[++i];
        } else if ((arg == "-i" || arg == "--input") && i + 1 < argc) {
            input_path = argv[++i];
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            output_path = argv[++i];
        } else if ((arg == "-t" || arg == "--threshold") && i + 1 < argc) {
            threshold = std::stof(argv[++i]);
        } else if ((arg == "-n" || arg == "--nms") && i + 1 < argc) {
            nms_threshold = std::stof(argv[++i]);
        } else if (arg == "--gpu") {
            use_gpu = true;
        } else if (arg == "--eval") {
            eval_mode = true;
        } else if (arg == "--list-models") {
            list_models_flag = true;
        } else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }

    // Handle --list-models
    if (list_models_flag) {
        return listModels(models_path);
    }

    // Handle evaluation mode
    if (eval_mode) {
        if (model_arg.empty()) {
            std::cerr << "Error: Model name is required for evaluation mode (-m)\n" << std::endl;
            printUsage(argv[0]);
            return 1;
        }
        if (input_path.empty()) {
            std::cerr << "Error: Input path is required for evaluation mode (-i)\n" << std::endl;
            printUsage(argv[0]);
            return 1;
        }
        if (output_path.empty()) {
            std::cerr << "Error: Output path is required for evaluation mode (-o)\n" << std::endl;
            printUsage(argv[0]);
            return 1;
        }

        EvaluationConfig eval_config;
        eval_config.model_name = model_arg;
        eval_config.models_path = models_path;
        eval_config.input_path = input_path;
        eval_config.output_path = output_path;
        eval_config.confidence_threshold = threshold;
        eval_config.nms_threshold = nms_threshold;
        eval_config.use_gpu = use_gpu;

        return runEvalMode(eval_config);
    }

    // Server mode
#ifdef USE_TFLITE
    if (model_arg.empty()) {
        std::cerr << "Error: Model is required (-m)\n" << std::endl;
        printUsage(argv[0]);
        return 1;
    }
#else
    if (model_arg.empty()) {
        model_arg = "(placeholder - TFLite disabled)";
    }
#endif

    return runServerMode(model_arg, labels_path, models_path,
                         threshold, nms_threshold, use_gpu);
}