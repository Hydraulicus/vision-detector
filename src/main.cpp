#include <iostream>
#include <string>
#include <csignal>

#include "detector.h"
#include "preprocessor.h"
#include "server.h"

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
    std::cout << "Usage: " << program << " [options]\n"
              << "\nOptions:\n"
              << "  -m, --model <path>     Path to TFLite model file (required)\n"
              << "  -l, --labels <path>    Path to labels file (optional)\n"
              << "  -t, --threshold <val>  Confidence threshold (default: 0.5)\n"
              << "  -n, --nms <val>        NMS threshold (default: 0.4)\n"
              << "  --gpu                  Use GPU delegate\n"
              << "  -h, --help             Show this help\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "Vision Detector Service v1.0\n" << std::endl;

    // Parse arguments
    DetectorConfig config;
    std::string labels_path;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            config.model_path = argv[++i];
        } else if ((arg == "-l" || arg == "--labels") && i + 1 < argc) {
            labels_path = argv[++i];
        } else if ((arg == "-t" || arg == "--threshold") && i + 1 < argc) {
            config.confidence_threshold = std::stof(argv[++i]);
        } else if ((arg == "-n" || arg == "--nms") && i + 1 < argc) {
            config.nms_threshold = std::stof(argv[++i]);
        } else if (arg == "--gpu") {
            config.use_gpu = true;
        } else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }

    if (config.model_path.empty()) {
        std::cerr << "Error: Model path is required\n" << std::endl;
        printUsage(argv[0]);
        return 1;
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