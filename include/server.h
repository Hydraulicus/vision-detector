#ifndef VISION_DETECTOR_SERVER_H
#define VISION_DETECTOR_SERVER_H

#include <atomic>
#include <thread>
#include <functional>
#include <detector_protocol/protocol.h>

namespace vision_detector {

class Detector;
class Preprocessor;

class DetectionServer {
public:
    DetectionServer();
    ~DetectionServer();

    // Non-copyable
    DetectionServer(const DetectionServer&) = delete;
    DetectionServer& operator=(const DetectionServer&) = delete;

    // Initialize server with detector and preprocessor
    bool initialize(Detector* detector, Preprocessor* preprocessor);

    // Start server (blocking or non-blocking)
    bool start(bool blocking = true);

    // Stop server
    void stop();

    // Check if running
    bool isRunning() const { return running_; }

    // Get statistics
    uint64_t getFramesProcessed() const { return frames_processed_; }
    float getAverageInferenceTimeMs() const;

private:
    Detector* detector_ = nullptr;
    Preprocessor* preprocessor_ = nullptr;

    int server_fd_ = -1;
    int client_fd_ = -1;
    int shm_fd_ = -1;
    void* shm_ptr_ = nullptr;

    std::atomic<bool> running_{false};
    std::thread server_thread_;

    uint64_t frames_processed_ = 0;
    float total_inference_time_ms_ = 0.0f;

    // Server loop
    void runLoop();

    // Handle client connection
    bool handleClient();

    // Process incoming frame
    bool processFrame(const detector_protocol::FrameReadyMessage& msg);

    // Send detection results
    bool sendResults(const detector_protocol::DetectionResultMessage& result);

    // Cleanup resources
    void cleanup();
};

}  // namespace vision_detector

#endif  // VISION_DETECTOR_SERVER_H