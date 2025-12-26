#include "server.h"
#include "detector.h"
#include "preprocessor.h"

#include <detector_protocol/protocol.h>
#include <detector_protocol/ipc_common.h>

#include <iostream>
#include <cstring>
#include <vector>

namespace vision_detector {

using namespace detector_protocol;

DetectionServer::DetectionServer() = default;

DetectionServer::~DetectionServer() {
    stop();
    cleanup();
}

bool DetectionServer::initialize(Detector* detector, Preprocessor* preprocessor) {
    detector_ = detector;
    preprocessor_ = preprocessor;

    if (!detector_ || !preprocessor_) {
        std::cerr << "Invalid detector or preprocessor" << std::endl;
        return false;
    }

    // Create shared memory
    shm_ptr_ = SharedMemory::create(SHM_NAME, SHM_SIZE, shm_fd_);
    if (!shm_ptr_) {
        std::cerr << "Failed to create shared memory" << std::endl;
        return false;
    }

    // Create server socket
    server_fd_ = UnixSocket::createServer(SOCKET_PATH);
    if (server_fd_ < 0) {
        std::cerr << "Failed to create server socket" << std::endl;
        cleanup();
        return false;
    }

    std::cout << "Server initialized:" << std::endl;
    std::cout << "  Socket: " << SOCKET_PATH << std::endl;
    std::cout << "  Shared memory: " << SHM_NAME << std::endl;

    return true;
}

bool DetectionServer::start(bool blocking) {
    if (running_) {
        return false;
    }

    running_ = true;

    if (blocking) {
        runLoop();
    } else {
        server_thread_ = std::thread(&DetectionServer::runLoop, this);
    }

    return true;
}

void DetectionServer::stop() {
    running_ = false;

    // Close client connection to unblock accept()
    if (client_fd_ >= 0) {
        UnixSocket::close(client_fd_);
        client_fd_ = -1;
    }

    if (server_thread_.joinable()) {
        server_thread_.join();
    }
}

float DetectionServer::getAverageInferenceTimeMs() const {
    if (frames_processed_ == 0) return 0.0f;
    return total_inference_time_ms_ / frames_processed_;
}

void DetectionServer::runLoop() {
    std::cout << "Server running, waiting for connections..." << std::endl;

    while (running_) {
        // Accept client connection
        client_fd_ = UnixSocket::accept(server_fd_);
        if (client_fd_ < 0) {
            if (running_) {
                std::cerr << "Accept failed" << std::endl;
            }
            continue;
        }

        std::cout << "Client connected" << std::endl;

        // Handle client
        if (!handleClient()) {
            std::cerr << "Client handler error" << std::endl;
        }

        std::cout << "Client disconnected" << std::endl;
        UnixSocket::close(client_fd_);
        client_fd_ = -1;
    }
}

bool DetectionServer::handleClient() {
    // Receive handshake
    HandshakeRequest request;
    ssize_t n = recv(client_fd_, &request, sizeof(request), 0);
    if (n != sizeof(request) || request.type != MessageType::HANDSHAKE_REQUEST) {
        std::cerr << "Invalid handshake request" << std::endl;
        return false;
    }

    std::cout << "Handshake: protocol v" << request.protocol_version << std::endl;

    // Send handshake response
    HandshakeResponse response;
    response.type = MessageType::HANDSHAKE_RESPONSE;
    response.protocol_version = PROTOCOL_VERSION;
    response.accepted = (request.protocol_version == PROTOCOL_VERSION) ? 1 : 0;
    response.model_input_width = detector_->getInputWidth();
    response.model_input_height = detector_->getInputHeight();
    response.num_classes = detector_->getNumClasses();
    std::strncpy(response.model_name, detector_->getModelName().c_str(),
                 sizeof(response.model_name) - 1);

    if (send(client_fd_, &response, sizeof(response), 0) != sizeof(response)) {
        std::cerr << "Failed to send handshake response" << std::endl;
        return false;
    }

    if (!response.accepted) {
        std::cerr << "Protocol version mismatch" << std::endl;
        return false;
    }

    // Main processing loop
    while (running_) {
        FrameReadyMessage frame_msg;
        n = recv(client_fd_, &frame_msg, sizeof(frame_msg), 0);

        if (n == 0) {
            // Client disconnected
            break;
        }

        if (n < 0) {
            std::cerr << "Receive error" << std::endl;
            break;
        }

        if (frame_msg.type == MessageType::SHUTDOWN) {
            std::cout << "Client requested shutdown" << std::endl;
            break;
        }

        if (frame_msg.type == MessageType::HEARTBEAT) {
            // Echo heartbeat with correct size
            HeartbeatMessage heartbeat;
            heartbeat.type = MessageType::HEARTBEAT;
            heartbeat.timestamp_ns = reinterpret_cast<HeartbeatMessage*>(&frame_msg)->timestamp_ns;
            send(client_fd_, &heartbeat, sizeof(HeartbeatMessage), 0);
            continue;
        }

        if (frame_msg.type != MessageType::FRAME_READY) {
            std::cerr << "Unexpected message type" << std::endl;
            continue;
        }

        // Process frame
        if (!processFrame(frame_msg)) {
            std::cerr << "Frame processing failed" << std::endl;
        }
    }

    return true;
}

bool DetectionServer::processFrame(const FrameReadyMessage& msg) {
    // Read frame from shared memory
    auto* header = reinterpret_cast<FrameHeader*>(shm_ptr_);
    auto* frame_data = reinterpret_cast<uint8_t*>(shm_ptr_) + sizeof(FrameHeader);

    if (header->frame_id != msg.frame_id) {
        std::cerr << "Frame ID mismatch" << std::endl;
        return false;
    }

    // Preprocess
    std::vector<float> input_tensor;
    preprocessor_->process(
        frame_data,
        header->width,
        header->height,
        header->stride,
        input_tensor
    );

    // Run detection
    auto detections = detector_->detect(
        input_tensor.data(),
        preprocessor_->getTargetWidth(),
        preprocessor_->getTargetHeight()
    );

    // Build result message
    DetectionResultMessage result;
    result.type = MessageType::DETECTION_RESULT;
    result.frame_id = msg.frame_id;
    result.num_detections = std::min(static_cast<size_t>(detections.size()),
                                      static_cast<size_t>(MAX_DETECTIONS));
    result.inference_time_ms = detector_->getLastInferenceTimeMs();

    for (size_t i = 0; i < result.num_detections; ++i) {
        result.detections[i] = detections[i];
    }

    // Send results
    if (!sendResults(result)) {
        return false;
    }

    // Update stats
    frames_processed_++;
    total_inference_time_ms_ += result.inference_time_ms;

    return true;
}

bool DetectionServer::sendResults(const DetectionResultMessage& result) {
    // Calculate actual message size (only send used detections)
    size_t msg_size = offsetof(DetectionResultMessage, detections) +
                      result.num_detections * sizeof(Detection);

    ssize_t n = send(client_fd_, &result, msg_size, 0);
    return n == static_cast<ssize_t>(msg_size);
}

void DetectionServer::cleanup() {
    if (client_fd_ >= 0) {
        UnixSocket::close(client_fd_);
        client_fd_ = -1;
    }

    if (server_fd_ >= 0) {
        UnixSocket::close(server_fd_);
        UnixSocket::unlink(SOCKET_PATH);
        server_fd_ = -1;
    }

    if (shm_ptr_) {
        SharedMemory::close(shm_ptr_, SHM_SIZE, shm_fd_);
        SharedMemory::unlink(SHM_NAME);
        shm_ptr_ = nullptr;
        shm_fd_ = -1;
    }
}

}  // namespace vision_detector