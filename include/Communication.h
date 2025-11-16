#pragma once

#include <string>
#include <thread>
#include <atomic>
#include <functional>
#include <json/json.h>  // We'll use jsoncpp library

class Communication {
public:
    using MessageCallback = std::function<void(const Json::Value&)>;

    Communication();
    ~Communication();

    bool connectToIDE(const std::string& host = "localhost", int port = 9999);
    void disconnectFromIDE();
    
    bool sendMessage(const Json::Value& message);
    void setMessageCallback(MessageCallback callback);
    
    bool isConnected() const { return connected_; }

private:
    std::string host_;
    int port_;
    int socket_fd_;
    std::atomic<bool> connected_;
    
    std::thread receive_thread_;
    std::atomic<bool> running_;
    
    MessageCallback message_callback_;
    
    void receiveLoop();
    bool sendRawData(const std::string& data);
    Json::Value parseMessage(const std::string& data);
};