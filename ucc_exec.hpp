//
// Created by niranda on 11/14/23.
//
#pragma once

#include <atomic>
#include <ucc/api/ucc.h>
#include <thread>

namespace ucc_test {

class UccProgressManager {
 public:
  explicit UccProgressManager() : ctx_(nullptr), shutdown_(false) {
  }

  void Init(ucc_context_h ctx) {
    if (shutdown_) return;
    ctx_ = ctx;
    progress_thread_ = std::thread(&UccProgressManager::Progress, this);
  }

  ucc_status_t Progress() {
    while (!shutdown_.load()) {
      ucc_context_progress(ctx_);
      std::this_thread::yield();
    }

    return UCC_OK;
  }

  void Shutdown() {
    shutdown_.store(true);
    progress_thread_.join();
  }

 private:
  std::thread progress_thread_;
  ucc_context_h ctx_;
  std::atomic<bool> shutdown_;
};

}