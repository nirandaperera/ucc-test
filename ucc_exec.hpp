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
  explicit UccProgressManager(ucc_context_h ctx) : ctx(ctx), shutdown(false) {
    progress_thread = std::thread(&UccProgressManager::Progress, this);
  }

  ucc_status_t Progress() {
    while (!shutdown.load()) {
      ucc_context_progress(ctx);
    }

    return UCC_OK;
  }

  void Shutdown() {
    shutdown.store(true);
    progress_thread.join();
  }

 private:
  std::thread progress_thread;
  ucc_context_h ctx;
  std::atomic<bool> shutdown;
};

}