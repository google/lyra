// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "sparse_matmul/os/coop_threads.h"

#include <atomic>

namespace csrblocksparse {

// All threads must execute a std::memory_order_seq_cst operation on
// |barrier_step_| this is what ensures the global memory consistency across
// the barrier.
//
// It is possible for the |barrier_step_| to roll over, but this is safe here.
//
// |yield| instructs the processor that it is in a spin loop and can stop doing
// things like out of order, speculative execution, prefetching, etc.  On hyper
// threaded machines it can also choose to swap in the other thread.  Note that
// this is a hardware level decision and the OS is never involved.
void SpinBarrier::barrier() {
  if (num_threads_ < 2) return;

  int old_step = barrier_step_.load(std::memory_order_relaxed);

  int val_threads = threads_at_barrier_.fetch_add(1, std::memory_order_acq_rel);

  if (val_threads == num_threads_ - 1) {
    // This is where the logic can go all wrong if the barrier is called by
    // more threads than |num_threads_| -- the assumption that we're the last
    // thread is inherently invalid.

    // Assuming num_threads_ are calling this barrier, then we're the last
    // thread to reach the barrier, reset and advance step count.
    threads_at_barrier_.store(0, std::memory_order_relaxed);
    barrier_step_.store(old_step + 1, std::memory_order_release);
  } else {
    // Wait for step count to advance, then continue.
    while (barrier_step_.load(std::memory_order_acquire) == old_step) {
      // Intel recommends the equivalent instruction PAUSE, not be called more
      // than once in a row, I can't find any recommendations for ARM, so
      // following that advice here.
#if defined __aarch64__ || defined __arm__
      asm volatile("yield\n" ::: "memory");
#else
      // No pause for x86! The pause instruction on Skylake takes 141 clock
      // cycles, which in an AVX2-down-clocked CPU is getting on for 70ns.
#endif
    }
  }
}

}  // namespace csrblocksparse
