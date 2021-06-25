/*
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef LYRA_CODEC_SPARSE_MATMUL_OS_COOP_THREADS_H_
#define LYRA_CODEC_SPARSE_MATMUL_OS_COOP_THREADS_H_

#include <atomic>
#include <thread>  // NOLINT
#include <vector>

#define _COOP_THREADS_USE_STD_THREAD 1

#include "absl/memory/memory.h"
#include "glog/logging.h"

namespace csrblocksparse {

// A re-usable barrier. Keeps threads in extremely tight sync without
// relinquishing control.  All memory writes _before_ this barrier are visible
// to all threads _after_ this barrier.   Similar in spirit to
// pthreads_barrier.  If you expect arrival times at this barrier to be varied
// by more than microseconds, this is probably not the right synchronization
// primitive for you.  If |num_threads| exceeds the number of physical threads
// that can run simultaneously, then using this is certainly a bad idea
// (although it should still be correct).
//
// Callers MUST NOT call barrier from more threads than |num_threads|.  The
// result is undefined behavior.
class SpinBarrier {
 public:
  explicit SpinBarrier(int num_threads)
      : num_threads_(num_threads), threads_at_barrier_(0), barrier_step_(0) {}

  void barrier();

 private:
  const int num_threads_;
  std::atomic<int32_t> threads_at_barrier_;
  std::atomic<uint32_t> barrier_step_;  // unsigned to make overflow defined.
};

// Producer-consumer API using the same underlying mechanism as SpinBarrier.
// This class is intended to allow >=1 producers to produce data for >=1
// consumers, without blocking the producers.
// The consumer will block if it is ready before all the producer(s) have
// produced.
// WARNING: By design this lock does not work without some other barrier that
// prevents any producer from producing again, or consumer from consuming again
// until all consumers have consumed. Basically any loop that uses
// ProducerConsumer must have at least two consume() calls in each thread (on
// different instances) in order for the lock to work correctly.
class ProducerConsumer {
 public:
  ProducerConsumer(int num_producers, int num_consumers)
      : num_producers_(num_producers),
        num_consumers_(num_consumers),
        producers_ready_(0),
        consumers_passed_(0) {}

  // Indicates that the data produced by this thread is ready. Does NOT block.
  // NOTE that some other lock must exist between the call to this produce and
  // looping back to call produce again on the same ProducerConsumer, that
  // depends on all consumers having called consume. One such candidate would
  // be a call to SpinBarrier above by all producers and consumers.
  // Another candidate would be a separate ProducerConsumer object in which
  // these producers consume some data produced by the threads that consume
  // the data produced here. Eg.
  // tid      0       1       2       3
  // action 1 produce produce consume consume (on ProducerConsumer 1)
  // action 2 consume consume produce produce (on ProducerConsumer 2)
  // action 3 produce produce consume consume (on ProducerConsumer 3)
  // action 4 consume consume produce produce (on ProducerConsumer 4)
  // loop back to action 1.
  // NOTE: It is inadequate to loop back after action2, as thread 0 could loop
  // back and consume again on PC2 while thread 1 is still completing its call
  // to consume. It is still inadequate to loop back after action 3 for the same
  // reason (but tsan doesn't seem to pick this up.)
  inline void produce() {
    producers_ready_.fetch_add(1, std::memory_order_acq_rel);
  }

  // Waits if necessary for all producers to have produced before proceeding.
  // The ProducerConsumer cannot be reused until all consumers have consumed.
  // See detailed comment and example on produce().
  inline void consume() {
    // We can't do anything until all the producers have produced.
    while (producers_ready_.load(std::memory_order_acquire) < num_producers_) {
#if defined __aarch64__ || defined __arm__
      asm volatile("yield\n" ::: "memory");
#else
      // No pause for x86! The pause instruction on Skylake takes 141 clock
      // cycles, which in an AVX2-down-clocked CPU is getting on for 70ns.
#endif
    }
    // NOTE: It is tempting to move this fetch_add to before the wait loop to
    // reduce contention for the memory location, but that would break the lock,
    // as then the last to arrive could zero out the producers_ready before the
    // other consumers have noticed that all producers have produced.
    // With the fetch_add after the wait loop, we are guaranteed that all
    // producers have produced AND all consumers have noticed that they have
    // produced before we zero out the counters.
    int consumers = consumers_passed_.fetch_add(1, std::memory_order_acq_rel);
    if (consumers == num_consumers_ - 1) {
      // The last consumer to pass has to reset everything for the next time.
      producers_ready_.store(0, std::memory_order_relaxed);
      consumers_passed_.store(0, std::memory_order_relaxed);
    }
  }
  int num_producers() const { return num_producers_; }
  int num_consumers() const { return num_consumers_; }

 private:
  const int num_producers_;
  const int num_consumers_;
  std::atomic<int32_t> producers_ready_;
  std::atomic<int32_t> consumers_passed_;
};

// We define Thread here, so we can easily change its type later.

using Thread = std::thread;
using ThreadId = std::thread::id;

// Creates (|num_threads|-1) threads and executes a total of |num_threads|
// copies of |func| (executes one on the calling thread).
//
// Useful for long running func bodies that are intended to run in lock step.
// A possible use case for this style parallelism over a thread pool is when
// we want tight control over which memory is resident in the L2 cache of a
// processor.  With a pool we have no control over which thread gets assigned
// which portion of the computation resulting in L2 thrashing.  With this
// breakdown we can make sure each thread only acceses a specific L2-sized
// portion of memory.
//
// func's signature must be (SpinBarrier*, int thread_id, ...);
template <class Function, class... Args>
void LaunchOnThreadsWithBarrier(int num_threads, Function&& func,
                                Args&&... args) {
  SpinBarrier spin_barrier(num_threads);

  std::vector<std::unique_ptr<Thread>> threads;
  threads.reserve(num_threads);
  for (int tid = 1; tid < num_threads; ++tid) {
    auto f = [&, tid]() { func(&spin_barrier, tid, args...); };

    threads.emplace_back(absl::make_unique<Thread>(f));
#ifndef _COOP_THREADS_USE_STD_THREAD
    CHECK_OK(threads.back()->Start());
#endif
  }

  const int kLocalTid = 0;
  func(&spin_barrier, kLocalTid, args...);

  for (auto& thread : threads) {
#ifdef _COOP_THREADS_USE_STD_THREAD
    thread->join();
#else
    CHECK_OK(thread->Join());
#endif
  }
}

}  // namespace csrblocksparse

#endif  // LYRA_CODEC_SPARSE_MATMUL_OS_COOP_THREADS_H_
