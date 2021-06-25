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

#include <algorithm>
#include <atomic>
#include <numeric>

#include "gtest/gtest.h"

TEST(Threads, LaunchThreads) {
  std::atomic<int> counter(0);

  auto f = [&](csrblocksparse::SpinBarrier* barrier, int tid) {
    counter.fetch_add(tid);
  };

  const int kNumThreads = 10;
  csrblocksparse::LaunchOnThreadsWithBarrier(kNumThreads, f);

  ASSERT_EQ(counter.load(), kNumThreads * (kNumThreads - 1) / 2);
}

TEST(Threads, SpinBarrier) {
  const int kNumThreads = 10;

  std::vector<int> tids(kNumThreads, 0);
  std::vector<std::vector<int>> expected;
  for (int i = 0; i < 10; ++i) {
    expected.emplace_back(kNumThreads);
    std::iota(expected.back().begin(), expected.back().end(), 0);
    std::transform(expected.back().begin(), expected.back().end(),
                   expected.back().begin(),
                   [i](int x) -> int { return (i + 1) * x; });
  }

  auto f = [&](csrblocksparse::SpinBarrier* barrier, int tid) {
    for (int i = 0; i < 10; ++i) {
      tids[tid] += tid;
      barrier->barrier();
      EXPECT_EQ(tids, expected[i]);
      barrier->barrier();
    }
  };

  csrblocksparse::LaunchOnThreadsWithBarrier(kNumThreads, f);
}

TEST(Threads, ProducerConsumer) {
  constexpr int kNumThreads = 4;
  constexpr int kNumIterations = 10;

  std::vector<int> shared_data(kNumThreads, 0);
  std::vector<std::pair<int, int>> expected;
  for (int i = 1; i <= kNumIterations; ++i) {
    // Execute the parallel work sequentially.
    // Last two threads write their id * iteration.
    std::pair<int, int> inputs =
        std::make_pair((kNumThreads - 2) * i, (kNumThreads - 1) * i);
    // First two threads compute sum and difference of those values.
    std::pair<int, int> diffs = std::make_pair(inputs.first + inputs.second,
                                               inputs.first - inputs.second);
    // Last two threads compute sum and product.
    std::pair<int, int> sums =
        std::make_pair(diffs.first + diffs.second, diffs.first * diffs.second);
    // First two threads compute product and difference of those values.
    expected.emplace_back(
        std::make_pair(sums.first * sums.second, sums.first - sums.second));
    // Last two threads will check for the correct result.
  }
  csrblocksparse::ProducerConsumer first_pc(2, 2);
  csrblocksparse::ProducerConsumer second_pc(2, 2);
  csrblocksparse::ProducerConsumer third_pc(2, 2);
  csrblocksparse::ProducerConsumer fourth_pc(2, 2);

  auto f = [&](csrblocksparse::SpinBarrier* barrier, int tid) {
    for (int i = 1; i <= kNumIterations; ++i) {
      if (tid == kNumThreads - 2) {
        // Last two threads write their id * iteration.
        shared_data[tid] = tid * i;
        first_pc.produce();
        second_pc.consume();
        // They then compute sum and product.
        shared_data[tid] = shared_data[0] + shared_data[1];
        third_pc.produce();
        // They finally check the result.
        fourth_pc.consume();
        EXPECT_EQ(expected[i - 1].first, shared_data[0]) << "i=" << i;
      } else if (tid == kNumThreads - 1) {
        shared_data[tid] = tid * i;
        first_pc.produce();
        second_pc.consume();
        shared_data[tid] = shared_data[0] * shared_data[1];
        third_pc.produce();
        fourth_pc.consume();
        EXPECT_EQ(expected[i - 1].second, shared_data[1]) << "i=" << i;
      } else if (tid == 0) {
        // First two threads compute sum and difference.
        first_pc.consume();
        shared_data[tid] =
            shared_data[kNumThreads - 2] + shared_data[kNumThreads - 1];
        second_pc.produce();
        // They then compute product and difference.
        third_pc.consume();
        shared_data[tid] =
            shared_data[kNumThreads - 2] * shared_data[kNumThreads - 1];
        fourth_pc.produce();
      } else if (tid == 1) {
        first_pc.consume();
        shared_data[tid] =
            shared_data[kNumThreads - 2] - shared_data[kNumThreads - 1];
        second_pc.produce();
        third_pc.consume();
        shared_data[tid] =
            shared_data[kNumThreads - 2] - shared_data[kNumThreads - 1];
        fourth_pc.produce();
      }
    }
  };

  csrblocksparse::LaunchOnThreadsWithBarrier(kNumThreads, f);
}
