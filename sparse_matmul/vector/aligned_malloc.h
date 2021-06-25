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

#ifndef LYRA_CODEC_SPARSE_MATMUL_VECTOR_ALIGNED_MALLOC_H_
#define LYRA_CODEC_SPARSE_MATMUL_VECTOR_ALIGNED_MALLOC_H_

#include <memory>
namespace csrblocksparse {

void Free(void* ptr);

void* Malloc(size_t size);

void aligned_free(void* aligned_memory);

void* aligned_malloc(size_t size, int minimum_alignment);
}  // namespace csrblocksparse

#endif  // LYRA_CODEC_SPARSE_MATMUL_VECTOR_ALIGNED_MALLOC_H_
