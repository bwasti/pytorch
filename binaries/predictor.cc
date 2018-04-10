/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <fcntl.h>
#include <linux/ashmem.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#define ASHMEM_DEVICE "/dev/ashmem"

#include <vector>
#include "caffe2/core/predictor.h"
#include "caffe2/core/timer.h"

int ashmem_create_region(const char* name, size_t size) {
  int fd, ret;

  fd = open(ASHMEM_DEVICE, O_RDWR);
  if (fd < 0)
    return fd;

  if (name) {
    char buf[ASHMEM_NAME_LEN];

    strlcpy(buf, name, sizeof(buf));
    ret = ioctl(fd, ASHMEM_SET_NAME, buf);
    if (ret < 0)
      goto error;
  }

  ret = ioctl(fd, ASHMEM_SET_SIZE, size);
  if (ret < 0)
    goto error;

  fprintf(stderr, "Successfully set up shared memory.\n");
  return fd;

error:
  fprintf(stderr, "Error setting up shared memory.\n");
  close(fd);
  return ret;
}

using namespace caffe2;

int main() {
  // We save as [input offset in bytes][output offset in bytes][input data ...
  // ][output data ... ] input_data is [dim0][dim1][dim2][dim3][ raw data ... ]
  // where dims are all 64bit integers
  std::vector<TIndex> indims({1, 256, 256, 256});
  TensorCPU fakeInputTensor(indims);
  std::vector<TIndex> outdims({1, 256, 256, 256});
  TensorCPU fakeOutputTensor(outdims);

  // input offset, output offset, cond offset, mutex offset
  size_t header_size = sizeof(int64_t) * 4;
  size_t input_offset = header_size;
  size_t input_size = 4 * sizeof(int64_t) +
      256 * 256 * 256 * sizeof(float); // fakeInputTensor.nbytes();
  size_t output_offset = input_offset + input_size;
  size_t output_size = 4 * sizeof(int64_t) +
      256 * 256 * 256 * sizeof(float); // fakeOutputTensor.nbytes();
  size_t cond_offset = output_offset + output_size;
  size_t cond_size = sizeof(pthread_cond_t);
  size_t mutex_offset = cond_offset + cond_size;
  size_t mutex_size = sizeof(pthread_mutex_t);

  size_t size = header_size + input_size + output_size + cond_size + mutex_size;
  fprintf(stderr, "Size is %zu\n", size);
  int fd = ashmem_create_region("predictor_shared_mem", size);
  void* addr = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

  int64_t* offsets = reinterpret_cast<int64_t*>(addr);
  offsets[0] = input_offset;
  offsets[1] = output_offset;
  offsets[2] = cond_offset;
  offsets[3] = mutex_offset;
  int64_t* input_dims = reinterpret_cast<int64_t*>((char*)addr + input_offset);
  int64_t* output_dims =
      reinterpret_cast<int64_t*>((char*)addr + output_offset);
  for (auto i = 0; i < indims.size(); ++i) {
    input_dims[i] = indims[i];
  }
  for (auto i = 0; i < outdims.size(); ++i) {
    output_dims[i] = outdims[i];
  }
  pthread_cond_t* cond_ptr =
      reinterpret_cast<pthread_cond_t*>((char*)addr + cond_offset);
  pthread_condattr_t cond_attr;
  if (pthread_condattr_init(&cond_attr)) {
    fprintf(stderr, "pthread_condattr_init");
  }
  if (pthread_condattr_setpshared(&cond_attr, PTHREAD_PROCESS_SHARED)) {
    fprintf(stderr, "pthread_condattr_setpshared");
  }
  if (pthread_cond_init(cond_ptr, &cond_attr)) {
    fprintf(stderr, "pthread_cond_init");
  }
  pthread_mutex_t* mutex_ptr =
      reinterpret_cast<pthread_mutex_t*>((char*)addr + mutex_offset);
  pthread_mutexattr_t mutex_attr;
  if (pthread_mutexattr_init(&mutex_attr)) {
    fprintf(stderr, "pthread_mutexattr_init");
  }
  if (pthread_mutexattr_setpshared(&mutex_attr, PTHREAD_PROCESS_SHARED)) {
    fprintf(stderr, "pthread_mutexattr_setpshared");
  }
  if (pthread_mutex_init(mutex_ptr, &mutex_attr)) {
    fprintf(stderr, "pthread_mutex_init");
  }

  char fd_as_str[12]; // 2^32 < 1e11
  char size_as_str[21]; // 2^64 < 1e20
  sprintf(fd_as_str, "%d", fd);
  sprintf(size_as_str, "%zu", size);

  // Starts with client holding the condvar
  // and then getting pinged that it is has been freed
  // The parent then holds the condvar
  if (fork() == 0) {
    char* args[] = {
        "/data/local/tmp/predictor_client", fd_as_str, size_as_str, NULL};
    execvp(args[0], args);
  } else {
    // Wait until the predictor is ready, then initialize the memory
    pthread_mutex_lock(mutex_ptr);
    pthread_cond_wait(cond_ptr, mutex_ptr);
    pthread_mutex_unlock(mutex_ptr);
    fprintf(stderr, "Predictor is initialized.\n");

    for (auto i = 0; i < 10; ++i) {
      Timer t;
      // Run the predictor
      pthread_mutex_lock(mutex_ptr);
      pthread_cond_signal(cond_ptr);
      pthread_mutex_unlock(mutex_ptr);
      fprintf(stderr, "Signaled to run predictor.\n");

      // Wait for the output
      pthread_mutex_lock(mutex_ptr);
      pthread_cond_wait(cond_ptr, mutex_ptr);
      pthread_mutex_unlock(mutex_ptr);
      fprintf(
          stderr,
          "Remote predictor finished in %f microseconds\n",
          t.MicroSeconds());
    }
  }

  printf("Finished!\n");
}
