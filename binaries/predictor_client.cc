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
#include <stdio.h>

#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include <vector>
#include "caffe2/core/predictor.h"
#include "caffe2/core/timer.h"
#include "caffe2/utils/proto_utils.h"

using namespace caffe2;

int main(int argc, char* argv[]) {
  int fd = -1;
  size_t size = 0;
  if (argc > 2) {
    fd = atoi(argv[1]);
    size = atol(argv[2]);
  }
  void* addr = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

  int64_t* offsets = reinterpret_cast<int64_t*>(addr);
  int64_t input_offset = offsets[0];
  int64_t output_offset = offsets[1];
  int64_t cond_offset = offsets[2];
  int64_t mutex_offset = offsets[3];
  int64_t* input_dims = reinterpret_cast<int64_t*>((char*)addr + input_offset);
  int64_t* output_dims =
      reinterpret_cast<int64_t*>((char*)addr + output_offset);
  float* input_data = reinterpret_cast<float*>(
      (char*)addr + input_offset + 4 * sizeof(int64_t));
  float* output_data = reinterpret_cast<float*>(
      (char*)addr + output_offset + 4 * sizeof(int64_t));
  pthread_mutex_t* mutex_ptr =
      reinterpret_cast<pthread_mutex_t*>((char*)addr + mutex_offset);
  pthread_cond_t* cond_ptr =
      reinterpret_cast<pthread_cond_t*>((char*)addr + cond_offset);

  printf(
      "Input tensor is of dim %ld %ld %ld %ld\n",
      input_dims[0],
      input_dims[1],
      input_dims[2],
      input_dims[3]);
  printf(
      "Output tensor is of dim %ld %ld %ld %ld\n",
      output_dims[0],
      output_dims[1],
      output_dims[2],
      output_dims[3]);

  caffe2::TensorCPU input;
  input.Resize(vector<TIndex>{
      input_dims[0], input_dims[1], input_dims[2], input_dims[3]});
  input.ShareExternalPointer(input_data);
  caffe2::TensorCPU output;
  output.Resize(vector<TIndex>{
      output_dims[0], output_dims[1], output_dims[2], output_dims[3]});
  output.ShareExternalPointer(output_data);
  NetDef initNet;
  {
    vector<TIndex> shape = {1, 256, 256, 256};
    TensorCPU X(shape);
    for (auto i = 0; i < X.size(); ++i) {
      X.mutable_data<float>()[i] = 123;
    }
    vector<float> v(X.data<float>(), X.data<float>() + X.size());
    auto op = CreateOperatorDef(
        "GivenTensorFill",
        "",
        {},
        {"W"},
        {MakeArgument<vector<TIndex>>("shape", shape),
         MakeArgument<vector<float>>("values", v)});
    *initNet.add_op() = op;
  }
  {
    vector<TIndex> shape = {1};
    TensorCPU X(shape);
    for (auto i = 0; i < X.size(); ++i) {
      X.mutable_data<float>()[i] = 0;
    }
    vector<float> v(X.data<float>(), X.data<float>() + X.size());
    auto op = CreateOperatorDef(
        "GivenTensorFill",
        "",
        {},
        {"b"},
        {MakeArgument<vector<TIndex>>("shape", shape),
         MakeArgument<vector<float>>("values", v)});
    *initNet.add_op() = op;
  }
  NetDef predictNet;
  predictNet.set_name("test_net");
  {
    auto op = CreateOperatorDef("FC", "", {"data", "W", "b"}, {"Y"});
    *predictNet.add_op() = op;
  }
  *predictNet.add_external_input() = std::string("data");
  *predictNet.add_external_input() = std::string("W");
  *predictNet.add_external_input() = std::string("b");
  *predictNet.add_external_output() = std::string("Y");

  Predictor predictor(initNet, predictNet);
  fprintf(stderr, "Finished initializing predictor.\n");
  pthread_mutex_lock(mutex_ptr);
  pthread_cond_signal(cond_ptr);
  pthread_mutex_unlock(mutex_ptr);

  Predictor::TensorVector inputvec{&input};
  Predictor::TensorVector outputvec{&output};

  while (true) {
    fprintf(stderr, "Waiting for signal to run.\n");
    pthread_mutex_lock(mutex_ptr);
    pthread_cond_wait(cond_ptr, mutex_ptr);
    pthread_mutex_unlock(mutex_ptr);

    Timer t;
    predictor.run_preallocated_output(inputvec, &outputvec);

    fprintf(
        stderr,
        "Finished running predictor locally in %f microseconds.\n",
        t.MicroSeconds());
    pthread_mutex_lock(mutex_ptr);
    pthread_cond_signal(cond_ptr);
    pthread_mutex_unlock(mutex_ptr);
  }
}
