// Copyright 2021 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// clang-format off
// To run the example, start two terminals:
// > bazel run //examples/cpp:fss_lr -- --dataset=examples/data/perfect_logit_a.csv --has_label=true
// > bazel run //examples/cpp:fss_lr -- --dataset=examples/data/perfect_logit_b.csv --rank=1
// > bazel run //examples/cpp:fss_lr -- --dataset=examples/data/breast_cancer_b.csv --has_label=true
// > bazel run //examples/cpp:fss_lr -- --dataset=examples/data/breast_cancer_a.csv --rank=1

// clang-format on

#include <fstream>
#include <iostream>
#include <vector>

#include "examples/cpp/utils.h"
#include "spdlog/spdlog.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xcsv.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"

#include "spu/device/io.h"
#include "spu/hal/hal.h"
#include "spu/hal/test_util.h"
#include "spu/hal/type_cast.h"

spu::hal::Value train_step(spu::HalContext* ctx, const spu::hal::Value& x,
                           const spu::hal::Value& y, const spu::hal::Value& w, size_t iter) {
    

  // Padding x
  auto padding = spu::hal::constant(ctx, 1.0F, {x.shape()[0], 1});
  auto padded_x =
      spu::hal::concatenate(ctx, {x, spu::hal::p2s(ctx, padding)}, 1);          

  auto grad = spu::hal::logreg(ctx, padded_x, w, y);

  SPDLOG_DEBUG("[FSS-LR] W = W - grad");
  
  // auto alpha = spu::hal::constant(ctx, 0.0001F);
  // auto lr_decay = spu::hal::constant(ctx, 1.0F);
  // auto a = spu::hal::mul(ctx, lr_decay, spu::hal::constant(ctx, (int32_t)iter));
  // auto b = spu::hal::add(ctx, lr_decay, a);
  // auto lr = spu::hal::mul(ctx, alpha, spu::hal::reciprocal(ctx, b));

  auto lr = spu::hal::constant(ctx, 0.000001F);

  auto msize = spu::hal::constant(ctx, static_cast<float>(y.shape()[0]));
  auto p1 = spu::hal::mul(ctx, lr, spu::hal::reciprocal(ctx, msize));
  auto step =
      spu::hal::mul(ctx, spu::hal::broadcast_to(ctx, p1, grad.shape()), grad);
  auto new_w = spu::hal::sub(ctx, w, step);

  return new_w;
}

spu::hal::Value train(spu::HalContext* ctx, const spu::hal::Value& x,
                      const spu::hal::Value& y, size_t num_epoch,
                      size_t bsize) {
  const size_t num_iter = x.shape()[0] / bsize;
  auto w = spu::hal::constant(ctx, 0.0F, {x.shape()[1] + 1, 1});

  // Run train loop
  for (size_t epoch = 0; epoch < num_epoch; ++epoch) {
    for (size_t iter = 0; iter < num_iter; ++iter) {
      SPDLOG_INFO("Running train iteration {}", iter);

      const int64_t rows_beg = iter * bsize;
      const int64_t rows_end = rows_beg + bsize;

      const auto x_slice =
          spu::hal::slice(ctx, x, {rows_beg, 0}, {rows_end, x.shape()[1]}, {});

      const auto y_slice =
          spu::hal::slice(ctx, y, {rows_beg, 0}, {rows_end, y.shape()[1]}, {});


      w = train_step(ctx, x_slice, y_slice, w, iter);
    }
  }

  return w;
}

spu::hal::Value inference(spu::HalContext* ctx, const spu::hal::Value& x,
                          const spu::hal::Value& weight) {
  auto padding = spu::hal::constant(ctx, 1.0F, {x.shape()[0], 1});
  auto padded_x =
      spu::hal::concatenate(ctx, {x, spu::hal::p2s(ctx, padding)}, 1);
  return spu::hal::matmul(ctx, padded_x, weight);
}

float SSE(const xt::xarray<float>& y_true, const xt::xarray<float>& y_pred) {
  float sse = 0;

  for (auto y_true_iter = y_true.begin(), y_pred_iter = y_pred.begin();
       y_true_iter != y_true.end() && y_pred_iter != y_pred.end();
       ++y_pred_iter, ++y_true_iter) {
    sse += std::pow(*y_true_iter - *y_pred_iter, 2);
  }
  return sse;
}

float MSE(const xt::xarray<float>& y_true, const xt::xarray<float>& y_pred) {
  auto sse = SSE(y_true, y_pred);

  return sse / static_cast<float>(y_true.size());
}

llvm::cl::opt<std::string> Dataset("dataset", llvm::cl::init("data.csv"),
                                   llvm::cl::desc("only csv is supported"));
llvm::cl::opt<uint32_t> SkipRows(
    "skip_rows", llvm::cl::init(1),
    llvm::cl::desc("skip number of rows from dataset"));
llvm::cl::opt<bool> HasLabel(
    "has_label", llvm::cl::init(false),
    llvm::cl::desc("if true, label is the last column of dataset"));
llvm::cl::opt<uint32_t> BatchSize("batch_size", llvm::cl::init(31),
                                  llvm::cl::desc("size of each batch"));
llvm::cl::opt<uint32_t> NumEpoch("num_epoch", llvm::cl::init(1),
                                 llvm::cl::desc("number of epoch"));

std::pair<spu::hal::Value, spu::hal::Value> infeed(spu::HalContext* hctx,
                                                   const xt::xarray<float>& ds,
                                                   bool self_has_label) {
  spu::device::ColocatedIo cio(hctx);
  if (self_has_label) {
    // the last column is label.
    using namespace xt::placeholders;  // required for `_` to work
    xt::xarray<float> dx =
        xt::view(ds, xt::all(), xt::range(_, ds.shape(1) - 1));
    xt::xarray<float> dy =
        xt::view(ds, xt::all(), xt::range(ds.shape(1) - 1, _));
    cio.hostSetVar(fmt::format("x-{}", hctx->lctx()->Rank()), dx);
    cio.hostSetVar("label", dy);
  } else {
    cio.hostSetVar(fmt::format("x-{}", hctx->lctx()->Rank()), ds);
  }
  cio.sync();

  auto x = cio.deviceGetVar("x-0");
  // Concatnate all slices
  for (size_t idx = 1; idx < cio.getWorldSize(); ++idx) {
    x = spu::hal::concatenate(
        hctx, {x, cio.deviceGetVar(fmt::format("x-{}", idx))}, 1);
  }
  auto y = cio.deviceGetVar("label");

  return std::make_pair(x, y);
}

int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  // read dataset.
  xt::xarray<float> ds;
  {
    std::ifstream file(Dataset.getValue());
    if (!file) {
      spdlog::error("open file={} failed", Dataset.getValue());
      exit(-1);
    }
    ds = xt::load_csv<float>(file, ',', SkipRows.getValue());
  }

  auto hctx = MakeHalContext();

  const auto& [x, y] = infeed(hctx.get(), ds, HasLabel.getValue());

  // std::cout<<"*************x&y***************"<<std::endl;
  // std::cout<<x.numel()<<std::endl;
  // std::cout<<y.numel()<<std::endl;

  const auto w =
      train(hctx.get(), x, y, NumEpoch.getValue(), BatchSize.getValue());

  

  const auto scores = inference(hctx.get(), x, w);

  xt::xarray<float> revealed_labels = spu::hal::test::dump_public_as<float>(
      hctx.get(), spu::hal::reveal(hctx.get(), y));
  xt::xarray<float> revealed_scores = spu::hal::test::dump_public_as<float>(
      hctx.get(), spu::hal::reveal(hctx.get(), scores));

  auto mse = MSE(revealed_labels, revealed_scores);
  std::cout << "MSE = " << mse << "\n";

  // std::cout<<"************file***********"<<std::endl;
  // std::cout<<Dataset.getValue()<<std::endl;
  return 0;
}
