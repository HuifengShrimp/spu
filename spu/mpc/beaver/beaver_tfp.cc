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

#include "spu/mpc/beaver/beaver_tfp.h"

#include <random>

#include "yasl/link/link.h"
#include "yasl/utils/serialize.h"

#include "spu/mpc/beaver/prg_tensor.h"
#include "spu/mpc/util/ring_ops.h"

//lj

namespace spu::mpc {
namespace {

uint128_t GetHardwareRandom128() {
  std::random_device rd;
  // call random_device four times, make sure uint128 is random in 2^128 set.
  uint64_t lhs = static_cast<uint64_t>(rd()) << 32 | rd();
  uint64_t rhs = static_cast<uint64_t>(rd()) << 32 | rd();
  return yasl::MakeUint128(lhs, rhs);
}

}  // namespace

BeaverTfpUnsafe::BeaverTfpUnsafe(std::shared_ptr<yasl::link::Context> lctx)
    : lctx_(lctx), seed_(GetHardwareRandom128()), counter_(0) {
  auto buf = yasl::SerializeUint128(seed_);
  std::vector<yasl::Buffer> all_bufs =
      yasl::link::Gather(lctx_, buf, 0, "BEAVER_TFP:SYNC_SEEDS");

  if (lctx_->Rank() == 0) {
    // Collects seeds from all parties.
    for (size_t rank = 0; rank < lctx_->WorldSize(); ++rank) {
      PrgSeed seed = yasl::DeserializeUint128(all_bufs[rank]);
      tp_.setSeed(rank, lctx_->WorldSize(), seed);
    }
  }
}

Beaver::Triple BeaverTfpUnsafe::Mul(FieldType field, size_t size) {
  std::vector<PrgArrayDesc> descs(3);

  auto a = prgCreateArray(field, size, seed_, &counter_, &descs[0]);
  auto b = prgCreateArray(field, size, seed_, &counter_, &descs[1]);
  auto c = prgCreateArray(field, size, seed_, &counter_, &descs[2]);

  if (lctx_->Rank() == 0) {
     std::cout<<"***********Mul : distribute shares for rank 0**********"<<std::endl;
    c = tp_.adjustMul(descs);
  }

  return {a, b, c};
}

Beaver::Triple BeaverTfpUnsafe::Dot(FieldType field, size_t M, size_t N,
                                    size_t K) {
  std::vector<PrgArrayDesc> descs(3);

  auto a = prgCreateArray(field, M * K, seed_, &counter_, &descs[0]);
  auto b = prgCreateArray(field, K * N, seed_, &counter_, &descs[1]);
  auto c = prgCreateArray(field, M * N, seed_, &counter_, &descs[2]);

  if (lctx_->Rank() == 0) {
    std::cout<<"***********distribute shares for rank 0**********"<<std::endl;
    c = tp_.adjustDot(descs, M, N, K);
  }

  return {a, b, c};
}

Beaver::Triple BeaverTfpUnsafe::And(FieldType field, size_t size) {
  std::vector<PrgArrayDesc> descs(3);

  auto a = prgCreateArray(field, size, seed_, &counter_, &descs[0]);
  auto b = prgCreateArray(field, size, seed_, &counter_, &descs[1]);
  auto c = prgCreateArray(field, size, seed_, &counter_, &descs[2]);

  if (lctx_->Rank() == 0) {
    std::cout<<"***********And : distribute shares for rank 0**********"<<std::endl;
    c = tp_.adjustAnd(descs);
  }

  return {a, b, c};
}

Beaver::Pair BeaverTfpUnsafe::Trunc(FieldType field, size_t size, size_t bits) {
  std::vector<PrgArrayDesc> descs(2);

  auto a = prgCreateArray(field, size, seed_, &counter_, &descs[0]);
  auto b = prgCreateArray(field, size, seed_, &counter_, &descs[1]);

  if (lctx_->Rank() == 0) {
    std::cout<<"***********Trunc : distribute shares for rank 0**********"<<std::endl;
    b = tp_.adjustTrunc(descs, bits);
  }

  return {a, b};
}

//lj
Beaver::Lr_set BeaverTfpUnsafe::lr(FieldType field, size_t M, size_t N, size_t K){
  std::vector<PrgArrayDesc> descs(8);

  //Party 0
  auto r1 = prgCreateArray(field, M * N, seed_, &counter_, &descs[0]);
  auto r2 = prgCreateArray(field, K * N, seed_, &counter_, &descs[1]);
  auto r3 = prgCreateArray(field, M * K, seed_, &counter_, &descs[2]);
  
  //Party 1
  auto r1_ = ring_rand(field, M * N);
  auto r2_ = ring_rand(field, K * N);
  auto r3_ = ring_rand(field, M * K);

  size_t i = 0, j = 0, index;
  
  auto c1 = prgCreateArray(field, K * M, seed_, &counter_, &descs[3]);
  
  ArrayRef r1T(makeType<RingTy>(field), N * M);

  for ( i = 0 ; i < M * N ; i++){
    index = (i % N) * M + (i / N);
    r1T.at<int32_t>(index) = r1.at<int32_t>(i);
  }

  //Party 0
  c1 = ring_mmul(r2, r1T, K, M, N);

  //Party 1
  auto c1_ = ring_rand(field, K * M);

  //Party 0
  auto c2 = prgCreateArray(field, (M * N) * N, seed_, &counter_, &descs[4]);

  index = 0;
  for (i = 0 ; i < M * N; i++) {
    for (j = 0; j < N; j++) {
      c2.at<int32_t>(index) = r2.at<int32_t>(j % N) * r1T.at<int32_t>(i / N);
      index = index + 1;
    }
  }

  //Party 1
  auto c2_ = ring_rand(field, (M * N) * N);

  //Party 0
  auto c3 = prgCreateArray(field, K * N, seed_, &counter_, &descs[5]);

  c3 = ring_mmul(c1, r1, K, N, M);

  //Party 1
  auto c3_ = ring_rand(field, K * N);

  auto c4 = prgCreateArray(field, K * N, seed_, &counter_, &descs[6]);

  //Party 0
  c4 = ring_mmul(r3, r1, K, N, M);

  //Party 1
  auto c4_ = ring_rand(field, K * N);

  auto c5 =  prgCreateArray(field, N * N, seed_, &counter_, &descs[7]);

  //Party 0
  c5 = ring_mmul(r1T, r1, N, N, M);

  //Party 1
  auto c5_ = ring_rand(field, N * N);

  
  //0916 return respectively
  if(lctx_->Rank() == 0){
    ring_sub_(r1, r1_);
    ring_sub_(r2, r2_);
    ring_sub_(r3, r3_);
    ring_sub_(c1, c1_);
    ring_sub_(c2, c2_);
    ring_sub_(c3, c3_);
    ring_sub_(c4, c4_);
    ring_sub_(c5, c5_);
    return {r1, r2, r3, c1, c2, c3, c4, c5};
  }

  return {r1_, r2_, r3_, c1_, c2_, c3_, c4_, c5_} ;

}

ArrayRef BeaverTfpUnsafe::RandBit(FieldType field, size_t size) {
  PrgArrayDesc desc{};
  auto a = prgCreateArray(field, size, seed_, &counter_, &desc);

  if (lctx_->Rank() == 0) {
    a = tp_.adjustRandBit(desc);
  }

  return a;
}

}  // namespace spu::mpc
