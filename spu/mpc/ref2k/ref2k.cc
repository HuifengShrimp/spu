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

#include "spu/mpc/ref2k/ref2k.h"

#include <mutex>

#include "spu/core/type.h"
#include "spu/mpc/common/prg_state.h"
#include "spu/mpc/common/pub2k.h"
#include "spu/mpc/util/ring_ops.h"

namespace spu::mpc {
namespace {

class Ref2kSecrTy : public TypeImpl<Ref2kSecrTy, RingTy, Secret> {
  using Base = TypeImpl<Ref2kSecrTy, RingTy, Secret>;

 public:
  using Base::Base;
  static std::string_view getStaticId() { return "ref2k.Sec"; }
  explicit Ref2kSecrTy(FieldType field) { field_ = field; }
};

void registerTypes() {
  regPub2kTypes();

  static std::once_flag flag;
  std::call_once(
      flag, []() { TypeContext::getTypeContext()->addTypes<Ref2kSecrTy>(); });
}

class Ref2kP2S : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "p2s";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override {
    return in.as(makeType<Ref2kSecrTy>(in.eltype().as<Ring2k>()->field()));
  }
};

class Ref2kS2P : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "s2p";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override {
    return in.as(makeType<Pub2kTy>(in.eltype().as<Ring2k>()->field()));
  }
};

class Ref2kRandS : public Kernel {
 public:
  static constexpr char kBindName[] = "rand_s";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  void evaluate(KernelEvalContext* ctx) const override {
    ctx->setOutput(
        proc(ctx, ctx->getParam<FieldType>(0), ctx->getParam<size_t>(1)));
  }

  ArrayRef proc(KernelEvalContext* ctx, FieldType field, size_t size) const {
    SPU_PROFILE_TRACE_KERNEL(ctx, size);
    auto* state = ctx->caller()->getState<PrgState>();
    return state->genPubl(field, size).as(makeType<Ref2kSecrTy>(field));
  }
};

class Ref2kNotS : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "not_s";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override {
    SPU_PROFILE_TRACE_KERNEL(ctx, in);
    const auto field = in.eltype().as<Ring2k>()->field();
    return ring_not(in).as(makeType<Ref2kSecrTy>(field));
  }
};

class Ref2kEqzS : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "eqz_s";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override {
    SPU_PROFILE_TRACE_KERNEL(ctx, in);
    const auto field = in.eltype().as<Ring2k>()->field();
    return ring_equal(in, ring_zeros(field, in.numel())).as(in.eltype());
  }
};

class Ref2kAddSS : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "add_ss";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_PROFILE_TRACE_KERNEL(ctx, lhs, rhs);
    YASL_ENFORCE(lhs.eltype() == rhs.eltype());
    return ring_add(lhs, rhs).as(lhs.eltype());
  }
};

class Ref2kAddSP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "add_sp";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_PROFILE_TRACE_KERNEL(ctx, lhs, rhs);
    return ring_add(lhs, rhs).as(lhs.eltype());
  }
};

class Ref2kMulSS : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "mul_ss";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_PROFILE_TRACE_KERNEL(ctx, lhs, rhs);
    YASL_ENFORCE(lhs.eltype() == rhs.eltype());
    return ring_mul(lhs, rhs).as(lhs.eltype());
  }
};

class Ref2kMulSP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "mul_sp";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_PROFILE_TRACE_KERNEL(ctx, lhs, rhs);
    return ring_mul(lhs, rhs).as(lhs.eltype());
  }
};

class Ref2kMatMulSS : public MatmulKernel {
 public:
  static constexpr char kBindName[] = "mmul_ss";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs, size_t M, size_t N,
                size_t K) const override {
    SPU_PROFILE_TRACE_KERNEL(ctx, lhs, rhs);
    YASL_ENFORCE(lhs.eltype() == rhs.eltype());
    return ring_mmul(lhs, rhs, M, N, K).as(lhs.eltype());
  }
};

class Ref2kMatMulSP : public MatmulKernel {
 public:
  static constexpr char kBindName[] = "mmul_sp";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs, size_t M, size_t N,
                size_t K) const override {
    SPU_PROFILE_TRACE_KERNEL(ctx, lhs, rhs);
    return ring_mmul(lhs, rhs, M, N, K).as(lhs.eltype());
  }
};

//lj
class Ref2kLogRegSS : public LogRegKernel {
 public:
  static constexpr char kBindName[] = "logreg_ss";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs, const ArrayRef& mhs,
                const ArrayRef& rhs, size_t M, size_t N,
                size_t K) const override {
    SPU_PROFILE_TRACE_KERNEL(ctx, lhs, mhs, rhs);
    YASL_ENFORCE(lhs.eltype() == mhs.eltype());
    YASL_ENFORCE(mhs.eltype() == rhs.eltype());

    const auto field = lhs.eltype().as<Ring2k>()->field();
    size_t i = 0, index;
    ArrayRef x_(makeType<RingTy>(field), N * M);
    for ( i = 0 ; i < lhs.elsize() ; i++){
      index = (i % N) * M + (i / N);
      x_.at<int32_t>(index) = lhs.at<int32_t>(i);
    }
    auto y_pred = ring_mmul(mhs, x_, 1, M, N);
    auto err = ring_sub(y_pred, rhs);
    auto grad = ring_mmul(err, lhs, 1, N, M);
    return grad.as(lhs.eltype());
    }
};

class Ref2kLogRegPP : public LogRegKernel {
 public:
  static constexpr char kBindName[] = "logreg_pp";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs, const ArrayRef& mhs, 
                const ArrayRef& rhs, size_t M, size_t N,
                size_t K) const override {
    SPU_PROFILE_TRACE_KERNEL(ctx, lhs, mhs, rhs);

    const auto field = lhs.eltype().as<Ring2k>()->field();
    
    size_t i = 0, index;
    ArrayRef x_(makeType<RingTy>(field), N * M);
    for ( i = 0 ; i < lhs.elsize() ; i++){
      index = (i % N) * M + (i / N);
      x_.at<int32_t>(index) = lhs.at<int32_t>(i);
    }
    auto y_pred = ring_mmul(mhs, x_, 1, M, N);
    auto err = ring_sub(y_pred, rhs);
    auto grad = ring_mmul(err, lhs, 1, N, M);
    return grad.as(lhs.eltype());
  }
};

class Ref2kAndSS : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "and_ss";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_PROFILE_TRACE_KERNEL(ctx, lhs, rhs);
    YASL_ENFORCE(lhs.eltype() == rhs.eltype());
    return ring_and(lhs, rhs).as(lhs.eltype());
  }
};

class Ref2kAndSP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "and_sp";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_PROFILE_TRACE_KERNEL(ctx, lhs, rhs);
    return ring_and(lhs, rhs).as(lhs.eltype());
  }
};

class Ref2kXorSS : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "xor_ss";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_PROFILE_TRACE_KERNEL(ctx, lhs, rhs);
    YASL_ENFORCE(lhs.eltype() == rhs.eltype());
    return ring_xor(lhs, rhs).as(lhs.eltype());
  }
};

class Ref2kXorSP : public BinaryKernel {
 public:
  static constexpr char kBindName[] = "xor_sp";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& lhs,
                const ArrayRef& rhs) const override {
    SPU_PROFILE_TRACE_KERNEL(ctx, lhs, rhs);
    return ring_xor(lhs, rhs).as(lhs.eltype());
  }
};

class Ref2kLShiftS : public ShiftKernel {
 public:
  static constexpr char kBindName[] = "lshift_s";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override {
    SPU_PROFILE_TRACE_KERNEL(ctx, in, bits);
    return ring_lshift(in, bits).as(in.eltype());
  }
};

class Ref2kRShiftS : public ShiftKernel {
 public:
  static constexpr char kBindName[] = "rshift_s";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override {
    SPU_PROFILE_TRACE_KERNEL(ctx, in, bits);
    return ring_rshift(in, bits).as(in.eltype());
  }
};

class Ref2kBitrevS : public BitrevKernel {
 public:
  static constexpr char kBindName[] = "bitrev_s";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in, size_t start,
                size_t end) const override {
    const auto field = in.eltype().as<Ring2k>()->field();
    YASL_ENFORCE(start <= end);
    YASL_ENFORCE(end <= SizeOf(field) * 8);

    SPU_PROFILE_TRACE_KERNEL(ctx, in, start, end);
    return ring_bitrev(in, start, end).as(in.eltype());
  }
};

class Ref2kARShiftS : public ShiftKernel {
 public:
  static constexpr char kBindName[] = "arshift_s";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in,
                size_t bits) const override {
    SPU_PROFILE_TRACE_KERNEL(ctx, in, bits);
    return ring_arshift(in, bits).as(in.eltype());
  }
};

class Ref2kMsbS : public UnaryKernel {
 public:
  static constexpr char kBindName[] = "msb_s";

  util::CExpr latency() const override { return util::Const(0); }

  util::CExpr comm() const override { return util::Const(0); }

  ArrayRef proc(KernelEvalContext* ctx, const ArrayRef& in) const override {
    SPU_PROFILE_TRACE_KERNEL(ctx, in);
    return ring_rshift(in, in.elsize() * 8 - 1).as(in.eltype());
  }
};

}  // namespace

std::unique_ptr<Object> makeRef2kProtocol(
    const std::shared_ptr<yasl::link::Context>& lctx) {
  registerTypes();

  auto obj = std::make_unique<Object>();

  // register random states & kernels.
  obj->addState<PrgState>();

  // register public kernels.
  regPub2kKernels(obj.get());

  // register compute kernels
  obj->regKernel<Ref2kP2S>();
  obj->regKernel<Ref2kS2P>();
  obj->regKernel<Ref2kNotS>();
  obj->regKernel<Ref2kEqzS>();
  obj->regKernel<Ref2kAddSS>();
  obj->regKernel<Ref2kAddSP>();
  obj->regKernel<Ref2kMulSS>();
  obj->regKernel<Ref2kMulSP>();
  obj->regKernel<Ref2kMatMulSS>();
  obj->regKernel<Ref2kMatMulSP>();

  //lj
  obj->regKernel<Ref2kLogRegPP>();
  obj->regKernel<Ref2kLogRegSS>();

  obj->regKernel<Ref2kAndSS>();
  obj->regKernel<Ref2kAndSP>();
  obj->regKernel<Ref2kXorSS>();
  obj->regKernel<Ref2kXorSP>();
  obj->regKernel<Ref2kLShiftS>();
  obj->regKernel<Ref2kRShiftS>();
  obj->regKernel<Ref2kBitrevS>();
  obj->regKernel<Ref2kARShiftS>();
  obj->regKernel<Ref2kARShiftS>("truncpr_s");
  obj->regKernel<Ref2kMsbS>();

  return obj;
}

std::vector<ArrayRef> Ref2kIo::toShares(const ArrayRef& raw,
                                        Visibility vis) const {
  YASL_ENFORCE(raw.eltype().isa<RingTy>(), "expected RingTy, got {}",
               raw.eltype());
  const auto field = raw.eltype().as<Ring2k>()->field();
  YASL_ENFORCE(field == field_, "expect raw value encoded in field={}, got={}",
               field_, field);

  if (vis == VIS_PUBLIC) {
    const auto share = raw.as(makeType<Pub2kTy>(field));
    return std::vector<ArrayRef>(world_size_, share);
  }
  YASL_ENFORCE(vis == VIS_SECRET, "expected SECRET, got {}", vis);

  // directly view the data as secret.
  const auto share = raw.as(makeType<Ref2kSecrTy>(field));
  return std::vector<ArrayRef>(world_size_, share);
}

ArrayRef Ref2kIo::fromShares(const std::vector<ArrayRef>& shares) const {
  const auto field = shares.at(0).eltype().as<Ring2k>()->field();
  // no matter Public or Secret, directly view the first share as public.
  return shares[0].as(makeType<RingTy>(field));
}

std::unique_ptr<Ref2kIo> makeRef2kIo(FieldType field, size_t npc) {
  registerTypes();
  return std::make_unique<Ref2kIo>(field, npc);
}

}  // namespace spu::mpc
