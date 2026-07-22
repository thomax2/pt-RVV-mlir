#ifndef EOT_OPS_TD
#define EOT_OPS_TD

include "EotDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class EotOp<string mnemonic, list<Trait> traits = []>
    : Op<EotDialect, mnemonic, traits>;

def PointFeaturesOp : EotOp<"point_features", [Pure]> {
  let arguments = (ins AnyRankedTensor:$points, AnyRankedTensor:$mask,
    DefaultValuedAttr<F64Attr, "1.0e-8">:$radius_epsilon,
    DefaultValuedAttr<F64Attr, "1.0e-10">:$theta_epsilon);
  let results = (outs AnyRankedTensor:$features);
  let assemblyFormat = "$points `,` $mask attr-dict `:` type($points) `,` type($mask) `->` type($features)";
  let hasVerifier = 1;
}

class EotMaskedReductionOp<string mnemonic> : EotOp<mnemonic, [Pure]> {
  let arguments = (ins AnyRankedTensor:$input, AnyRankedTensor:$mask,
    DefaultValuedAttr<I64Attr, "1">:$axis);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input `,` $mask attr-dict `:` type($input) `,` type($mask) `->` type($result)";
  let hasVerifier = 1;
}

def MaskedMeanOp : EotMaskedReductionOp<"masked_mean">;
def MaskedVarianceOp : EotMaskedReductionOp<"masked_variance">;

def MaskedMaxOp : EotOp<"masked_max", [Pure]> {
  let arguments = (ins AnyRankedTensor:$input, AnyRankedTensor:$mask,
    DefaultValuedAttr<I64Attr, "1">:$axis,
    DefaultValuedAttr<F64Attr, "-1.0e9">:$empty_value);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input `,` $mask attr-dict `:` type($input) `,` type($mask) `->` type($result)";
  let hasVerifier = 1;
}

def MaskedMinOp : EotOp<"masked_min", [Pure]> {
  let arguments = (ins AnyRankedTensor:$input, AnyRankedTensor:$mask,
    DefaultValuedAttr<I64Attr, "1">:$axis,
    DefaultValuedAttr<F64Attr, "1.0e9">:$empty_value);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input `,` $mask attr-dict `:` type($input) `,` type($mask) `->` type($result)";
  let hasVerifier = 1;
}

def GruCellOp : EotOp<"gru_cell", [Pure]> {
  let arguments = (ins AnyRankedTensor:$x, AnyRankedTensor:$h_prev,
    AnyRankedTensor:$weight_ih, AnyRankedTensor:$weight_hh,
    AnyRankedTensor:$bias_ih, AnyRankedTensor:$bias_hh,
    DefaultValuedAttr<StrAttr, "\"rzn\"">:$gate_order,
    DefaultValuedAttr<BoolAttr, "true">:$linear_before_reset);
  let results = (outs AnyRankedTensor:$h_next);
  let assemblyFormat = "$x `,` $h_prev `,` $weight_ih `,` $weight_hh `,` $bias_ih `,` $bias_hh attr-dict `:` type($x) `,` type($h_prev) `,` type($weight_ih) `,` type($weight_hh) `,` type($bias_ih) `,` type($bias_hh) `->` type($h_next)";
  let hasVerifier = 1;
}

def GmParameterizeOp : EotOp<"gm_parameterize", [Pure]> {
  let arguments = (ins AnyRankedTensor:$pi_logits, AnyRankedTensor:$tau_logits,
    AnyRankedTensor:$c_xi, AnyRankedTensor:$raw_logvar, AnyRankedTensor:$qN,
    DefaultValuedAttr<F64Attr, "0.003">:$c_min,
    DefaultValuedAttr<F64Attr, "0.70">:$c_max,
    DefaultValuedAttr<F64Attr, "1.0">:$tau_min,
    DefaultValuedAttr<F64Attr, "100.0">:$tau_max,
    DefaultValuedAttr<F64Attr, "1.0">:$alpha_base,
    DefaultValuedAttr<F64Attr, "-6.0">:$logvar_min,
    DefaultValuedAttr<F64Attr, "1.5">:$logvar_max);
  let results = (outs AnyRankedTensor:$pi, AnyRankedTensor:$tau,
    AnyRankedTensor:$alpha_pi, AnyRankedTensor:$alpha0,
    AnyRankedTensor:$s_pi, AnyRankedTensor:$c_mean,
    AnyRankedTensor:$c_logvar, AnyRankedTensor:$c_var,
    AnyRankedTensor:$q_c);
  let assemblyFormat = "$pi_logits `,` $tau_logits `,` $c_xi `,` $raw_logvar `,` $qN attr-dict `:` type($pi_logits) `,` type($tau_logits) `,` type($c_xi) `,` type($raw_logvar) `,` type($qN) `->` type($pi) `,` type($tau) `,` type($alpha_pi) `,` type($alpha0) `,` type($s_pi) `,` type($c_mean) `,` type($c_logvar) `,` type($c_var) `,` type($q_c)";
  let hasVerifier = 1;
}

#endif
