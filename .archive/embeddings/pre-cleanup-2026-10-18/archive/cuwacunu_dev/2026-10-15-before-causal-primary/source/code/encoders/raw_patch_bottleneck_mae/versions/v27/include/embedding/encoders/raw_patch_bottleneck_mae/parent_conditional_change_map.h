// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#pragma once
#include "embedding/encoders/raw_patch_bottleneck_mae/parent_conditional_change_config.h"

namespace embedding::encoders::raw_patch_bottleneck_mae {
inline constexpr const char *kParentConditionalMapRecipe="parent-f32-f64-population-normalization;conditioned-bank-f32;coordinate-masked-f64-cholesky-ridge1-slopes-unpenalized-intercept-v1";
struct ParentConditionalBindings {
  std::string source_identity,input_scaler_identity,normalizer_identity,parent_identity;
};
struct ParentConditionalMap {
  ParentConditionalBindings bindings;
  std::string recipe_identity{kParentConditionalMapRecipe},bank_layout{kConditionedChangeBankLayout};
  std::vector<int64_t> input_geometry;
  double sampling_interval{0},floor{kConditionedChangeScaleFloor},ridge{1.};int64_t minimum_support{2};bool frozen{false};
  torch::Tensor parent_counts,parent_sum,parent_sumsq,parent_mean,parent_scale;
  torch::Tensor counts,support,gram,rhs,target_sumsq,coefficients,normal_equation_residual;
  void validate(const Config &c,const ParentConditionalBindings *expected=nullptr)const{
    const auto D=c.export_width,W=consecutive_change_bank_width(c),K=D+1;
    std::vector<int64_t> geometry{c.channel_count,c.history_length,c.input_width,D};const auto ids=resolved_channel_ids(c);geometry.insert(geometry.end(),ids.begin(),ids.end());
    TORCH_CHECK(frozen && input_geometry==geometry && sampling_interval==c.sampling_interval && floor==1e-6 && ridge==1. && minimum_support==2 &&
        recipe_identity==kParentConditionalMapRecipe && bank_layout==kConditionedChangeBankLayout,"[rpb-v27 map] frozen geometry/recipe mismatch");
    const auto text_ok=[](const std::string &s){return !s.empty() && s.size()<=4096 && s.find('\0')==std::string::npos;};
    TORCH_CHECK(text_ok(bindings.source_identity)&&text_ok(bindings.input_scaler_identity)&&text_ok(bindings.normalizer_identity)&&text_ok(bindings.parent_identity),"[rpb-v27 map] missing calibration bindings");
    if(expected)TORCH_CHECK(bindings.source_identity==expected->source_identity && bindings.input_scaler_identity==expected->input_scaler_identity &&
        bindings.normalizer_identity==expected->normalizer_identity && bindings.parent_identity==expected->parent_identity,"[rpb-v27 map] calibration identity mismatch");
    const auto tensor=[](const torch::Tensor &t,torch::ScalarType dtype,const std::vector<int64_t> &shape){TORCH_CHECK(t.defined() && t.device().is_cpu() && !t.requires_grad() &&
        t.scalar_type()==dtype && t.sizes().vec()==shape && (!t.is_floating_point()||torch::isfinite(t).all().item<bool>()),"[rpb-v27 map] finite CPU statistic contract mismatch");};
    tensor(parent_counts,torch::kInt64,{D});for(const auto &t:{parent_sum,parent_sumsq,parent_mean,parent_scale})tensor(t,torch::kFloat64,{D});
    tensor(counts,torch::kInt64,{W});tensor(support,torch::kBool,{W});tensor(gram,torch::kFloat64,{W,K,K});tensor(rhs,torch::kFloat64,{W,K});
    tensor(target_sumsq,torch::kFloat64,{W});tensor(coefficients,torch::kFloat64,{K,W});tensor(normal_equation_residual,torch::kFloat64,{W,K});
    TORCH_CHECK(parent_counts.ge(2).all().item<bool>() && parent_counts.eq(parent_counts[0]).all().item<bool>() && counts.ge(0).all().item<bool>() &&
        counts.le(parent_counts[0]).all().item<bool>() && parent_sumsq.ge(0).all().item<bool>() && target_sumsq.ge(0).all().item<bool>() &&
        torch::equal(support,counts.ge(2)),"[rpb-v27 map] count/support mismatch");
    const auto mean=parent_sum/parent_counts.to(torch::kFloat64),scale=(parent_sumsq/parent_counts.to(torch::kFloat64)-mean.square()).clamp_min(0).sqrt().clamp_min(floor);
    TORCH_CHECK(torch::equal(mean,parent_mean)&&torch::equal(scale,parent_scale),"[rpb-v27 map] parent population statistics mismatch");
    const auto active=support.unsqueeze(1),active3=active.unsqueeze(2);
    TORCH_CHECK(torch::where(active3,torch::zeros_like(gram),gram).eq(0).all().item<bool>() &&
        torch::where(active,torch::zeros_like(rhs),rhs).eq(0).all().item<bool>() &&
        torch::where(active,torch::zeros_like(coefficients.transpose(0,1)),coefficients.transpose(0,1)).eq(0).all().item<bool>() &&
        torch::where(support,torch::zeros_like(target_sumsq),target_sumsq).eq(0).all().item<bool>(),"[rpb-v27 map] unsupported coefficients/statistics must be zero");
    const auto symmetry_scale=torch::maximum(gram.abs(),gram.transpose(1,2).abs()).clamp_min(1.);
    TORCH_CHECK(((gram-gram.transpose(1,2)).abs()/symmetry_scale).max().item<double>()<=1e-12 && gram.diagonal(0,1,2).ge(0).all().item<bool>() &&
        torch::equal(gram.select(1,D).select(1,D),torch::where(support,counts,torch::zeros_like(counts)).to(torch::kFloat64)),"[rpb-v27 map] Gram/intercept count mismatch");
    const auto residual=equation_residual();TORCH_CHECK(torch::equal(residual,normal_equation_residual),"[rpb-v27 map] normal-equation witness mismatch");
    TORCH_CHECK((residual.abs()/(rhs.abs()+1.)).max().item<double>()<=1e-10,"[rpb-v27 map] inaccurate frozen normal equations");
  }
  torch::Tensor equation_residual()const{
    const auto D=coefficients.size(0)-1;
    auto penalty=torch::eye(D+1,torch::kFloat64)*ridge;penalty[D][D]=0.;
    const auto residual=torch::matmul(gram+penalty.unsqueeze(0),coefficients.transpose(0,1).unsqueeze(2)).squeeze(2)-rhs;
    return torch::where(support.unsqueeze(1),residual,torch::zeros_like(residual));
  }
  torch::Tensor residual(const torch::Tensor &parent,const torch::Tensor &sample_valid,const ConditionedChangeTargets &bank,const BankStandardizer &normalizer)const{
    const auto D=parent_mean.numel(),W=counts.numel();BankStandardizer::validate_targets(bank,W);
    TORCH_CHECK(frozen && normalizer.identity()==bindings.normalizer_identity && normalizer.input_scaler_identity==bindings.input_scaler_identity,"[rpb-v27 map] runtime normalization binding mismatch");
    TORCH_CHECK(parent.defined() && parent.dim()==2 && parent.size(1)==D && parent.size(0)==bank.values.size(0) && parent.scalar_type()==torch::kFloat32 && parent.device()==bank.values.device() &&
        sample_valid.defined() && sample_valid.sizes()==torch::IntArrayRef({parent.size(0)}) && sample_valid.scalar_type()==torch::kBool && sample_valid.device()==parent.device(),"[rpb-v27 map] runtime parent/support mismatch");
    const auto valid_parent=torch::where(sample_valid.unsqueeze(1),parent,torch::zeros_like(parent)).detach();
    TORCH_CHECK(torch::isfinite(valid_parent).all().item<bool>(),"[rpb-v27 map] finite sample-valid parent required");
    const auto options=parent.options().requires_grad(false);const auto E=(valid_parent-parent_mean.to(options))/parent_scale.to(options);
    const auto X=torch::cat({E,torch::ones({parent.size(0),1},options)},1),Q=X.matmul(coefficients.to(options));
    const auto eligible=bank.valid.logical_and(normalizer.support.to(parent.device()).unsqueeze(0)).logical_and(support.to(parent.device()).unsqueeze(0)).logical_and(sample_valid.unsqueeze(1));
    const auto B=normalizer.transform(bank),safe=torch::where(eligible,B,torch::zeros_like(B));
    const auto R=torch::where(eligible,safe-Q,torch::zeros_like(safe));
    TORCH_CHECK(torch::isfinite(R).all().item<bool>(),"[rpb-v27 map] finite conditional residual required");return R.detach();
  }
  ParentConditionalMap clone()const{
    auto out=*this;out.parent_counts=parent_counts.clone();out.parent_sum=parent_sum.clone();out.parent_sumsq=parent_sumsq.clone();out.parent_mean=parent_mean.clone();out.parent_scale=parent_scale.clone();
    out.counts=counts.clone();out.support=support.clone();out.gram=gram.clone();out.rhs=rhs.clone();out.target_sumsq=target_sumsq.clone();out.coefficients=coefficients.clone();out.normal_equation_residual=normal_equation_residual.clone();return out;
  }
  std::string identity()const{
    TORCH_CHECK(frozen,"[rpb-v27 map] frozen identity required");uint64_t hash=14695981039346656037ULL;
    const auto bytes=[&](const void *ptr,size_t count){const auto *v=static_cast<const uint8_t *>(ptr);for(size_t j=0;j<count;++j){hash^=v[j];hash*=1099511628211ULL;}};
    const auto text=[&](const std::string &s){const uint64_t n=s.size();bytes(&n,sizeof(n));bytes(s.data(),s.size());};
    for(const auto &s:{bindings.source_identity,bindings.input_scaler_identity,bindings.normalizer_identity,bindings.parent_identity,recipe_identity,bank_layout})text(s);
    bytes(input_geometry.data(),input_geometry.size()*sizeof(int64_t));bytes(&sampling_interval,sizeof(sampling_interval));bytes(&floor,sizeof(floor));bytes(&ridge,sizeof(ridge));bytes(&minimum_support,sizeof(minimum_support));
    for(const auto &t:{parent_counts,parent_sum,parent_sumsq,parent_mean,parent_scale,counts,support,gram,rhs,target_sumsq,coefficients,normal_equation_residual}){
      const auto cpu=t.detach().to(torch::kCPU).contiguous();const int64_t dtype=int64_t(cpu.scalar_type()),n=cpu.numel();bytes(&dtype,sizeof(dtype));bytes(&n,sizeof(n));bytes(cpu.const_data_ptr(),cpu.nbytes());
    }
    std::ostringstream out;out<<"rpb-parent-conditional-map-v1-fnv1a64-"<<std::hex<<std::setfill('0')<<std::setw(16)<<hash;return out.str();
  }
  void save(torch::serialize::OutputArchive &a)const{
    namespace d=configurable_checkpoint_detail;a.write("artifact_kind",d::text_tensor("rpb_parent_conditional_map_v1"),true);
    a.write("source_identity",d::text_tensor(bindings.source_identity),true);a.write("input_scaler_identity",d::text_tensor(bindings.input_scaler_identity),true);
    a.write("normalizer_identity",d::text_tensor(bindings.normalizer_identity),true);a.write("parent_identity",d::text_tensor(bindings.parent_identity),true);
    a.write("recipe_identity",d::text_tensor(recipe_identity),true);a.write("bank_layout",d::text_tensor(bank_layout),true);a.write("map_identity",d::text_tensor(identity()),true);
    a.write("input_geometry",torch::tensor(input_geometry,torch::kInt64),true);a.write("sampling_interval",torch::tensor(sampling_interval,torch::kFloat64),true);
    a.write("floor",torch::tensor(floor,torch::kFloat64),true);a.write("ridge",torch::tensor(ridge,torch::kFloat64),true);a.write("minimum_support",torch::tensor(minimum_support,torch::kInt64),true);
    a.write("parent_counts",parent_counts,true);a.write("parent_sum",parent_sum,true);a.write("parent_sumsq",parent_sumsq,true);a.write("parent_mean",parent_mean,true);a.write("parent_scale",parent_scale,true);
    a.write("counts",counts,true);a.write("support",support,true);a.write("gram",gram,true);a.write("rhs",rhs,true);a.write("target_sumsq",target_sumsq,true);a.write("coefficients",coefficients,true);a.write("normal_equation_residual",normal_equation_residual,true);
  }
  static ParentConditionalMap load(torch::serialize::InputArchive &a,const Config &c,const ParentConditionalBindings *expected=nullptr){
    namespace d=configurable_checkpoint_detail;TORCH_CHECK(d::read_text(a,"artifact_kind")=="rpb_parent_conditional_map_v1","[rpb-v27 map] typed map required");ParentConditionalMap out;
    out.bindings={d::read_text(a,"source_identity"),d::read_text(a,"input_scaler_identity"),d::read_text(a,"normalizer_identity"),d::read_text(a,"parent_identity")};
    out.recipe_identity=d::read_text(a,"recipe_identity");out.bank_layout=d::read_text(a,"bank_layout");const auto id=d::read_text(a,"map_identity");
    const auto g=d::read_vector(a,"input_geometry",torch::kInt64,4+c.channel_count);const auto *gp=g.const_data_ptr<int64_t>();out.input_geometry.assign(gp,gp+g.numel());
    const auto number=[&](const char *name){torch::Tensor t;a.read(name,t,true);TORCH_CHECK(t.dim()==0&&t.scalar_type()==torch::kFloat64,"[rpb-v27 map] F64 scalar required");return t.item<double>();};
    out.sampling_interval=number("sampling_interval");out.floor=number("floor");out.ridge=number("ridge");torch::Tensor n;a.read("minimum_support",n,true);TORCH_CHECK(n.dim()==0&&n.scalar_type()==torch::kInt64,"[rpb-v27 map] I64 minimum required");out.minimum_support=n.item<int64_t>();
    const auto read=[&](const char *name,torch::Tensor &t){a.read(name,t,true);t=t.to(torch::kCPU);};
    read("parent_counts",out.parent_counts);read("parent_sum",out.parent_sum);read("parent_sumsq",out.parent_sumsq);read("parent_mean",out.parent_mean);read("parent_scale",out.parent_scale);
    read("counts",out.counts);read("support",out.support);read("gram",out.gram);read("rhs",out.rhs);read("target_sumsq",out.target_sumsq);read("coefficients",out.coefficients);read("normal_equation_residual",out.normal_equation_residual);
    out.frozen=true;out.validate(c,expected);TORCH_CHECK(out.identity()==id,"[rpb-v27 map] content identity mismatch");return out;
  }
};
inline ParentConditionalMap fit_parent_conditional_map(const Config &c,const torch::Tensor &parent,const torch::Tensor &sample_valid,
    const ConditionedChangeTargets &bank,const BankStandardizer &normalizer,const ParentConditionalBindings &bindings){
  normalizer.validate(c,nullptr,&bindings.input_scaler_identity);TORCH_CHECK(normalizer.frozen&&normalizer.identity()==bindings.normalizer_identity,"[rpb-v27 map] exact frozen normalizer binding required");
  const auto D=c.export_width,W=consecutive_change_bank_width(c),K=D+1;BankStandardizer::validate_targets(bank,W);
  TORCH_CHECK(parent.defined()&&parent.dim()==2&&parent.size(1)==D&&parent.size(0)==bank.values.size(0)&&parent.scalar_type()==torch::kFloat32&&
      sample_valid.defined()&&sample_valid.sizes()==torch::IntArrayRef({parent.size(0)})&&sample_valid.scalar_type()==torch::kBool,"[rpb-v27 map] saved F32 parent/sample-valid schema required");
  torch::NoGradGuard guard;const auto valid=sample_valid.detach().to(torch::kCPU);const auto clean=torch::where(sample_valid.to(parent.device()).unsqueeze(1),parent,torch::zeros_like(parent)).detach().to(torch::kCPU,torch::kFloat64);
  TORCH_CHECK(torch::isfinite(clean).all().item<bool>()&&valid.sum().item<int64_t>()>=2,"[rpb-v27 map] at least2 finite sample-valid TRAIN rows required");
  ParentConditionalMap out;out.bindings=bindings;out.input_geometry={c.channel_count,c.history_length,c.input_width,D};const auto ids=resolved_channel_ids(c);out.input_geometry.insert(out.input_geometry.end(),ids.begin(),ids.end());out.sampling_interval=c.sampling_interval;
  out.parent_counts=torch::full({D},valid.sum().item<int64_t>(),torch::kInt64);out.parent_sum=clean.sum(0);out.parent_sumsq=clean.square().sum(0);out.parent_mean=out.parent_sum/out.parent_counts.to(torch::kFloat64);
  out.parent_scale=(out.parent_sumsq/out.parent_counts.to(torch::kFloat64)-out.parent_mean.square()).clamp_min(0).sqrt().clamp_min(1e-6);
  const auto E=(clean-out.parent_mean)/out.parent_scale,X=torch::cat({E,torch::ones({parent.size(0),1},torch::kFloat64)},1);
  const auto B=normalizer.transform(bank).to(torch::kCPU,torch::kFloat64),eligible=bank.valid.to(torch::kCPU).logical_and(normalizer.support.unsqueeze(0)).logical_and(valid.unsqueeze(1));
  out.counts=eligible.sum(0);out.support=out.counts.ge(2);out.gram=torch::zeros({W,K,K},torch::kFloat64);out.rhs=torch::zeros({W,K},torch::kFloat64);
  out.target_sumsq=torch::zeros({W},torch::kFloat64);out.coefficients=torch::zeros({K,W},torch::kFloat64);auto penalty=torch::eye(K,torch::kFloat64);penalty[D][D]=0.;
  for(int64_t j=0;j<W;++j)if(out.support[j].item<bool>()){
    const auto rows=eligible.select(1,j).nonzero().flatten(),A=X.index_select(0,rows),y=B.select(1,j).index_select(0,rows);
    const auto G=A.transpose(0,1).matmul(A),rhs=A.transpose(0,1).matmul(y),factor=torch::linalg_cholesky(G+penalty);
    out.gram[j].copy_(G);out.rhs[j].copy_(rhs);out.target_sumsq[j].copy_(y.square().sum());out.coefficients.select(1,j).copy_(torch::cholesky_solve(rhs.unsqueeze(1),factor).squeeze(1));
  }
  out.normal_equation_residual=out.equation_residual();out.frozen=true;out.validate(c,&bindings);return out;
}
} // namespace embedding::encoders::raw_patch_bottleneck_mae
