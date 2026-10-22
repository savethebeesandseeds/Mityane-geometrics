// Copyright (c) 2026 Waajacu. MIT; see LICENSE at repository root.
// CPB-v19 adds a training-only EMA teacher and learned-coordinate predictor.
#pragma once
#include "embedding/encoders/causal_patch_bottleneck_mae/family/training.h"
#include <cmath>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>

namespace embedding::encoders::causal_patch_bottleneck_mae::v19 {
namespace implementation=family::implementation;
namespace base_training=family::training;
using Config=family::Config;using Encoder=family::Encoder;using FrozenScaler=family::FrozenScaler;
using BankStandardizer=family::BankStandardizer;using RawHistory=family::RawHistory;using Encoding=family::Encoding;
using TrainingIdentity=base_training::TrainingIdentity;using TrainingProgress=base_training::TrainingProgress;
using OptimizerConfig=base_training::OptimizerConfig;using PreprocessingIdentity=base_training::PreprocessingIdentity;
using PreprocessingFitter=base_training::PreprocessingFitter;using FittedPreprocessing=base_training::FittedPreprocessing;
using family::read_settings;using family::dimensions;using family::parameter_counts;using family::validate_config;
using base_training::make_mask;using base_training::training_counter_seed;
inline constexpr const char *kModelTag="CPB-v19";
inline constexpr const char *kInferenceKind="cpb_v19_teacher_student_inference_v1";
inline constexpr const char *kTrainingKind="cpb_v19_teacher_student_training_v1";
inline constexpr const char *kRecipeKind="cpb-learned-shape-EMA-teacher-predictor-v1";
inline constexpr double kAuxiliaryWeight=.1,kTeacherDecay=.99,kNormFloor=1e-6;
inline constexpr int64_t kPredictorSeedOffset=4000;
// Constants remain fields so wrong-recipe typed loads can be deliberately refused.
struct Recipe {double auxiliary_weight{kAuxiliaryWeight},teacher_decay{kTeacherDecay},norm_floor{kNormFloor};};
inline void validate_recipe(const Recipe &r={}){
  TORCH_CHECK(r.auxiliary_weight==kAuxiliaryWeight&&r.teacher_decay==kTeacherDecay&&r.norm_floor==kNormFloor,
      "[cpb-v19] exact fixed auxiliary/EMA/normalization recipe required");
}
inline std::string recipe_identity(const Recipe &r={}){
  validate_recipe(r);return "CPB-v19;cpb-learned-shape-EMA-teacher-predictor-v1;GaussianNLL=1;temporal=0.1;lags=1,4,16;fixed-mean3;originalQ-both-endpoints;variance-input-gradient-stop;scattered-family-mask;auxiliary=0.1;EMA=mul-inplace0.99-then-add-inplace-alpha0.01;norm-floor=1e-6;F32-outputs-promote-F64-before-L2;normalization=sqrt-clamp-min-sum-squares-floor-squared;row-loss=2-minus-2-normalized-dot;eligible=NLL-and-student-valid-and-teacher-valid;learned-shape-prefix-only;predictor=Linear-D-2D-GELU-Linear-2D-D-bias;predictor-seed=explicit-nonnegative-bound;teacher-full-observed-same-cutoff-no-grad;EMA-learned-only-after-successful-AdamW;one-global-clip-student-and-predictor;teacher-never-optimizer;dropout0";
}

struct PredictorImpl:torch::nn::Module {
  explicit PredictorImpl(int64_t width):width_(width){
    TORCH_CHECK(width>0&&width<=std::numeric_limits<int64_t>::max()/2,"[cpb-v19] derived positive predictor width required");
    (void)family::shape_detail::multiply({4,family::shape_detail::add({family::shape_detail::multiply({4,width,width}),family::shape_detail::multiply({3,width})})});
    first=register_module("first",torch::nn::Linear(width,2*width));
    second=register_module("second",torch::nn::Linear(2*width,width));
  }
  torch::Tensor forward(const torch::Tensor &x){
    TORCH_CHECK(x.dim()==2&&x.size(1)==width_,"[cpb-v19] predictor takes configured learned coordinates only");
    return second(torch::gelu(first(x)));
  }
  int64_t width()const{return width_;}
  torch::nn::Linear first{nullptr},second{nullptr};
 private:int64_t width_;
};
TORCH_MODULE(Predictor);
inline std::string predictor_identity(const Predictor &p){
  TORCH_CHECK(p,"[cpb-v19] predictor required");uint64_t hash=14695981039346656037ULL;
  const auto bytes=[&](const void *pointer,size_t count){const auto *b=static_cast<const unsigned char*>(pointer);for(size_t i=0;i<count;++i){hash^=b[i];hash*=1099511628211ULL;}};
  for(const auto &v:p->named_parameters()){bytes(v.key().data(),v.key().size());const auto x=v.value().detach().to(torch::kCPU).contiguous();
    const auto dtype=int64_t(x.scalar_type());bytes(&dtype,sizeof(dtype));for(const auto size:x.sizes())bytes(&size,sizeof(size));bytes(x.const_data_ptr(),x.nbytes());}
  std::ostringstream out;out<<"cpb-v19-predictor-fnv1a64-"<<std::hex<<std::setw(16)<<std::setfill('0')<<hash;return out.str();
}
struct TrainingBundle {
  Config config;Encoder model{nullptr};FrozenScaler scaler;Encoder teacher{nullptr};Predictor predictor{nullptr};
  int64_t predictor_seed{0},teacher_updates{0};std::string initial_teacher_identity,initial_predictor_identity;
};
inline void validate_teacher(const TrainingBundle &b){
  TORCH_CHECK(b.model&&b.teacher&&b.predictor,"[cpb-v19] complete training bundle required");
  TORCH_CHECK(family::checkpoint_detail::d::same_config(b.config.level,b.model->config().level)&&b.config.change_width==b.model->config().change_width&&
      family::checkpoint_detail::d::same_config(b.config.level,b.teacher->config().level)&&b.config.change_width==b.teacher->config().change_width,
      "[cpb-v19] exact model/teacher/bundle configuration required");
  TORCH_CHECK(b.predictor_seed>=0,"[cpb-v19] bound nonnegative predictor seed required");
  TORCH_CHECK(b.teacher->initialization_seed()==b.model->initialization_seed()&&
      b.teacher->initial_state_identity()==b.model->initial_state_identity()&&b.initial_teacher_identity==b.model->initial_state_identity(),
      "[cpb-v19] teacher must originate from exact same fresh base initialization");
  TORCH_CHECK(b.teacher_updates==b.model->causal_updates()&&b.teacher->causal_updates()==b.teacher_updates,
      "[cpb-v19] one teacher update per completed optimizer update required");
  TORCH_CHECK(b.model->input_scaler_identity()==b.scaler.identity()&&b.teacher->input_scaler_identity()==b.scaler.identity()&&
      b.teacher->normalizer().identity()==b.model->normalizer().identity(),"[cpb-v19] shared immutable TRAIN preprocessing required");
  TORCH_CHECK(b.predictor->width()==dimensions(b.config).shape_width&&!b.initial_predictor_identity.empty(),"[cpb-v19] configured predictor and initial identity required");
  TORCH_CHECK(!b.teacher->is_training(),"[cpb-v19] teacher must stay in eval mode");
  for(const auto &p:b.teacher->named_parameters())TORCH_CHECK(!p.value().grad().defined(),"[cpb-v19] teacher gradients forbidden");
  const auto student=b.model->named_parameters(),teacher=b.teacher->named_parameters();
  TORCH_CHECK(student.size()==teacher.size(),"[cpb-v19] equal teacher/student parameter inventories required");
  for(const auto &p:student){const auto *t=teacher.find(p.key());TORCH_CHECK(t&&t->sizes()==p.value().sizes()&&t->device()==p.value().device()&&
      t->scalar_type()==p.value().scalar_type()&&t->requires_grad()==p.value().requires_grad()&&
      t->unsafeGetTensorImpl()!=p.value().unsafeGetTensorImpl(),"[cpb-v19] independent corresponding teacher parameters required");
    if(!p.value().requires_grad())TORCH_CHECK(family::boundary::same_bits(p.value(),*t),"[cpb-v19] fixed teacher priors must remain exact");}
  const auto sb=b.model->named_buffers(),tb=b.teacher->named_buffers();TORCH_CHECK(sb.size()==tb.size(),"[cpb-v19] equal buffer inventories required");
  for(const auto &p:sb)TORCH_CHECK(tb.contains(p.key())&&family::boundary::same_bits(p.value(),tb[p.key()]),"[cpb-v19] immutable teacher buffers must match");
  const auto &c=b.config.level;const auto D=dimensions(b.config).shape_width;
  const auto named=b.predictor->named_parameters();TORCH_CHECK(named.size()==4&&b.predictor->named_buffers().size()==0,"[cpb-v19] exact two-linear predictor inventory required");
  const std::vector<std::pair<std::string,std::vector<int64_t>>> shapes{{"first.weight",{2*D,D}},{"first.bias",{2*D}},{"second.weight",{D,2*D}},{"second.bias",{D}}};
  for(const auto &[name,shape]:shapes){const auto *p=named.find(name);TORCH_CHECK(p&&p->sizes().vec()==shape&&p->requires_grad()&&p->device()==c.device&&
      p->scalar_type()==c.dtype&&torch::isfinite(*p).all().item<bool>(),"[cpb-v19] finite configured trainable predictor tensor required: ",name);}
  TORCH_CHECK(named["first.weight"].unsafeGetTensorImpl()==b.predictor->first->weight.unsafeGetTensorImpl()&&named["first.bias"].unsafeGetTensorImpl()==b.predictor->first->bias.unsafeGetTensorImpl()&&
      named["second.weight"].unsafeGetTensorImpl()==b.predictor->second->weight.unsafeGetTensorImpl()&&named["second.bias"].unsafeGetTensorImpl()==b.predictor->second->bias.unsafeGetTensorImpl(),
      "[cpb-v19] registered predictor and live holders must agree");
}
inline TrainingBundle fresh(const Config &config,const FrozenScaler &scaler,const BankStandardizer &bank,int64_t seed,int64_t predictor_seed){
  TORCH_CHECK(seed>=0&&predictor_seed>=0,"[cpb-v19] explicit nonnegative encoder and predictor seeds required");
  auto student=family::fresh(config,scaler,bank,seed),teacher=family::fresh(config,scaler,bank,seed);
  TORCH_CHECK(student.model->state_identity()==teacher.model->state_identity(),"[cpb-v19] exact fresh teacher/student state required");
  const auto sp=student.model->named_parameters(),tp=teacher.model->named_parameters();
  for(const auto &p:sp)TORCH_CHECK(family::boundary::same_bits(p.value(),tp[p.key()]),"[cpb-v19] full initial teacher/student tensor parity required");
  torch::manual_seed(predictor_seed);auto predictor=Predictor(dimensions(config).shape_width);predictor->to(config.level.device,config.level.dtype);
  teacher.model->eval();TrainingBundle out{config,student.model,student.scaler,teacher.model,predictor,predictor_seed,0,
      teacher.model->state_identity(),predictor_identity(predictor)};validate_teacher(out);return out;
}
inline auto named_learned_parameters(const TrainingBundle &b){
  std::vector<std::pair<std::string,torch::Tensor>> out;
  for(const auto &p:b.model->named_parameters())if(p.value().requires_grad())out.emplace_back("student."+p.key(),p.value());
  for(const auto &p:b.predictor->named_parameters())out.emplace_back("predictor."+p.key(),p.value());return out;
}
inline std::vector<torch::Tensor> learned_parameters(const TrainingBundle &b){std::vector<torch::Tensor> out;for(const auto &p:named_learned_parameters(b))out.push_back(p.second);return out;}
inline std::string parameter_order(const TrainingBundle &b){std::string out;for(const auto &p:named_learned_parameters(b))out+=p.first+'\n';return out;}
inline std::unique_ptr<torch::optim::AdamW> make_optimizer(const TrainingBundle &b,const OptimizerConfig &options={}){
  validate_teacher(b);base_training::validate_optimizer_config(options);return std::make_unique<torch::optim::AdamW>(learned_parameters(b),base_training::adamw_options(options));
}
inline std::vector<int64_t> validate_optimizer(const TrainingBundle &b,const torch::optim::AdamW &optimizer,const TrainingProgress &progress,const OptimizerConfig &recipe={}){
  validate_teacher(b);base_training::validate_optimizer_config(recipe);base_training::common::validate_progress(progress);
  b.model->core()->validate_state();b.model->core()->validate_fixed_prior();b.teacher->core()->validate_state();b.teacher->core()->validate_fixed_prior();
  TORCH_CHECK(progress.completed_updates==b.model->causal_updates(),"[cpb-v19] completed live model count required");
  const auto *defaults=dynamic_cast<const torch::optim::AdamWOptions*>(&optimizer.defaults());
  TORCH_CHECK(defaults&&*defaults==base_training::adamw_options(recipe)&&optimizer.param_groups().size()==1,"[cpb-v19] configured single-group AdamW required");
  const auto params=learned_parameters(b);const auto &group=optimizer.param_groups().front();const auto *options=dynamic_cast<const torch::optim::AdamWOptions*>(&group.options());
  TORCH_CHECK(options&&*options==base_training::adamw_options(recipe)&&params.size()==group.params().size(),"[cpb-v19] optimizer student/predictor group mismatch");
  std::set<void*> known;std::vector<int64_t> steps;int64_t maximum_step=0;
  for(size_t i=0;i<params.size();++i){const auto &p=params[i];auto *key=p.unsafeGetTensorImpl();TORCH_CHECK(group.params()[i].unsafeGetTensorImpl()==key&&known.insert(key).second,"[cpb-v19] optimizer parameter order/alias mismatch");
    const auto found=optimizer.state().find(key);if(found==optimizer.state().end()){steps.push_back(-1);continue;}
    const auto *state=dynamic_cast<const torch::optim::AdamWParamState*>(found->second.get());
    TORCH_CHECK(state&&state->step()>0&&state->step()<=progress.completed_updates&&!state->max_exp_avg_sq().defined(),"[cpb-v19] optimizer step/flags mismatch");
    for(const auto &m:{state->exp_avg(),state->exp_avg_sq()})TORCH_CHECK(m.defined()&&m.device()==p.device()&&m.scalar_type()==p.scalar_type()&&m.sizes()==p.sizes()&&torch::isfinite(m).all().item<bool>(),"[cpb-v19] finite matching optimizer moments required");
    TORCH_CHECK(state->exp_avg_sq().ge(0).all().item<bool>(),"[cpb-v19] nonnegative second moments required");steps.push_back(state->step());maximum_step=std::max(maximum_step,state->step());}
  for(const auto &item:optimizer.state())TORCH_CHECK(known.count(item.first),"[cpb-v19] orphan/fixed/teacher optimizer state forbidden");
  TORCH_CHECK(maximum_step==progress.completed_updates&&(progress.completed_updates==0?optimizer.state().empty():!optimizer.state().empty()),"[cpb-v19] completed count and optimizer state disagree");return steps;
}
inline std::string optimizer_identity(const TrainingBundle &b,const torch::optim::AdamW &optimizer){
  uint64_t hash=14695981039346656037ULL;const auto bytes=[&](const void *p,size_t n){const auto *v=static_cast<const unsigned char*>(p);for(size_t i=0;i<n;++i){hash^=v[i];hash*=1099511628211ULL;}};
  for(const auto &[name,p]:named_learned_parameters(b)){bytes(name.data(),name.size());const auto found=optimizer.state().find(p.unsafeGetTensorImpl());const uint8_t present=found!=optimizer.state().end();bytes(&present,sizeof(present));if(!present)continue;
    const auto *state=dynamic_cast<const torch::optim::AdamWParamState*>(found->second.get());TORCH_CHECK(state,"[cpb-v19] AdamW moments required");const int64_t step=state->step();bytes(&step,sizeof(step));
    for(const auto &moment:{state->exp_avg(),state->exp_avg_sq()}){const auto cpu=moment.detach().to(torch::kCPU).contiguous();bytes(cpu.const_data_ptr(),cpu.nbytes());}}
  std::ostringstream out;out<<"cpb-v19-joint-optimizer-fnv1a64-"<<std::hex<<std::setw(16)<<std::setfill('0')<<hash;return out.str();
}

struct Dispersion {int64_t rows{0},near_zero_rows{0};bool supported{false};double norm_min{0},norm_max{0},std_mean{0},std_min{0},std_max{0};torch::Tensor coordinate_std;};
inline Dispersion dispersion(const torch::Tensor &values,const torch::Tensor &supported){
  TORCH_CHECK(values.dim()==2&&supported.dim()==1&&supported.size(0)==values.size(0)&&supported.scalar_type()==torch::kBool,"[cpb-v19] dispersion rows/support mismatch");
  const auto x=values.detach().to(torch::kFloat64).index({supported});Dispersion out;out.rows=x.size(0);out.supported=out.rows>0;
  out.coordinate_std=torch::zeros({values.size(1)},values.options().dtype(torch::kFloat64));if(!out.supported)return out;
  TORCH_CHECK(torch::isfinite(x).all().item<bool>(),"[cpb-v19] finite eligible diagnostic rows required");
  const auto norms=x.square().sum(1).sqrt();out.near_zero_rows=norms.le(kNormFloor).sum().item<int64_t>();out.norm_min=norms.min().item<double>();out.norm_max=norms.max().item<double>();
  out.coordinate_std=(x-x.mean(0)).square().mean(0).sqrt();out.std_mean=out.coordinate_std.mean().item<double>();out.std_min=out.coordinate_std.min().item<double>();out.std_max=out.coordinate_std.max().item<double>();return out;
}
struct AuxiliaryOutput {torch::Tensor loss,supported_rows;int64_t rows{0};};
inline AuxiliaryOutput auxiliary_loss(const torch::Tensor &prediction,const torch::Tensor &teacher,const torch::Tensor &support,const Recipe &recipe={}){
  validate_recipe(recipe);TORCH_CHECK(prediction.dim()==2&&prediction.size(1)>0&&prediction.is_cuda()&&prediction.scalar_type()==torch::kFloat32&&
      teacher.sizes()==prediction.sizes()&&teacher.device()==prediction.device()&&teacher.scalar_type()==torch::kFloat32&&support.scalar_type()==torch::kBool&&
      support.sizes()==prediction.sizes().slice(0,1)&&support.device()==prediction.device(),"[cpb-v19] matching F32 predictor/teacher and Boolean row support required");
  const auto mask=support.unsqueeze(-1);const auto p=torch::where(mask,prediction,torch::zeros_like(prediction)).to(torch::kFloat64);
  const auto t=torch::where(mask,teacher.detach(),torch::zeros_like(teacher)).to(torch::kFloat64);
  TORCH_CHECK(torch::isfinite(p).all().item<bool>()&&torch::isfinite(t).all().item<bool>(),"[cpb-v19] finite eligible F64 auxiliary coordinates required");
  // Clamp squared norms before sqrt: identical norm-floor values, finite
  // derivatives at an exactly zero eligible vector instead of sqrt'(0).
  const auto pn=p/p.square().sum(1,true).clamp_min(recipe.norm_floor*recipe.norm_floor).sqrt(),tn=t/t.square().sum(1,true).clamp_min(recipe.norm_floor*recipe.norm_floor).sqrt();
  const auto row=2.-2.*(pn*tn).sum(1);AuxiliaryOutput out;out.supported_rows=support;out.rows=support.sum().item<int64_t>();
  out.loss=out.rows?row.masked_select(support).mean():p.sum()*0.;TORCH_CHECK(torch::isfinite(out.loss).item<bool>(),"[cpb-v19] finite auxiliary objective required");return out;
}
struct ForwardTrace {implementation::LocationScaleForwardTrace base;AuxiliaryOutput auxiliary;torch::Tensor student_learned,teacher_learned,prediction,total_loss;int64_t teacher_forwards{0};};
inline ForwardTrace forward_at(TrainingBundle &b,const RawHistory &raw,const torch::Tensor &cutoffs,const torch::Tensor &hidden,const Recipe &recipe={}){
  validate_recipe(recipe);validate_teacher(b);ForwardTrace out;out.base=b.model->forward_at(raw,cutoffs,hidden,b.scaler);
  const auto D=dimensions(b.config).shape_width;out.student_learned=out.base.output.encoding.z_contextual_global.narrow(1,0,D);out.prediction=b.predictor(out.student_learned);
  Encoding target;{torch::NoGradGuard guard;target=b.teacher->encode_at(raw,cutoffs,b.scaler);}out.teacher_forwards=1;out.teacher_learned=target.native.narrow(1,0,D).detach();
  const auto support=out.base.output.eligible_examples&out.base.output.encoding.sample_valid_mask&target.valid;
  out.auxiliary=auxiliary_loss(out.prediction,out.teacher_learned,support,recipe);out.total_loss=out.base.output.loss+recipe.auxiliary_weight*out.auxiliary.loss;
  TORCH_CHECK(torch::isfinite(out.total_loss).item<bool>(),"[cpb-v19] finite combined objective required");return out;
}
inline void update_teacher(TrainingBundle &b,const Recipe &recipe={}){
  validate_recipe(recipe);TORCH_CHECK(b.model->causal_updates()==b.teacher_updates+1,"[cpb-v19] teacher EMA must follow exactly one successful student update");
  torch::NoGradGuard guard;const auto student=b.model->named_parameters(),teacher=b.teacher->named_parameters();
  for(const auto &p:student){auto t=teacher[p.key()];if(p.value().requires_grad())t.mul_(recipe.teacher_decay).add_(p.value(),.01);
    else TORCH_CHECK(family::boundary::same_bits(t,p.value()),"[cpb-v19] fixed priors cannot be EMA averaged");}
  ++b.teacher_updates;b.teacher->set_causal_updates(b.teacher_updates);b.teacher->eval();validate_teacher(b);
}
struct StepResult:base_training::StepResult {double auxiliary_loss{0};int64_t auxiliary_rows{0},teacher_updates{0},teacher_forwards{0};Dispersion student_dispersion,teacher_dispersion,predictor_dispersion;};
inline StepResult train_step(TrainingBundle &b,torch::optim::AdamW &optimizer,TrainingProgress &progress,const RawHistory &raw,const torch::Tensor &cutoffs,
    const Recipe &recipe={},const OptimizerConfig &options={}){
  validate_recipe(recipe);TORCH_CHECK(torch::GradMode::is_enabled(),"[cpb-v19] gradient-enabled training required");validate_optimizer(b,optimizer,progress,options);
  TORCH_CHECK(progress.attempted_batches<std::numeric_limits<int64_t>::max()-1,"[cpb-v19] incrementable attempt count required");
  const auto masks=make_mask(raw,cutoffs,b.config,progress.mask_seed,progress.attempted_batches);StepResult result;
  result.attempt_index=progress.attempted_batches;result.completed_updates=progress.completed_updates;result.teacher_updates=b.teacher_updates;
  if(!masks.eligible_channels.any().item<bool>()){optimizer.zero_grad(true);b.model->clear_gradients();b.predictor->zero_grad(true);++progress.attempted_batches;return result;}
  const auto bank=b.model->normalizer().identity();optimizer.zero_grad(true);b.model->train();b.predictor->train();const auto trace=forward_at(b,raw,cutoffs,masks.hidden,recipe);
  result.gaussian_nll=trace.base.nll.loss.item<double>();result.temporal_loss=trace.base.objective.temporal.loss.item<double>();result.auxiliary_loss=trace.auxiliary.loss.item<double>();
  result.auxiliary_rows=trace.auxiliary.rows;result.teacher_forwards=trace.teacher_forwards;result.total_loss=trace.total_loss.item<double>();
  result.eligible_examples=trace.base.nll.eligible_example_count;result.target_cells=trace.base.nll.target_cell_count;
  result.student_dispersion=dispersion(trace.student_learned,trace.auxiliary.supported_rows);result.teacher_dispersion=dispersion(trace.teacher_learned,trace.auxiliary.supported_rows);result.predictor_dispersion=dispersion(trace.prediction,trace.auxiliary.supported_rows);
  trace.total_loss.backward();for(const auto &[name,p]:named_learned_parameters(b)){const auto g=p.grad();if(!g.defined())continue;TORCH_CHECK(g.is_cuda()&&torch::isfinite(g).all().item<bool>(),"[cpb-v19] finite learned gradient required: ",name);++result.gradient_tensors;}
  for(const auto &p:b.teacher->named_parameters())TORCH_CHECK(!p.value().grad().defined(),"[cpb-v19] teacher must receive no gradients");
  TORCH_CHECK(result.gradient_tensors>0,"[cpb-v19] objective must reach learned parameters");result.gradient_norm=torch::nn::utils::clip_grad_norm_(learned_parameters(b),options.clip_norm);
  TORCH_CHECK(std::isfinite(result.gradient_norm),"[cpb-v19] finite joint gradient norm required");optimizer.step();++progress.completed_updates;++progress.attempted_batches;b.model->set_causal_updates(progress.completed_updates);
  optimizer.zero_grad(true);b.model->clear_gradients();b.predictor->zero_grad(true);update_teacher(b,recipe);
  TORCH_CHECK(b.model->normalizer().identity()==bank&&b.scaler.identity()==b.model->input_scaler_identity(),"[cpb-v19] frozen preprocessing changed");validate_optimizer(b,optimizer,progress,options);
  result.updated=true;result.completed_updates=progress.completed_updates;result.teacher_updates=b.teacher_updates;return result;
}

namespace artifact_detail {
namespace d=family::checkpoint_detail::d;
inline void write_recipe(torch::serialize::OutputArchive &a,const Recipe &recipe,const char *kind){
  validate_recipe(recipe);a.write("artifact_kind",d::text_tensor(kind),true);a.write("model_tag",d::text_tensor(kModelTag),true);a.write("recipe_kind",d::text_tensor(kRecipeKind),true);a.write("recipe_binding",d::text_tensor(recipe_identity(recipe)),true);
  a.write("recipe_f64",torch::tensor(std::vector<double>{recipe.auxiliary_weight,recipe.teacher_decay,recipe.norm_floor},torch::kFloat64),true);
  a.write("structural_codec",d::text_tensor("cpb-family-state-container-v1;outer-v19-learning-recipe-is-authoritative"),true);
}
inline void read_recipe(torch::serialize::InputArchive &a,const Recipe &recipe,const char *kind){
  validate_recipe(recipe);TORCH_CHECK(d::read_text(a,"artifact_kind")==kind&&d::read_text(a,"model_tag")==kModelTag&&d::read_text(a,"recipe_kind")==kRecipeKind&&
      d::read_text(a,"recipe_binding")==recipe_identity(recipe)&&d::read_text(a,"structural_codec")=="cpb-family-state-container-v1;outer-v19-learning-recipe-is-authoritative"&&
      family::boundary::same_bits(d::read_vector(a,"recipe_f64",torch::kFloat64,3),torch::tensor(std::vector<double>{recipe.auxiliary_weight,recipe.teacher_decay,recipe.norm_floor},torch::kFloat64)),"[cpb-v19] exact typed artifact/recipe/codec required");
}
inline std::string progress_binding(const TrainingIdentity &i,const TrainingProgress &p,const TrainingBundle &b,const Recipe &r){return recipe_identity(r)+":"+base_training::progress_binding(i,p)+":"+std::to_string(b.predictor_seed)+":"+std::to_string(b.teacher_updates)+":"+b.initial_teacher_identity+":"+b.initial_predictor_identity;}
}
struct Loaded {family::Loaded numerical;Recipe recipe;};
inline void save_checkpoint(const std::string &path,const Encoder &model,const FrozenScaler &scaler,int64_t updates,const Recipe &recipe={}){
  TORCH_CHECK(model,"[cpb-v19] student required");model->validate_frozen();torch::serialize::OutputArchive a,e;artifact_detail::write_recipe(a,recipe,kInferenceKind);
  family::checkpoint_detail::write(e,model,scaler,updates);a.write("structural_encoder",e);artifact_detail::d::publish_checkpoint(a,path);
}
inline Loaded load_checkpoint(const std::string &path,torch::Device device=torch::Device(torch::kCUDA,0),const Config *expected=nullptr,const Recipe &recipe={}){
  torch::serialize::InputArchive a,e;a.load_from(path,device);artifact_detail::read_recipe(a,recipe,kInferenceKind);a.read("structural_encoder",e);return{family::checkpoint_detail::read(e,device,expected),recipe};
}
struct LoadedTrainingState {TrainingBundle bundle;std::unique_ptr<torch::optim::AdamW> optimizer;TrainingIdentity identity;TrainingProgress progress;OptimizerConfig optimizer_config;Recipe recipe;};
inline void save_training_state(const std::string &path,const TrainingBundle &b,const torch::optim::AdamW &optimizer,const TrainingIdentity &identity,const TrainingProgress &progress,const Recipe &recipe={},const OptimizerConfig &options={}){
  validate_recipe(recipe);base_training::common::validate_identity(identity);const auto steps=validate_optimizer(b,optimizer,progress,options);b.teacher->validate_frozen();
  for(const auto &p:b.model->parameters())TORCH_CHECK(!p.grad().defined(),"[cpb-v19] clear student gradients before training checkpoint");
  for(const auto &p:b.predictor->parameters())TORCH_CHECK(!p.grad().defined(),"[cpb-v19] clear predictor gradients before checkpoint");
  torch::serialize::OutputArchive a,e,t,p,o;artifact_detail::write_recipe(a,recipe,kTrainingKind);const auto text=[&](const char *key,const std::string &value){a.write(key,artifact_detail::d::text_tensor(value),true);};
  text("counter_contract",implementation::consecutive_change_training::kChangeTrainingCounterContract);text("input_source_identity",identity.input_source_identity);text("split_identity",identity.split_identity);text("recipe_identity",identity.recipe_identity);
  text("progress_binding",artifact_detail::progress_binding(identity,progress,b,recipe));text("parameter_order",parameter_order(b));text("optimizer_parameter_keys",base_training::common::optimizer_parameter_keys(optimizer));text("optimizer_state_identity",optimizer_identity(b,optimizer));
  text("initial_teacher_identity",b.initial_teacher_identity);text("initial_predictor_identity",b.initial_predictor_identity);text("teacher_state_identity",b.teacher->state_identity());text("predictor_state_identity",predictor_identity(b.predictor));
  a.write("optimizer_config_f64",torch::tensor(base_training::optimizer_config_values(options),torch::kFloat64),true);a.write("progress_v1",torch::tensor(std::vector<int64_t>{progress.completed_updates,progress.attempted_batches,progress.row_seed,progress.mask_seed,progress.context_seed},torch::kInt64),true);
  a.write("teacher_predictor_v1",torch::tensor(std::vector<int64_t>{b.teacher_updates,b.predictor_seed,b.predictor->width()},torch::kInt64),true);a.write("optimizer_steps_v1",torch::tensor(steps,torch::kInt64),true);
  family::checkpoint_detail::write(e,b.model,b.scaler,progress.completed_updates);family::checkpoint_detail::write(t,b.teacher,b.scaler,progress.completed_updates);b.predictor->save(p);optimizer.save(o);
  a.write("structural_encoder",e);a.write("structural_teacher",t);a.write("predictor",p);a.write("optimizer",o);artifact_detail::d::publish_checkpoint(a,path);
}
inline LoadedTrainingState load_training_state(const std::string &path,const Config &expected,const TrainingIdentity &identity,const Recipe &recipe={},const OptimizerConfig &options={}){
  validate_recipe(recipe);family::validate_config(expected);base_training::common::validate_identity(identity);base_training::validate_optimizer_config(options);torch::serialize::InputArchive a,e,t,p,o;a.load_from(path,expected.level.device);
  artifact_detail::read_recipe(a,recipe,kTrainingKind);TORCH_CHECK(artifact_detail::d::read_text(a,"counter_contract")==implementation::consecutive_change_training::kChangeTrainingCounterContract&&
      family::boundary::same_bits(artifact_detail::d::read_vector(a,"optimizer_config_f64",torch::kFloat64,3),torch::tensor(base_training::optimizer_config_values(options),torch::kFloat64)),"[cpb-v19] counter/optimizer recipe mismatch");
  TrainingIdentity actual{artifact_detail::d::read_text(a,"input_source_identity"),artifact_detail::d::read_text(a,"split_identity"),artifact_detail::d::read_text(a,"recipe_identity")};base_training::common::validate_identity(actual);TORCH_CHECK(base_training::common::same_identity(actual,identity),"[cpb-v19] source/split/application recipe mismatch");
  const auto values=artifact_detail::d::read_vector(a,"progress_v1",torch::kInt64,5),extra=artifact_detail::d::read_vector(a,"teacher_predictor_v1",torch::kInt64,3);const auto v=values.accessor<int64_t,1>(),x=extra.accessor<int64_t,1>();
  TrainingProgress progress{v[0],v[1],v[2],v[3],v[4]};base_training::common::validate_progress(progress);TORCH_CHECK(x[0]==progress.completed_updates&&x[2]==dimensions(expected).shape_width,"[cpb-v19] EMA count/configured predictor width mismatch");
  a.read("structural_encoder",e);a.read("structural_teacher",t);auto student=family::checkpoint_detail::read(e,expected.level.device,&expected),teacher=family::checkpoint_detail::read(t,expected.level.device,&expected);
  TORCH_CHECK(student.causal_updates==progress.completed_updates&&teacher.causal_updates==progress.completed_updates&&student.scaler.identity()==teacher.scaler.identity(),"[cpb-v19] structural model/progress/scaler mismatch");
  torch::manual_seed(x[1]);auto predictor=Predictor(x[2]);predictor->to(expected.level.device,expected.level.dtype);const auto initial_predictor=predictor_identity(predictor);
  TORCH_CHECK(artifact_detail::d::read_text(a,"initial_predictor_identity")==initial_predictor,"[cpb-v19] predictor seed/initial-state mismatch");a.read("predictor",p);predictor->load(p);
  TrainingBundle b{expected,student.model,student.scaler,teacher.model,predictor,x[1],x[0],artifact_detail::d::read_text(a,"initial_teacher_identity"),initial_predictor};b.teacher->eval();validate_teacher(b);
  TORCH_CHECK(artifact_detail::d::read_text(a,"progress_binding")==artifact_detail::progress_binding(identity,progress,b,recipe)&&artifact_detail::d::read_text(a,"parameter_order")==parameter_order(b)&&
      artifact_detail::d::read_text(a,"teacher_state_identity")==b.teacher->state_identity()&&artifact_detail::d::read_text(a,"predictor_state_identity")==predictor_identity(b.predictor),"[cpb-v19] progress/seed/order/teacher/predictor state binding mismatch");
  auto optimizer=make_optimizer(b,options);a.read("optimizer",o);base_training::common::validate_optimizer_archive(o,artifact_detail::d::read_text(a,"optimizer_parameter_keys"),learned_parameters(b).size());optimizer->load(o);
  const auto steps=validate_optimizer(b,*optimizer,progress,options);TORCH_CHECK(artifact_detail::d::read_text(a,"optimizer_state_identity")==optimizer_identity(b,*optimizer)&&
      torch::equal(artifact_detail::d::read_vector(a,"optimizer_steps_v1",torch::kInt64,steps.size()),torch::tensor(steps,torch::kInt64)),"[cpb-v19] optimizer content/order/step binding mismatch");
  b.model->eval();b.model->clear_gradients();b.predictor->zero_grad(true);return{b,std::move(optimizer),actual,progress,options,recipe};
}
} // namespace embedding::encoders::causal_patch_bottleneck_mae::v19
