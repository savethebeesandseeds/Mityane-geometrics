// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
// Artificial compatibility witnesses; no market fitting or quality claims.
#include "cuwacunu/artifacts/v1/tensor_state.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <unistd.h>

namespace av=cuwacunu::artifacts::v1;
namespace fs=std::filesystem;
namespace {
void check(bool condition,const std::string& message) {if(!condition)throw std::runtime_error("[artifact v1 test] "+message);}
template<class F> void rejects(F action,const char* message) {
  bool rejected=false;try{action();}catch(const std::exception&){rejected=true;}check(rejected,message);
}
struct TemporaryDirectory {
  fs::path path;
  TemporaryDirectory(){auto text=(fs::temp_directory_path()/"artifact-v1-XXXXXX").string();std::vector<char> bytes(text.begin(),text.end());bytes.push_back('\0');const auto result=::mkdtemp(bytes.data());check(result,"private fixture directory");path=result;}
  ~TemporaryDirectory(){std::error_code ignored;fs::remove_all(path,ignored);}
};
void write(const fs::path& path,std::string_view text) {fs::create_directories(path.parent_path());std::ofstream out(path,std::ios::binary);out.write(text.data(),text.size());out.close();check(bool(out),"fixture write");}
std::string replace(std::string text,const std::string& before,const std::string& after) {
  const auto pos=text.find(before);check(pos!=std::string::npos,"replacement fixture present");text.replace(pos,before.size(),after);return text;
}
av::Component component(std::string_view name,const std::string& state,std::vector<av::ParentRef> parents={}) {
  return av::make_component({std::string(name),"v1",av::sha256_bytes(std::string(name)+" source")},state,av::sha256_bytes(std::string(name)+" config"),std::move(parents));
}
void hash_witnesses() {
  check(av::sha256_bytes("")=="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","empty SHA256 golden");
  check(av::sha256_bytes("abc")=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad","abc SHA256 golden");
  check(av::sha256_bytes(std::string(1000000,'a'))=="cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0","multiblock SHA256 golden");
  av::Sha256 streamed;streamed.update("a");streamed.update("bc");check(streamed.finish()==av::sha256_bytes("abc"),"streamed SHA256 matches bytes");
  check(av::digest_fields("domain",{"ab","c"})!=av::digest_fields("domain",{"a","bc"}),"length delimiting avoids ambiguity");
  check(av::digest_fields("one",{"same"})!=av::digest_fields("two",{"same"}),"record domains are distinct");
  TemporaryDirectory temp;write(temp.path/"archive.pt",std::string("a\0b",3));check(av::sha256_file(temp.path/"archive.pt")==av::sha256_bytes(std::string("a\0b",3)),"raw file includes zero bytes");
  write(temp.path/"one"/"state.h","first source");write(temp.path/"two"/"state.h","first source");
  check(av::source_closure_sha256(temp.path/"one",{"state.h"})==av::source_closure_sha256(temp.path/"two",{"state.h"}),"source digest independent of checkout path");
  write(temp.path/"two"/"state.h","edited source");
  check(av::source_closure_sha256(temp.path/"one",{"state.h"})!=av::source_closure_sha256(temp.path/"two",{"state.h"}),"local source edit changes identity");
  rejects([&]{av::source_closure_sha256(temp.path/"one",{"../archive.pt"});},"source closure cannot escape root");
  rejects([&]{av::source_closure_sha256(temp.path/"one",{"state.h","state.h"});},"source closure rejects repeated members");
}
void tensor_witnesses() {
  const auto options=torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
  const auto values=torch::arange(6,options).reshape({2,3});
  const auto first=av::tensor_state_sha256({{"weights",values},{"frozen_scaler",torch::ones({3},options)}});
  check(first==av::tensor_state_sha256({{"frozen_scaler",torch::ones({3},options)},{"weights",values.clone()}}),"named tensor order and allocation do not change state");
  check(av::tensor_state_sha256({{"weights",values.transpose(0,1)}})==av::tensor_state_sha256({{"weights",values.transpose(0,1).contiguous()}}),"stride/storage do not affect contiguous state bytes");
  check(av::tensor_state_sha256({{"weights",values}})!=av::tensor_state_sha256({{"weights",values.reshape({3,2})}}),"shape belongs to state identity");
  check(av::tensor_state_sha256({{"weights",values}})!=av::tensor_state_sha256({{"weights",values.to(torch::kFloat32)}}),"dtype belongs to state identity");
  auto changed=values.clone();changed[0][0]=std::nextafter(0.,1.);
  check(av::tensor_state_sha256({{"weights",values}})!=av::tensor_state_sha256({{"weights",changed}}),"state digest never rounds away exact changes");
  rejects([&]{av::tensor_state_sha256({{"weights",values},{"weights",values}});},"duplicate tensor names reject");
  torch::nn::Linear module(torch::nn::LinearOptions(3,2));module->register_buffer("frozen_scale",torch::ones({2}));
  const auto before=av::module_state_sha256(*module);
  module->named_buffers()["frozen_scale"][0]=2;
  check(av::module_state_sha256(*module)!=before,"registered frozen buffers influence state");
  const auto explicit_before=av::module_state_sha256(*module,{{"preprocessing",torch::ones({2})}});
  check(explicit_before!=av::module_state_sha256(*module,{{"preprocessing",torch::zeros({2})}}),"adapter frozen non-parameter state influences state");
}
void lineage_witnesses() {
  const auto state=av::sha256_bytes("weights");
  const auto encoder=component(av::encoder_component,state);
  const auto representation=component(av::state_representation_component,state,{{"embedding",std::string(av::encoder_component),encoder.model_ref}});
  const auto target=component(av::target_state_representation_component,state,{{"embedding",std::string(av::encoder_component),encoder.model_ref}});
  const auto dynamics=component(av::dynamics_component,state,{{"prediction_target",std::string(av::target_state_representation_component),target.model_ref},{"input_history",std::string(av::state_representation_component),representation.model_ref}});
  const std::vector<av::Component> selected{dynamics,encoder,target,representation};av::validate_parents(selected);
  check(representation.model_ref!=target.model_ref,"identical parameters do not erase representation role");
  auto stale=selected;stale[1]=component(av::encoder_component,av::sha256_bytes("changed weights"));
  rejects([&]{av::validate_parents(stale);},"Encoder weight change invalidates downstream parent bindings");
  stale=selected;auto changed_module=encoder.module_ref;changed_module.source_sha256=av::sha256_bytes("changed implementation");stale[1]=av::make_component(changed_module,encoder.state_sha256,encoder.inference_config_sha256,{});
  rejects([&]{av::validate_parents(stale);},"Encoder implementation change invalidates downstream parent bindings");
  stale=selected;stale[1]=av::make_component(encoder.module_ref,encoder.state_sha256,av::sha256_bytes("changed config"),{});
  rejects([&]{av::validate_parents(stale);},"Encoder mapping change invalidates downstream parent bindings");
  auto swapped=dynamics;std::swap(swapped.parents[0].model_ref,swapped.parents[1].model_ref);swapped.model_ref=av::derive_model_ref(swapped);
  rejects([&]{av::validate_parents({encoder,representation,target,swapped});},"same-width online/target swap rejects");
  rejects([&]{av::validate_parents({representation,target,dynamics});},"missing parent rejects");
  rejects([&]{component(av::portfolio_policy_component,state);},"reserved Portfolio Policy cannot be selected");
  rejects([&]{component(av::dynamics_component,state,{{"input_history",std::string(av::state_representation_component),representation.model_ref}});},"missing Dynamics Model target rejects");
  rejects([&]{av::verify_module_ref(encoder,{std::string(av::encoder_component),"v2",encoder.module_ref.source_sha256});},"unsupported implementation rejects");
  auto only_dynamics=selected;only_dynamics[0]=component(av::dynamics_component,av::sha256_bytes("new Dynamics Model weights"),dynamics.parents);av::validate_parents(only_dynamics);
  check(only_dynamics[2]==target&&only_dynamics[3]==representation,"Dynamics Model-only weights preserve upstream snapshots");
  const auto sidecar=av::sha256_bytes("pinned sidecar");
  const auto bundle=av::bundle_ref({{sidecar,encoder.module_ref.component,encoder.model_ref},{sidecar,dynamics.module_ref.component,dynamics.model_ref}});
  check(bundle==av::bundle_ref({{sidecar,dynamics.module_ref.component,dynamics.model_ref},{sidecar,encoder.module_ref.component,encoder.model_ref}}),"bundle does not depend on selection enumeration order");
  check(bundle!=av::bundle_ref({{sidecar,encoder.module_ref.component,encoder.model_ref},{av::sha256_bytes("repacked"),dynamics.module_ref.component,dynamics.model_ref}}),"repackaging changes bundle pin");
}
void archive_witnesses() {
  TemporaryDirectory temp;const auto path=temp.path/"legacy.pt";
  const std::string bytes("original\0archive\n",17);write(path,bytes);
  const auto raw=av::sha256_file(path);const auto encoder=component(av::encoder_component,av::sha256_bytes("state"));
  const av::Manifest manifest{"inference",raw,{}, {encoder}};
  const auto serialized=av::serialize_manifest(manifest);check(av::parse_manifest(serialized)==manifest,"sidecar roundtrip preserves exact refs");
  const auto pin=av::publish_sidecar(path,manifest);
  check(av::sha256_file(path)==raw,"sidecar adoption preserves legacy archive bytes");
  const auto admitted=av::read_verified(path,pin,"inference");check(admitted.bytes==bytes&&admitted.manifest==manifest,"pinned bytes and sidecar admit together");
  rejects([&]{av::publish_sidecar(path,manifest);},"published sidecar cannot be overwritten");
  check(av::sha256_file(av::sidecar_path(path))==pin,"failed publication preserves original sidecar");
  rejects([&]{av::read_verified(path,av::sha256_bytes("wrong pin"),"inference");},"wrong trusted sidecar pin rejects");
  rejects([&]{av::read_verified(path,"","inference");},"absent pin rejects");
  rejects([&]{av::read_verified(path,pin,"training");},"inference cannot be exact-resumed as training");
  write(path,"changed archive");
  rejects([&]{av::read_verified(path,pin,"inference");},"raw archive corruption rejects");
  check(admitted.bytes==bytes,"already admitted memory remains the exact verified bytes after path mutation");
  rejects([&]{av::parse_manifest(replace(serialized,"\"schema\":","\"schema\":\"duplicate\",\"schema\":"));},"duplicate JSON field rejects");
  rejects([&]{av::parse_manifest(replace(serialized,"\"schema\":","\"unknown\":\"field\",\"schema\":"));},"unknown JSON field rejects");
  rejects([&]{av::parse_manifest(replace(serialized,"model_artifact_sidecar_v1","model_artifact_sidecar_v2"));},"unknown schema rejects");
  rejects([&]{av::parse_manifest(replace(serialized,encoder.model_ref,av::sha256_bytes("forged model")));},"forged derived ModelRef rejects");
  rejects([&]{av::parse_manifest(serialized+"{}");},"trailing JSON rejects");
  rejects([&]{av::parse_manifest(serialized.substr(0,serialized.size()-3));},"truncated JSON rejects");
  rejects([&]{av::parse_manifest(replace(serialized,"\"archive_kind\":\"inference\"","\"archive_kind\":\"inference\",\"resume_identity_sha256\":\"\""));},"inference cannot contain resume identity");
  rejects([&]{av::parse_manifest(std::string(262145,' '));},"oversized sidecar rejects");
  write(temp.path/"training.pt","training archive");
  av::Manifest training{"training",av::sha256_file(temp.path/"training.pt"),av::sha256_bytes("strict training identity"),{encoder}};
  const auto training_pin=av::publish_sidecar(temp.path/"training.pt",training);av::read_verified(temp.path/"training.pt",training_pin,"training");
  rejects([&]{av::read_verified(temp.path/"training.pt",training_pin,"inference");},"training archive cannot be admitted for inference");
  training.resume_identity_sha256.clear();rejects([&]{av::validate_manifest(training);},"exact-resume identity is mandatory");
}
} // namespace
int main() {
  try {hash_witnesses();tensor_witnesses();lineage_witnesses();archive_witnesses();std::cout<<"Artifact sidecar v1 witnesses passed\n";return 0;}
  catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
