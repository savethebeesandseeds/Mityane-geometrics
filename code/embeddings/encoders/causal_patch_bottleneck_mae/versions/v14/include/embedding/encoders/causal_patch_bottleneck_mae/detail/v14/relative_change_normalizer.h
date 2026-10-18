// Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../../../../../../LICENSE.
#pragma once
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v14/relative_change_bank.h"
#include "embedding/encoders/causal_patch_bottleneck_mae/detail/v14/conditioned_change_normalizer.h"

namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v14 {
// The unchanged v26 accumulator is reused beneath a distinct typed family.
struct RelativeBankStandardizer:BankStandardizer {
  RelativeBankStandardizer()=default;
  RelativeBankStandardizer(const Config &c,const std::string &source,const std::string &scaler):
      BankStandardizer(c,std::string(kRelativeChangeSourcePrefix)+source,scaler){}
  std::string bank_family{kRelativeChangeBankFamily},raw_contract{kRelativeChangeRawContract},increment_recipe{kRelativeChangeIncrementRecipe};
  void validate(const Config &c,const std::string *source=nullptr,const std::string *scaler=nullptr)const{
    BankStandardizer::validate(c,source,scaler);
    TORCH_CHECK(bank_family==kRelativeChangeBankFamily&&raw_contract==kRelativeChangeRawContract&&increment_recipe==kRelativeChangeIncrementRecipe&&
        source_identity.rfind(kRelativeChangeSourcePrefix,0)==0&&source_identity.size()>std::string(kRelativeChangeSourcePrefix).size(),"[rpb-v28] relative bank family/raw/recipe/source mismatch");
  }
  RelativeBankStandardizer clone()const{RelativeBankStandardizer out=*this;static_cast<BankStandardizer&>(out)=BankStandardizer::clone();return out;}
  std::string identity()const{return std::string("rpb-relative-bank-v1:")+BankStandardizer::identity();}
  void save(torch::serialize::OutputArchive &a)const{
    namespace d=configurable_checkpoint_detail;a.write("artifact_kind",d::text_tensor("rpb_relative_bank_standardizer_v1"),true);
    a.write("bank_family",d::text_tensor(bank_family),true);a.write("raw_contract",d::text_tensor(raw_contract),true);a.write("increment_recipe",d::text_tensor(increment_recipe),true);
    a.write("relative_identity",d::text_tensor(identity()),true);torch::serialize::OutputArchive bank;BankStandardizer::save(bank);a.write("bank",bank);
  }
  static RelativeBankStandardizer load(torch::serialize::InputArchive &a,const Config &c,const std::string *source=nullptr,const std::string *scaler=nullptr){
    namespace d=configurable_checkpoint_detail;TORCH_CHECK(d::read_text(a,"artifact_kind")=="rpb_relative_bank_standardizer_v1","[rpb-v28] typed relative bank normalizer required");
    RelativeBankStandardizer out;out.bank_family=d::read_text(a,"bank_family");out.raw_contract=d::read_text(a,"raw_contract");out.increment_recipe=d::read_text(a,"increment_recipe");
    const auto id=d::read_text(a,"relative_identity");torch::serialize::InputArchive bank;a.read("bank",bank);static_cast<BankStandardizer&>(out)=BankStandardizer::load(bank,c,source,scaler);
    out.validate(c,source,scaler);TORCH_CHECK(out.identity()==id,"[rpb-v28] relative normalizer content mismatch");return out;
  }
};
} // namespace embedding::encoders::causal_patch_bottleneck_mae::detail::v14
