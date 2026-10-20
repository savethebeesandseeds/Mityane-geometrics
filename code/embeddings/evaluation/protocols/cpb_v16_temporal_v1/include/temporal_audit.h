// Copyright (c) 2026 Waajacu. See LICENSE.
// Included inside namespace independent after archive/volume readers. No producer
// header, Tensor numerical kernel, autograd, model or optimizer is used here.
#pragma once

constexpr double gradient_absolute_tolerance=2e-7,gradient_relative_tolerance=2e-5;
void close_gradient(double actual,double expected,const std::string& why){
  need(std::isfinite(actual)&&std::isfinite(expected)&&std::abs(actual-expected)<=gradient_absolute_tolerance+gradient_relative_tolerance*std::abs(expected),why);
}
torch::Tensor temporal_field(Archive& a,const std::string& key,torch::ScalarType type,const std::vector<int64_t>& dimensions){
  auto t=a.get(key);need(t.scalar_type()==type&&t.sizes().vec()==dimensions,"exact temporal field dtype/shape: "+key);return t;
}
void same_temporal_operand(Archive& a,Archive& b,const std::string& key,torch::ScalarType type,const std::vector<int64_t>& dimensions){
  const auto x=temporal_field(a,key,type,dimensions),y=temporal_field(b,key,type,dimensions);
  if(type==torch::kFloat32)need(std::memcmp(x.const_data_ptr<float>(),y.const_data_ptr<float>(),size_t(x.numel())*sizeof(float))==0,"same-step exact F32 operand bits: "+key);
  else need(type==torch::kBool&&std::memcmp(x.const_data_ptr<bool>(),y.const_data_ptr<bool>(),size_t(x.numel())*sizeof(bool))==0,"same-step exact Bool operand bits: "+key);
}
std::string temporal_waveform(const std::vector<std::string>& paths){
  need(paths.size()==2,"temporal audit requires dedicated and same-step original objective witnesses");
  Archive a(paths[0]),original(paths[1]);
  const auto mu=volume(a,"mean"),v=volume(a,"variance"),y=volume(a,"targets");
  need(mu.dimensions==v.dimensions&&mu.dimensions==y.dimensions&&mu.values.size()<=1000000,"bounded temporal distribution geometry");
  const auto B=mu.dimensions[0],C=mu.dimensions[1],H=mu.dimensions[2],F=mu.dimensions[3],N=int64_t(mu.values.size());
  need(B<=64&&C<=64&&F<=256,"bounded temporal batch/channel/feature geometry");
  const auto q=temporal_field(a,"target_mask",torch::kBool,mu.dimensions),visible=temporal_field(a,"visible",torch::kBool,mu.dimensions);
  const auto Q=q.const_data_ptr<bool>(),V=visible.const_data_ptr<bool>();
  for(const auto& key:{"mean","variance","targets"})same_temporal_operand(a,original,key,torch::kFloat32,mu.dimensions);
  for(const auto& key:{"target_mask","visible"})same_temporal_operand(a,original,key,torch::kBool,mu.dimensions);
  const auto update=scalar_i(a,"update"),attempt=scalar_i(a,"attempt");
  need((update==1||update==2048)&&attempt>=update-1&&attempt<4096&&update==scalar_i(original,"update")&&attempt==scalar_i(original,"attempt"),"same actual first/last update and legal attempt");
  need(integers(a,"lags",3)==std::vector<int64_t>({1,4,16}),"exact ordered raw-row lags1/4/16");
  need(scalar_f(a,"coefficient")==.1&&scalar_f(a,"huber_delta")==1.&&scalar_f(a,"endpoint_scale")==std::sqrt(2.),"exact saved temporal coefficient/Huber delta/sqrt2 divisor");
  const auto pair_masks=temporal_field(a,"pair_masks",torch::kBool,{3,B,C,H,F});
  const auto pair_counts=temporal_field(a,"pair_counts",torch::kInt64,{3,B,C});
  const auto channel_losses=temporal_field(a,"lag_channel_losses",torch::kFloat64,{3,B,C});
  const auto example_losses=temporal_field(a,"lag_example_losses",torch::kFloat64,{3,B});
  const auto saved_losses=vector(a,"lag_losses",3);
  const auto saved_supported=boolean(a,"lag_supported",3,1);
  const auto saved_channel_eligible=temporal_field(a,"lag_eligible_channels",torch::kBool,{3,B,C}),saved_example_eligible=temporal_field(a,"lag_eligible_examples",torch::kBool,{3,B});
  const auto ce=saved_channel_eligible.const_data_ptr<bool>(),ee=saved_example_eligible.const_data_ptr<bool>();
  const auto mean_gradient=volume(a,"mean_total_gradient"),variance_gradient=volume(a,"variance_total_gradient");
  need(mean_gradient.dimensions==mu.dimensions&&variance_gradient.dimensions==mu.dimensions,"actual retained PRECLIP F32 output-gradient geometry");
  const auto pm=pair_masks.const_data_ptr<bool>();const auto pc=pair_counts.const_data_ptr<int64_t>();
  const auto cl=channel_losses.const_data_ptr<double>(),el=example_losses.const_data_ptr<double>();
  const auto at=[&](int64_t b,int64_t c,int64_t t,int64_t f){return size_t(((b*C+c)*H+t)*F+f);};
  std::vector<int64_t> base_counts(size_t(B*C),0),base_channels(size_t(B),0);
  int64_t base_examples=0,target_cells=0;
  for(int64_t b=0;b<B;++b){
    for(int64_t c=0;c<C;++c){for(int64_t t=0;t<H;++t)for(int64_t f=0;f<F;++f){const auto i=at(b,c,t,f);need(!(Q[i]&&V[i]),"original target and final visible masks must be disjoint");if(Q[i]){
      need(std::isfinite(mu.values[i])&&std::isfinite(y.values[i])&&std::isfinite(v.values[i])&&v.values[i]>0,"finite original supported operands and positive variance");
      ++base_counts[size_t(b*C+c)];++target_cells;
    }}base_channels[size_t(b)]+=base_counts[size_t(b*C+c)]>0;}
    base_examples+=base_channels[size_t(b)]>0;
  }
  need(base_examples>0&&target_cells>0,"nonvacuous actual base-NLL update");
  const auto original_eligible=mask_matrix(original,"eligible_channels",B,C);
  std::vector<Sum> mean_adj(static_cast<size_t>(N)),variance_adj(static_cast<size_t>(N));Sum base_loss;
  for(int64_t b=0;b<B;++b){Sum row;for(int64_t c=0;c<C;++c){const auto count=base_counts[size_t(b*C+c)];need(bool(original_eligible[size_t(b*C+c)])==bool(count),"original same-step channel eligibility");if(!count)continue;Sum channel;
    const double weight=1./(double(base_examples)*double(base_channels[size_t(b)])*double(count));
    for(int64_t t=0;t<H;++t)for(int64_t f=0;f<F;++f){const auto i=at(b,c,t,f);if(!Q[i])continue;
      const double error=mu.values[i]-y.values[i],variance=v.values[i];
      channel.add(.5*(error*error/variance+std::log(variance)));
      mean_adj[i].add(weight*error/variance);variance_adj[i].add(weight*.5*(1./variance-error*error/(variance*variance)));
    }row.add(channel.value/double(count));
  }if(base_channels[size_t(b)])base_loss.add(row.value/double(base_channels[size_t(b)]));}
  const double nll=base_loss.value/double(base_examples);
  close(scalar_f(a,"base_nll"),nll,"temporal witness base NLL");close(scalar_f(original,"reporting_nll"),nll,"original witness same-step reporting NLL");
  const std::vector<int64_t> lags{1,4,16};std::vector<double> losses(3,0);std::vector<int64_t> totals(3,0),supported(3,0);
  const double inv_sqrt_two=1./std::sqrt(2.);
  for(int64_t l=0;l<3;++l){
    std::vector<int64_t> counts(size_t(B*C),0),channels(size_t(B),0);int64_t examples=0;
    for(int64_t b=0;b<B;++b){for(int64_t c=0;c<C;++c){
      for(int64_t t=0;t<H;++t)for(int64_t f=0;f<F;++f){const auto i=at(b,c,t,f);const bool expected=t>=lags[size_t(l)]&&Q[i]&&Q[at(b,c,t-lags[size_t(l)],f)];
        need(pm[size_t(l*N)+i]==expected,"right-aligned temporal pair mask requires both ORIGINAL hidden endpoints, no wrapping/context substitution");counts[size_t(b*C+c)]+=expected;
      }const auto count=counts[size_t(b*C+c)];need(pc[size_t((l*B+b)*C+c)]==count,"independent temporal pair count");need(ce[size_t((l*B+b)*C+c)]==bool(count),"independent temporal channel eligibility");channels[size_t(b)]+=count>0;totals[size_t(l)]+=count;
    }need(ee[size_t(l*B+b)]==bool(channels[size_t(b)]),"independent temporal example eligibility");examples+=channels[size_t(b)]>0;}
    supported[size_t(l)]=examples>0;need(bool(saved_supported[size_t(l)])==bool(examples),"independent lag-supported flag");Sum lag_sum;
    for(int64_t b=0;b<B;++b){Sum row;for(int64_t c=0;c<C;++c){const auto count=counts[size_t(b*C+c)];Sum channel;
      if(count)for(int64_t t=lags[size_t(l)];t<H;++t)for(int64_t f=0;f<F;++f){const auto i=at(b,c,t,f),j=at(b,c,t-lags[size_t(l)],f);if(!(Q[i]&&Q[j]))continue;
        // All operands already promoted from F32; never subtract poisoned inactive endpoints.
        const double error=((mu.values[i]-mu.values[j])-(y.values[i]-y.values[j]))*inv_sqrt_two;
        need(std::isfinite(error),"finite F64 temporal residual");const double absolute=std::abs(error),cell=absolute<=1.?.5*error*error:absolute-.5;
        need(std::isfinite(cell),"finite independent Huber1 temporal cell loss");channel.add(cell);
        const double derivative=std::max(-1.,std::min(1.,error))*inv_sqrt_two;
        const double weight=.1/(3.*double(examples)*double(channels[size_t(b)])*double(count));
        mean_adj[i].add(weight*derivative);mean_adj[j].add(-weight*derivative);
      }
      const auto index=size_t((l*B+b)*C+c);if(count){close(cl[index],channel.value/double(count),"temporal pair-to-channel loss");row.add(channel.value/double(count));}else need(cl[index]==0.,"unsupported temporal channel is exact graphzero");
    }const auto index=size_t(l*B+b);if(channels[size_t(b)]){const double value=row.value/double(channels[size_t(b)]);close(el[index],value,"temporal channel-to-example loss");lag_sum.add(value);}else need(el[index]==0.,"unsupported temporal example is exact graphzero");}
    if(examples){losses[size_t(l)]=lag_sum.value/double(examples);close(saved_losses[size_t(l)],losses[size_t(l)],"temporal example-to-lag loss");}else need(saved_losses[size_t(l)]==0.,"unsupported lag is exact graphzero");
  }
  Sum temporal_sum;for(double value:losses)temporal_sum.add(value);const double temporal=temporal_sum.value/3.,total=nll+.1*temporal;
  close(scalar_f(a,"temporal_loss"),temporal,"FIXED one-third temporal lag reduction including unsupported lags");
  close(scalar_f(original,"temporal_huber"),temporal,"same-step original unweighted temporal component");
  close(scalar_f(a,"total_loss"),total,"V4 NLL plus0.1 temporal objective");close(scalar_f(original,"training_objective"),total,"same-step original total objective");
  for(int64_t i=0;i<N;++i){const auto k=size_t(i);close_gradient(mean_gradient.values[k],mean_adj[k].value,"retained PRECLIP mean total gradient versus independent output adjoint");close_gradient(variance_gradient.values[k],variance_adj[k].value,"retained PRECLIP variance total gradient versus unchanged base-NLL adjoint");if(!Q[k])need(mean_gradient.values[k]==0.&&variance_gradient.values[k]==0.,"unsupported output gradients are exact zero");}
  std::ostringstream out;out<<std::setprecision(17)<<"{\"kind\":\"objective_temporal\",\"witness\":"<<quoted(paths[0])<<",\"original_objective_witness\":"<<quoted(paths[1])<<",\"update\":"<<update<<",\"attempt\":"<<attempt<<",\"examples\":"<<B<<",\"valid_examples\":"<<base_examples<<",\"target_cells\":"<<target_cells<<",\"lags\":[1,4,16],\"lag_pair_totals\":[";
  for(size_t l=0;l<3;++l){if(l)out<<',';out<<totals[l];}out<<"],\"lag_supported\":[";for(size_t l=0;l<3;++l){if(l)out<<',';out<<(supported[l]?"true":"false");}out<<"],\"lag_losses\":[";for(size_t l=0;l<3;++l){if(l)out<<',';out<<losses[l];}
  out<<"],\"temporal_loss\":"<<temporal<<",\"base_nll\":"<<nll<<",\"total_loss\":"<<total<<",\"outer_weight\":0.1,\"lag_weights\":[0.3333333333333333,0.3333333333333333,0.3333333333333333],\"difference_divisor\":\"sqrt2\",\"huber_delta\":1,\"same_step_operands_bitwise_equal\":true,\"actual_preclip_output_gradients_audited\":true,\"gradient_absolute_tolerance\":"<<gradient_absolute_tolerance<<",\"gradient_relative_tolerance\":"<<gradient_relative_tolerance<<",\"arithmetic_only\":true,\"gradient_reconstruction\":\"direct_output_adjoints_only\",\"encoder_jacobian_reconstructed\":false,\"optimizer_replayed\":false,\"global_clipping_reconstructed\":false}";return out.str();
}
