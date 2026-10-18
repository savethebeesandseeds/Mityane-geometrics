// Settings parsing inherited under MIT; see LICENSES/NOTICE.
// RPB-v27 additions Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../../../../../LICENSE.
#pragma once
#include "cuwacunu/config/document.h"
#include "embedding/encoders/raw_patch_bottleneck_mae/parent_conditional_change_config.h"

namespace embedding::encoders::raw_patch_bottleneck_mae {
inline ParentConditionalChangeConfig parent_conditional_change_settings(const cuwacunu::config::Document &document){
  document.reject_unknown_blocks({"ENCODER"});const auto &b=document.block("ENCODER");
  b.reject_unknown_keys({"MODEL_TAG","CHANNEL_COUNT","HISTORY_LENGTH","INPUT_WIDTH","PATCH_LENGTH","EMBEDDING_SIZE",
      "LEVEL_EMBEDDING_SIZE","ENCODER_WIDTH","NUM_LAYERS","NUM_HEADS","FEEDFORWARD_WIDTH","DECODER_HIDDEN_WIDTH",
      "LAYER_NORM_EPSILON","MASK_RATIO","HUBER_DELTA","SCALE_FLOOR","SAMPLING_INTERVAL","CHANNEL_IDS","DEVICE"});
  TORCH_CHECK(b.string("MODEL_TAG")==kParentConditionalChangeModelTag,"[rpb-v27 settings] explicit MODEL_TAG must be RPB-v27");
  const auto device=b.string_or("DEVICE","cuda:0");
  TORCH_CHECK(device=="cuda" || device.rfind("cuda:",0)==0,"[rpb-v27 settings] CUDA device required");
  auto c=default_content_ordered_relation_config(torch::Device(device));
  c.channel_count=b.integer_or("CHANNEL_COUNT",c.channel_count);c.history_length=b.integer_or("HISTORY_LENGTH",c.history_length);
  c.input_width=b.integer_or("INPUT_WIDTH",c.input_width);c.patch_length=b.integer_or("PATCH_LENGTH",c.patch_length);
  c.export_width=b.integer_or("LEVEL_EMBEDDING_SIZE",c.export_width);c.encoder_width=b.integer_or("ENCODER_WIDTH",c.encoder_width);
  c.num_layers=b.integer_or("NUM_LAYERS",c.num_layers);c.num_heads=b.integer_or("NUM_HEADS",c.num_heads);
  c.feedforward_width=b.integer_or("FEEDFORWARD_WIDTH",c.feedforward_width);
  c.decoder_hidden_width=b.integer_or("DECODER_HIDDEN_WIDTH",c.decoder_hidden_width);
  c.layer_norm_epsilon=b.number_or("LAYER_NORM_EPSILON",c.layer_norm_epsilon);
  c.mask_ratio=b.number_or("MASK_RATIO",c.mask_ratio);c.huber_delta=b.number_or("HUBER_DELTA",c.huber_delta);
  c.scale_floor=b.number_or("SCALE_FLOOR",c.scale_floor);c.sampling_interval=b.number_or("SAMPLING_INTERVAL",c.sampling_interval);
  c.channel_ids=b.integers_or("CHANNEL_IDS",{});
  TORCH_CHECK(c.export_width>0 && c.export_width<=std::numeric_limits<int64_t>::max()-4,"[rpb-v27 settings] invalid level dimension");
  const auto native=b.integer_or("EMBEDDING_SIZE",c.export_width+4);
  TORCH_CHECK(native>c.export_width,"[rpb-v27 settings] native width must exceed level width");
  ParentConditionalChangeConfig result{c,native-c.export_width};validate_parent_conditional_change_config(result);
  result.level.channel_ids=resolved_channel_ids(result.level);return result;
}
inline ParentConditionalChangeConfig read_parent_conditional_change_settings(const std::filesystem::path &p){return parent_conditional_change_settings(cuwacunu::config::Document::read(p));}
} // namespace embedding::encoders::raw_patch_bottleneck_mae
