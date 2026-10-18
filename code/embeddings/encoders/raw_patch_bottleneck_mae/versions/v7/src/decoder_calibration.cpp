// SPDX-License-Identifier: MIT
#include "embedding/encoders/raw_patch_bottleneck_mae/frozen_decoder_calibration.h"

#ifndef DECODER_CALIBRATION_SOURCE_ID
#ifdef EVALUATION_SOURCE_ID
#define DECODER_CALIBRATION_SOURCE_ID EVALUATION_SOURCE_ID
#else
#define DECODER_CALIBRATION_SOURCE_ID "unrecorded"
#endif
#endif

namespace embedding::encoders::raw_patch_bottleneck_mae {

// Historical v7 identity, namespace, positive-parent admission and typed format
// remain fixed. Only the implementation is shared with the new fresh protocol.
DecoderCalibrationRun make_decoder_calibration(const DecoderCalibrationOptions &options,
    const embedding::evaluation::ProviderFitInput &fit) {
  return frozen_decoder_detail::make(options, fit, frozen_decoder_detail::Binding::historical_v7,
      DECODER_CALIBRATION_SOURCE_ID);
}
embedding::evaluation::CurveSnapshot make_decoder_calibration_snapshot(const std::string &path,
    const std::string &parent_path, const embedding::evaluation::ProviderFitInput &fit) {
  return frozen_decoder_detail::snapshot(path, parent_path, fit, frozen_decoder_detail::Binding::historical_v7,
      DECODER_CALIBRATION_SOURCE_ID);
}
} // namespace embedding::encoders::raw_patch_bottleneck_mae
