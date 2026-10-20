// Copyright (c) 2026 Waajacu. See LICENSE.
#pragma once
#ifdef CPB_OBJECTIVE_CANDIDATE
#include "embedding/encoders/causal_patch_bottleneck_mae/versions/v16.h"
#define CPB_OBJECTIVE_VERSION v16
namespace cpb_v16_temporal { inline constexpr const char *arm="v4_nll_plus_hidden_temporal_huber_v16"; }
#else
#include "embedding/encoders/causal_patch_bottleneck_mae/versions/v4.h"
#define CPB_OBJECTIVE_VERSION v4
namespace cpb_v16_temporal { inline constexpr const char *arm="fresh_v4_partial_nll_control"; }
#endif
