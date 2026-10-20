// Copyright (c) 2026 Waajacu. See LICENSE.
#pragma once
#ifdef CPB_OBJECTIVE_CANDIDATE
#include "embedding/encoders/causal_patch_bottleneck_mae/versions/v15.h"
#define CPB_OBJECTIVE_VERSION v15
namespace cpb_v15_objective { inline constexpr const char *arm="unit_huber1_plus_variance_nll_v15"; }
#else
#include "embedding/encoders/causal_patch_bottleneck_mae/versions/v4.h"
#define CPB_OBJECTIVE_VERSION v4
namespace cpb_v15_objective { inline constexpr const char *arm="fresh_v4_partial_nll_control"; }
#endif
