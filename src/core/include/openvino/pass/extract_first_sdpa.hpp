// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/pass/pass.hpp"

namespace ov {
namespace pass {
class OPENVINO_API ExtractFirstSDPA : public ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ExtractFirstSDPA");

    ExtractFirstSDPA() = default;
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};
}  // namespace pass
}  // namespace ov
