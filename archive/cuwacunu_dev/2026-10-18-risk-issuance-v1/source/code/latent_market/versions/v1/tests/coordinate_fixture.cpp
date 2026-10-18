// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
#include <torch/serialize.h>
#include <filesystem>
#include <iostream>

// Deliberately corrupt only a copied synthetic state-bank coordinate. This is
// test tooling, never an archive migration or an inference compatibility shim.
int main(int argc, char** argv) try {
  TORCH_CHECK(argc == 4, "usage: latent_market_v1_coordinate_fixture SOURCE NEW_DESTINATION retired-token|stale-checkpoint");
  const std::filesystem::path source=argv[1], destination=argv[2];
  const auto root=std::filesystem::path("/opt/cuwacunu/build");
  TORCH_CHECK(std::filesystem::canonical(source).string().starts_with(root.string()+"/") &&
      std::filesystem::weakly_canonical(destination).string().starts_with(root.string()+"/") &&
      !std::filesystem::exists(destination), "Coordinate fixtures require a new managed-build destination");
  torch::serialize::InputArchive input; input.load_from(source.string(), torch::kCPU);
  torch::serialize::OutputArchive output;
  for (const auto* key:{"artifact_kind","timeline_sha256","checkpoint_sha256","target_coordinate_id","pair_ledger_sha256"}) {
    torch::IValue value; input.read(key,value);
    TORCH_CHECK(value.isString(), "Expected string identity: ", key);
    if (std::string(key)=="target_coordinate_id") {
      auto coordinate=std::string(value.toStringRef());
      const std::string token=":Target State Representation Network:";
      const auto offset=coordinate.find(token);
      TORCH_CHECK(offset!=std::string::npos && coordinate.size()>=64, "Expected a current synthetic target coordinate");
      const std::string mode=argv[3];
      if (mode=="retired-token") coordinate.replace(offset,token.size(),":target-G-bar:");
      else {
        TORCH_CHECK(mode=="stale-checkpoint", "Unknown coordinate corruption mode");
        coordinate.back()=coordinate.back()=='0'?'1':'0';
      }
      value=torch::IValue(coordinate);
    }
    output.write(key,value);
  }
  for (const auto* key:{"online","target","valid","endpoint_ms","fitting_forecasts","fitting_forecast_valid","fit_cutoff_ms"}) {
    torch::Tensor value; input.read(key,value,true); output.write(key,value,true);
  }
  output.save_to(destination.string());
  return 0;
} catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
