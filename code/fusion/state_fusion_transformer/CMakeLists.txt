cmake_minimum_required(VERSION 3.18)
project(state_fusion LANGUAGES CXX)

# CUDA toolkit 12.4 and the LibTorch 2.6.0 CUDA build at
# /opt/cuwacunu_embedding/libtorch. This package has no .cu sources.
# Attention and training run through LibTorch's CUDA kernels.
# The managed image has no cmake binary. The Makefile is the container
# entry, and it passes the same include path, C++17 mode, ABI, and libraries.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED CMAKE_PREFIX_PATH)
  list(APPEND CMAKE_PREFIX_PATH "/opt/cuwacunu_embedding/libtorch")
endif()

find_package(Torch 2.6 REQUIRED)
find_library(STATE_FUSION_TORCH_CUDA torch_cuda HINTS "${TORCH_INSTALL_PREFIX}/lib")
if(NOT STATE_FUSION_TORCH_CUDA)
  message(FATAL_ERROR "state_fusion requires the LibTorch CUDA build (libtorch_cuda).")
endif()

add_library(state_fusion
  src/blocks.cpp
  src/embeddings.cpp
  src/model.cpp
  src/synthetic.cpp)
target_include_directories(state_fusion PUBLIC include)
target_link_libraries(state_fusion PUBLIC "${TORCH_LIBRARIES}")
target_compile_definitions(state_fusion PUBLIC _GLIBCXX_USE_CXX11_ABI=1)

function(state_fusion_executable name source)
  add_executable(${name} ${source})
  target_link_libraries(${name} PRIVATE state_fusion)
  set_target_properties(${name} PROPERTIES
    BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib")
endfunction()

state_fusion_executable(state_fusion_train src/train.cpp)
state_fusion_executable(state_fusion_capacity src/capacity.cpp)
state_fusion_executable(state_fusion_lego src/lego.cpp)
state_fusion_executable(state_fusion_invariance tests/test_invariance.cpp)
state_fusion_executable(state_fusion_fidelity tests/test_fidelity.cpp)
state_fusion_executable(state_fusion_query tests/test_query.cpp)

enable_testing()
add_test(NAME invariance COMMAND state_fusion_invariance)
add_test(NAME fidelity COMMAND state_fusion_fidelity)
add_test(NAME query COMMAND state_fusion_query)
