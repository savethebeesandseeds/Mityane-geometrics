// SPDX-License-Identifier: AGPL-3.0-only
// Use the exact installed runtime libraries; development headers are not in the
// pinned image. These declarations match the public C ABIs used by the former
// ctypes implementation. This helper performs no package installation.
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
class Library {
 public:
  explicit Library(const char* name) : handle_(dlopen(name, RTLD_NOW | RTLD_LOCAL)) {
    if (!handle_) throw std::runtime_error(dlerror());
  }
  ~Library() { dlclose(handle_); }
  Library(const Library&) = delete;
  Library& operator=(const Library&) = delete;
  template <typename Function> Function bind(const char* name) const {
    dlerror();
    auto function = reinterpret_cast<Function>(dlsym(handle_, name));
    if (const auto* error = dlerror()) throw std::runtime_error(error);
    return function;
  }
 private:
  void* handle_;
};

int prepare(const char* source, int size) {
  Library pixbuf("libgdk_pixbuf-2.0.so.0"), webp("libwebp.so.7"), objects("libgobject-2.0.so.0");
  auto load = pixbuf.bind<void* (*)(const char*, void**)>("gdk_pixbuf_new_from_file");
  auto scale = pixbuf.bind<void* (*)(void*, int, int, int)>("gdk_pixbuf_scale_simple");
  auto pixels = pixbuf.bind<unsigned char* (*)(void*)>("gdk_pixbuf_get_pixels");
  auto stride = pixbuf.bind<int (*)(void*)>("gdk_pixbuf_get_rowstride");
  auto channels = pixbuf.bind<int (*)(void*)>("gdk_pixbuf_get_n_channels");
  auto width = pixbuf.bind<int (*)(void*)>("gdk_pixbuf_get_width");
  auto height = pixbuf.bind<int (*)(void*)>("gdk_pixbuf_get_height");
  auto unref = objects.bind<void (*)(void*)>("g_object_unref");
  auto encode = webp.bind<std::size_t (*)(const unsigned char*, int, int, int, float, unsigned char**)>("WebPEncodeRGBA");
  auto free = webp.bind<void (*)(void*)>("WebPFree");
  auto info = webp.bind<int (*)(const unsigned char*, std::size_t, int*, int*)>("WebPGetInfo");
  std::unique_ptr<void, decltype(unref)> original(load(source, nullptr), unref);
  if (!original) throw std::runtime_error("Cannot decode source artwork.");
  if (channels(original.get()) != 4 || width(original.get()) != height(original.get()))
    throw std::runtime_error("Expected square RGBA artwork.");
  // GDK_INTERP_BILINEAR = 2; retain the former 82.0 quality exactly.
  std::unique_ptr<void, decltype(unref)> scaled(scale(original.get(), size, size, 2), unref);
  if (!scaled) throw std::runtime_error("Cannot resize artwork.");
  unsigned char* output = nullptr;
  const auto length = encode(pixels(scaled.get()), size, size, stride(scaled.get()), 82.0f, &output);
  std::unique_ptr<unsigned char, decltype(free)> data(output, free);
  if (!length || !data) throw std::runtime_error("Cannot encode artwork.");
  int decoded_width = 0, decoded_height = 0;
  if (!info(data.get(), length, &decoded_width, &decoded_height) || decoded_width != size || decoded_height != size)
    throw std::runtime_error("Encoded artwork failed verification.");
  if (std::fwrite(data.get(), 1, length, stdout) != length || std::fflush(stdout) != 0)
    throw std::runtime_error("Cannot write encoded artwork.");
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--version") {
      Library webp("libwebp.so.7");
      const int version = webp.bind<int (*)()>("WebPGetEncoderVersion")();
      std::printf("libwebp %d.%d.%d\n", version >> 16, (version >> 8) & 255, version & 255);
      return 0;
    }
    if (argc != 3 || (std::string(argv[2]) != "256" && std::string(argv[2]) != "384"))
      throw std::runtime_error("Usage: prepare-branding-native <square-rgba.png> <256|384> | --version");
    return prepare(argv[1], std::atoi(argv[2]));
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
