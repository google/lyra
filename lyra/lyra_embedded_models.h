#ifndef LYRA_EMBEDDED_MODELS_H_
#define LYRA_EMBEDDED_MODELS_H_

namespace chromemedia {
namespace codec {

struct LyraModel {
  const char* buffer;
  size_t size;
};

struct LyraModels {
  LyraModel lyra_config_proto;
  LyraModel lyragan;
  LyraModel quantizer;
  LyraModel soundstream_encoder;
};

}  // namespace codec
}  // namespace chromemedia

#endif  // LYRA_EMBEDDED_MODELS_H_
