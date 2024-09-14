#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include "Windows.h"

#include "lyra/lyra_config.h"
#include "lyra/lyra_encoder.h"
#include "lyra/lyra_decoder.h"
#include "lyra/model_coeffs/_models.h"

#define BYTES_PER_SAMPLE 2

BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved)
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}

std::unique_ptr<chromemedia::codec::LyraEncoder> m_Encoder = nullptr;
std::unique_ptr<chromemedia::codec::LyraDecoder> m_Decoder = nullptr;

extern "C" __declspec(dllexport) bool Initialize()
{
    const int samplerate = 16000;
    const int bitrate = 3200;

    const chromemedia::codec::LyraModels models = GetEmbeddedLyraModels();

    if (!m_Encoder)
    {
        m_Encoder = chromemedia::codec::LyraEncoder::Create(samplerate, 1, bitrate, false, models);
    }
    if (!m_Decoder)
    {
        m_Decoder = chromemedia::codec::LyraDecoder::Create(samplerate, 1, models);
    }
    return m_Encoder != nullptr && m_Decoder != nullptr;
}

extern "C" __declspec(dllexport) void Shutdown()
{
    m_Encoder.reset();
    m_Decoder.reset();
}

extern "C" __declspec(dllexport) void Encode(const int16_t* uncompressed, size_t uncompressed_size, uint8_t* compressed, size_t compressed_size)
{
    const int num_samples_per_packet = m_Encoder->sample_rate_hz() / m_Encoder->frame_rate();
    const int raw_frame_size = num_samples_per_packet * BYTES_PER_SAMPLE;

    assert(uncompressed_size >= num_samples_per_packet);

    std::vector<int16_t> uncompressed_vector(uncompressed, uncompressed + num_samples_per_packet);
    std::optional<std::vector<uint8_t>> encoded = m_Encoder->Encode(uncompressed_vector);

    if (!encoded.has_value())
    {
        return;
    }

    assert(encoded->size() == chromemedia::codec::BitrateToPacketSize(m_Encoder->bitrate()));
    assert(compressed_size >= encoded->size());

    memcpy_s(compressed, compressed_size, encoded->data(), encoded->size());
}

extern "C" __declspec(dllexport) void Decode(const int8_t* compressed, size_t compressed_size, uint16_t* uncompressed, size_t uncompressed_size)
{
    const int num_samples_per_packet = m_Encoder->sample_rate_hz() / m_Encoder->frame_rate();
    const int packet_size = chromemedia::codec::BitrateToPacketSize(m_Encoder->bitrate());

    assert(compressed_size == packet_size);

    bool valid = m_Decoder->SetEncodedPacket(absl::MakeSpan(reinterpret_cast<const uint8_t*>(compressed), compressed_size));
    assert(valid == true);
    if (!valid) return;

    std::optional<std::vector<int16_t>> decoded = m_Decoder->DecodeSamples(num_samples_per_packet);
    if (!decoded.has_value())
    {
        assert(decoded.has_value());
        return;
    }

    assert(decoded->size() == num_samples_per_packet);
    assert(uncompressed_size >= decoded->size());

    memcpy_s(uncompressed, uncompressed_size * sizeof(uint16_t), decoded->data(), decoded->size() * sizeof(uint16_t));
}
