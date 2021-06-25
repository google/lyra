/*
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef THIRD_PARTY_LYRA_CODEC_SPARSE_MATMUL_ZLIB_ZLIBWRAPPER_H
#define THIRD_PARTY_LYRA_CODEC_SPARSE_MATMUL_ZLIB_ZLIBWRAPPER_H

#include "zlib.h"

namespace csrblocksparse {

class GZipHeader;

class ZLib {
 public:
  ZLib();
  ~ZLib();

  // Set this to true if you want to be flexible with the gzip footer.
  static void set_should_be_flexible_with_gzip_footer(bool b) {
    should_be_flexible_with_gzip_footer_ = b;
  }

  static bool should_be_flexible_with_gzip_footer() {
    return should_be_flexible_with_gzip_footer_;
  }

  // Wipe a ZLib object to a virgin state.  This differs from Reset()
  // in that it also breaks any dictionary, gzip, etc, state.
  void Reinit();

  // Call this to make a zlib buffer as good as new.  Here's the only
  // case where they differ:
  //    CompressChunk(a); CompressChunk(b); CompressChunkDone();   vs
  //    CompressChunk(a); Reset(); CompressChunk(b); CompressChunkDone();
  // You'll want to use Reset(), then, when you interrupt a compress
  // (or uncompress) in the middle of a chunk and want to start over.
  void Reset();

  // Sets no_header_mode appropriately.  Note that using NoHeaderMode
  // in conjunction with a preset dictionary is not supported (zlib
  // starts behaving oddly if you try to do this).
  void SetNoHeaderMode(bool no_header_mode);

  // Returns our current no_header_mode.
  bool no_header_mode() const { return settings_.no_header_mode_; }

  // Uses a gzip header/footer; the output is a valid gzip file.
  // This also causes us to generate a crc32 checksum used with gzip
  void SetGzipHeaderMode();

  // By default UncompressAtMostOrAll will return Z_OK upon hitting the end of
  // the input stream. This function modifies that behavior by returning
  // Z_STREAM_END instead. This is useful when getting multiple compressed
  // documents in a single stream. Returning Z_STREAM_END will indicate the end
  // of a document.
  void SetDontHideStreamEnd();

  // Sets the compression level to be used
  void SetCompressionLevel(int level) { settings_.compression_level_ = level; }

  // Sets the size of the window (history buffer) used by the compressor.
  // The size is expressed in bits (log base 2 of the desired size).
  void SetCompressionWindowSizeInBits(int bits) {
    settings_.window_bits_ = bits;
  }

  // Controls the amount of memory used by the compresser.
  // Legal value are 1 through 9. See zlib.h for more info.
  void SetCompressionMemLevel(int level) { settings_.mem_level_ = level; }

  // Sets the initial dictionary to be used for decompression.
  void SetDictionary(const char* initial_dict, unsigned int dict_len);

  // According to the zlib manual, when you Compress, the destination
  // buffer must have size at least src + .1%*src + 12.  This function
  // helps you calculate that.  Augment this to account for a potential
  // gzip header and footer, plus a few bytes of slack.
  static uLong MinCompressbufSize(uLong uncompress_size) {
    return uncompress_size + uncompress_size / 1000 + 40;
  }

  // The minimum size of footers written by CompressChunkDone().
  int MinFooterSize() const;

  // Compresses the source buffer into the destination buffer.
  // sourceLen is the byte length of the source buffer.
  // Upon entry, destLen is the total size of the destination buffer,
  // which must be of size at least MinCompressbufSize(sourceLen).
  // Upon exit, destLen is the actual size of the compressed buffer.
  //
  // This function can be used to compress a whole file at once if the
  // input file is mmap'ed.
  //
  // Returns Z_OK if success, Z_MEM_ERROR if there was not
  // enough memory, Z_BUF_ERROR if there was not enough room in the
  // output buffer. Note that if the output buffer is exactly the same
  // size as the compressed result, we still return Z_BUF_ERROR.
  // (check CL#1936076)
  //
  // If the values of *destLen or sourceLen do not fit in an unsigned int,
  // Z_BUF_ERROR is returned.
  int Compress(Bytef* dest, uLongf* destLen, const Bytef* source,
               uLong sourceLen);

  // Uncompresses the source buffer into the destination buffer.
  // The destination buffer must be long enough to hold the entire
  // decompressed contents.
  //
  // Returns Z_OK on success, otherwise, it returns a zlib error code.
  //
  // If the values of *destLen or sourceLen do not fit in an unsigned int,
  // Z_BUF_ERROR is returned.
  int Uncompress(Bytef* dest, uLongf* destLen, const Bytef* source,
                 uLong sourceLen);

  // Get the uncompressed size from the gzip header. Returns 0 if source is too
  // short (len < 5).
  uLongf GzipUncompressedLength(const Bytef* source, uLong len);

  // Special helper function to help uncompress gzipped documents:
  // We'll allocate (with malloc) a destination buffer exactly big
  // enough to hold the gzipped content.  We set dest and destLen.
  // If we don't return Z_OK, *dest will be NULL, otherwise you
  // should free() it when you're done with it.
  // Returns Z_OK on success, otherwise, it returns a zlib error code.
  // Its the responsibility of the user to set *destLen to the
  // expected maximum size of the uncompressed data. The size of the
  // uncompressed data is read from the compressed buffer gzip footer.
  // This value cannot be trusted, so we compare it to the expected
  // maximum size supplied by the user, returning Z_MEM_ERROR if its
  // greater than the expected maximum size.
  int UncompressGzipAndAllocate(Bytef** dest, uLongf* destLen,
                                const Bytef* source, uLong sourceLen);

  // Streaming compression and decompression methods come in two
  // variations.  {Unc,C}ompressAtMost() and {Unc,C}ompressChunk().
  // The former decrements sourceLen by the amount of data that was
  // consumed: if it returns Z_BUF_ERROR, set the source of the next
  // {Unc,C}ompressAtMost() to the unconsumed data.
  // {Unc,C}ompressChunk() is the legacy interface and does not do
  // this, thus it cannot recover from a Z_BUF_ERROR (except for in
  // the first chunk).

  // Compresses data one chunk at a time -- ie you can call this more
  // than once.  This is useful for a webserver, for instance, which
  // might want to use chunked encoding with compression.  To get this
  // to work you need to call start and finish routines.
  //
  // Returns Z_OK if success, Z_MEM_ERROR if there was not
  // enough memory, Z_BUF_ERROR if there was not enough room in the
  // output buffer.

  int CompressAtMost(Bytef* dest, uLongf* destLen, const Bytef* source,
                     uLong* sourceLen);

  int CompressChunk(Bytef* dest, uLongf* destLen, const Bytef* source,
                    uLong sourceLen);

  // Emits gzip footer information, as needed.
  // destLen should be at least MinFooterSize() long.
  // Returns Z_OK, Z_MEM_ERROR, and Z_BUF_ERROR as in CompressChunk().
  int CompressChunkDone(Bytef* dest, uLongf* destLen);

  // Uncompress data one chunk at a time -- ie you can call this
  // more than once.  To get this to work you need to call per-chunk
  // and "done" routines.
  //
  // Returns Z_OK if success, Z_MEM_ERROR if there was not
  // enough memory, Z_BUF_ERROR if there was not enough room in the
  // output buffer.

  int UncompressAtMost(Bytef* dest, uLongf* destLen, const Bytef* source,
                       uLong* sourceLen);
  int UncompressChunk(Bytef* dest, uLongf* destLen, const Bytef* source,
                      uLong sourceLen);

  // Checks gzip footer information, as needed.  Mostly this just
  // makes sure the checksums match.  Whenever you call this, it
  // will assume the last 8 bytes from the previous UncompressChunk
  // call are the footer.  Returns true iff everything looks ok.
  bool UncompressChunkDone();

  // Only meaningful for chunked compressing/uncompressing. It's true
  // after initialization or reset and before the first chunk of
  // user data is received.
  bool first_chunk() const { return first_chunk_; }

  // Returns a pointer to our current dictionary:
  const Bytef* dictionary() const { return settings_.dictionary_; }

  // Convenience method to check if a bytestream has a header.  This
  // is intended as a quick test: "Is this likely a GZip file?"
  static bool HasGzipHeader(const char* source, int sourceLen);

  // Have we parsed the complete gzip footer, and does it match the
  // length and CRC checksum of the content that we have uncompressed
  // so far?
  bool IsGzipFooterValid() const;

  // Accessor for the uncompressed size (first added to address issue #509976)
  uLong uncompressed_size() const { return uncompressed_size_; }

 private:
  int InflateInit();  // sets up the zlib inflate structure
  int DeflateInit();  // sets up the zlib deflate structure

  // These init the zlib data structures for compressing/uncompressing
  int CompressInit(Bytef* dest, uLongf* destLen, const Bytef* source,
                   uLong* sourceLen);
  int UncompressInit(Bytef* dest, uLongf* destLen, const Bytef* source,
                     uLong* sourceLen);
  // Initialization method to be called if we hit an error while
  // uncompressing. On hitting an error, call this method before
  // returning the error.
  void UncompressErrorInit();
  // Helper functions to write gzip-specific data
  int WriteGzipHeader();
  int WriteGzipFooter(Bytef* dest, uLongf destLen);

  // Helper function for both Compress and CompressChunk
  int CompressChunkOrAll(Bytef* dest, uLongf* destLen, const Bytef* source,
                         uLong sourceLen, int flush_mode);
  int CompressAtMostOrAll(Bytef* dest, uLongf* destLen, const Bytef* source,
                          uLong* sourceLen, int flush_mode);

  // Likewise for UncompressAndUncompressChunk
  int UncompressChunkOrAll(Bytef* dest, uLongf* destLen, const Bytef* source,
                           uLong sourceLen, int flush_mode);

  int UncompressAtMostOrAll(Bytef* dest, uLongf* destLen, const Bytef* source,
                            uLong* sourceLen, int flush_mode);

  // Initialization method to be called if we hit an error while
  // compressing. On hitting an error, call this method before
  // returning the error.
  void CompressErrorInit();

  // Makes sure the parameters are valid
  void CheckValidParams();

  struct Settings {
    // null if we don't want an initial dictionary
    Bytef* dictionary_;  // NOLINT

    // initial dictionary length
    unsigned int dict_len_;  // NOLINT

    // compression level
    int compression_level_;  // NOLINT

    // log base 2 of the window size used in compression
    int window_bits_;  // NOLINT

    // specifies the amount of memory to be used by compressor (1-9)
    int mem_level_;  // NOLINT

    // true if we want/expect no zlib headers
    bool no_header_mode_;  // NOLINT

    // true if we want/expect gzip headers
    bool gzip_header_mode_;  // NOLINT

    // Controls behavior of UncompressAtMostOrAll with regards to returning
    // Z_STREAM_END. See comments for SetDontHideStreamEnd.
    bool dont_hide_zstream_end_;  // NOLINT
  };

  // We allow all kinds of bad footers when this flag is true.
  // Some web servers send bad pages corresponding to these cases
  // and IE is tolerant with it.
  // - Extra bytes after gzip footer (see bug 69126)
  // - No gzip footer (see bug 72896)
  // - Incomplete gzip footer (see bug 71871706)
  static bool should_be_flexible_with_gzip_footer_;

  // "Current" settings. These will be used whenever we next configure zlib.
  // For example changing compression level or header mode will be recorded
  // in these, but don't usually get applied immediately but on next compress.
  Settings settings_;

  // Settings last used to initialise and configure zlib. These are needed
  // to know if the current desired configuration in settings_ is sufficiently
  // compatible with the previous configuration and we can just reconfigure the
  // underlying zlib objects, or have to recreate them from scratch.
  Settings init_settings_;

  z_stream comp_stream_;    // Zlib stream data structure
  bool comp_init_;          // True if we have initialized comp_stream_
  z_stream uncomp_stream_;  // Zlib stream data structure
  bool uncomp_init_;        // True if we have initialized uncomp_stream_

  // These are used only in gzip compression mode
  uLong crc_;  // stored in gzip footer, fitting 4 bytes
  uLong uncompressed_size_;

  GZipHeader* gzip_header_;  // our gzip header state

  Byte gzip_footer_[8];    // stored footer, used to uncompress
  int gzip_footer_bytes_;  // num of footer bytes read so far, or -1

  // These are used only with chunked compression.
  bool first_chunk_;  // true if we need to emit headers with this chunk
};

}  // namespace csrblocksparse

#endif  //  THIRD_PARTY_LYRA_CODEC_SPARSE_MATMUL_ZLIB_ZLIBWRAPPER_H
