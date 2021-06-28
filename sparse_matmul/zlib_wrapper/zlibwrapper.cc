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

#include "sparse_matmul/zlib_wrapper/zlibwrapper.h"

#include <assert.h>
#include <stdio.h>

#include <algorithm>
#include <memory>
#include <string>

#include "glog/logging.h"
#include "sparse_matmul/zlib_wrapper/gzipheader.h"
#include "zconf.h"
#include "zlib.h"

// The GZIP header (see RFC 1952):
//   +---+---+---+---+---+---+---+---+---+---+
//   |ID1|ID2|CM |FLG|     MTIME     |XFL|OS |
//   +---+---+---+---+---+---+---+---+---+---+
//     ID1     \037
//     ID2     \213
//     CM      \010 (compression method == DEFLATE)
//     FLG     \000 (special flags that we do not support)
//     MTIME   Unix format modification time (0 means not available)
//     XFL     2-4? DEFLATE flags
//     OS      ???? Operating system indicator (255 means unknown)

// Header value we generate:
// We use a #define so sizeof() works correctly
#define GZIP_HEADER "\037\213\010\000\000\000\000\000\002\377"

namespace csrblocksparse {

// We allow all kinds of bad footers when this flag is true.
// Some web servers send bad pages corresponding to these cases
// and IE is tolerant with it.
// - Extra bytes after gzip footer (see bug 69126)
// - No gzip footer (see bug 72896)
// - Incomplete gzip footer (see bug 71871706)
bool ZLib::should_be_flexible_with_gzip_footer_ = false;

// Initialize the ZLib class
ZLib::ZLib()
    : comp_init_(false), uncomp_init_(false), gzip_header_(new GZipHeader) {
  Reinit();
  init_settings_ = settings_;
}

ZLib::~ZLib() {
  if (comp_init_) {
    deflateEnd(&comp_stream_);
  }
  if (uncomp_init_) {
    inflateEnd(&uncomp_stream_);
  }
  delete gzip_header_;
}

void ZLib::Reinit() {
  settings_.dictionary_ = nullptr;
  settings_.dict_len_ = 0;
  settings_.compression_level_ = Z_DEFAULT_COMPRESSION;
  settings_.window_bits_ = MAX_WBITS;
  settings_.mem_level_ = 8;  // DEF_MEM_LEVEL
  settings_.no_header_mode_ = false;
  settings_.gzip_header_mode_ = false;
  settings_.dont_hide_zstream_end_ = false;

  if (comp_init_) {
    int err = deflateReset(&comp_stream_);
    if (err != Z_OK) {
      deflateEnd(&comp_stream_);
      comp_init_ = false;
    }
  }
  if (uncomp_init_) {
    // Use negative window bits size to indicate bare stream with no header.
    int wbits = (settings_.no_header_mode_ ? -MAX_WBITS : MAX_WBITS);
    int err = inflateReset2(&uncomp_stream_, wbits);
    if (err == Z_OK) {
      init_settings_.no_header_mode_ = settings_.no_header_mode_;
    } else {
      inflateEnd(&uncomp_stream_);
      uncomp_init_ = false;
    }
  }
  crc_ = 0;
  uncompressed_size_ = 0;
  gzip_header_->Reset();
  gzip_footer_bytes_ = -1;
  first_chunk_ = true;
}

void ZLib::Reset() {
  first_chunk_ = true;
  gzip_header_->Reset();
}

void ZLib::CheckValidParams() {
  if (settings_.dictionary_ != nullptr &&
      (settings_.no_header_mode_ || settings_.gzip_header_mode_)) {
    LOG(FATAL)
        << "Incompatible params: require zlib headers with preset dictionary";
  }
}

void ZLib::SetNoHeaderMode(bool no_header_mode) {
  settings_.no_header_mode_ = no_header_mode;
  if (init_settings_.no_header_mode_ != settings_.no_header_mode_) {
    // Once the header mode changes, we have to reinitialize all our streams
    if (comp_init_) {
      deflateEnd(&comp_stream_);
      comp_init_ = false;
    }
    if (uncomp_init_) {
      inflateEnd(&uncomp_stream_);
      uncomp_init_ = false;
    }
  } else {
    // Mode hasn't changed, but treat this as a reset request nevertheless
    Reset();
  }
  CheckValidParams();
}

void ZLib::SetGzipHeaderMode() {
  settings_.gzip_header_mode_ = true;
  SetNoHeaderMode(true);  // we use gzip headers, not zlib headers
  CheckValidParams();
}

void ZLib::SetDictionary(const char* initial_dict, unsigned int dict_len) {
  settings_.dictionary_ = (Bytef*)initial_dict;  // NOLINT
  settings_.dict_len_ = dict_len;
  CheckValidParams();
}

void ZLib::SetDontHideStreamEnd() { settings_.dont_hide_zstream_end_ = true; }

int ZLib::MinFooterSize() const {
  int min_footer_size = 2;  // Room for empty chunk.
  if (settings_.gzip_header_mode_) {
    min_footer_size += 8;  // Room for actual footer.
  }
  return min_footer_size;
}

// --------- COMPRESS MODE

// Initialization method to be called if we hit an error while
// compressing. On hitting an error, call this method before returning
// the error.
void ZLib::CompressErrorInit() {
  if (comp_init_) {
    deflateEnd(&comp_stream_);
    comp_init_ = false;
  }
  Reset();
}

// These probably return Z_OK, but may return Z_BUF_ERROR if outbuf is full
int ZLib::WriteGzipHeader() {
  if (comp_stream_.avail_out < sizeof(GZIP_HEADER)) return Z_BUF_ERROR;
  memcpy(comp_stream_.next_out, GZIP_HEADER, sizeof(GZIP_HEADER) - 1);
  comp_stream_.next_out += sizeof(GZIP_HEADER) - 1;
  comp_stream_.avail_out -= sizeof(GZIP_HEADER) - 1;
  return Z_OK;
}

int ZLib::WriteGzipFooter(Bytef* dest, uLongf destLen) {
  if (destLen < 8)  // not enough space for footer
    return Z_BUF_ERROR;
  *dest++ = (crc_ >> 0) & 255;
  *dest++ = (crc_ >> 8) & 255;
  *dest++ = (crc_ >> 16) & 255;
  *dest++ = (crc_ >> 24) & 255;
  *dest++ = (uncompressed_size_ >> 0) & 255;
  *dest++ = (uncompressed_size_ >> 8) & 255;
  *dest++ = (uncompressed_size_ >> 16) & 255;
  *dest++ = (uncompressed_size_ >> 24) & 255;
  return Z_OK;
}

int ZLib::DeflateInit() {
  int err =
      deflateInit2(&comp_stream_, settings_.compression_level_, Z_DEFLATED,
                   (settings_.no_header_mode_ ? -settings_.window_bits_
                                              : settings_.window_bits_),
                   settings_.mem_level_, Z_DEFAULT_STRATEGY);
  if (err == Z_OK) {
    // Save parameters for later reusability checks
    init_settings_.compression_level_ = settings_.compression_level_;
    init_settings_.window_bits_ = settings_.window_bits_;
    init_settings_.mem_level_ = settings_.mem_level_;
    init_settings_.no_header_mode_ = settings_.no_header_mode_;
  }
  return err;
}

int ZLib::CompressInit(Bytef* dest, uLongf* destLen, const Bytef* source,
                       uLong* sourceLen) {
  int err;

  comp_stream_.next_in = (Bytef*)source;  // NOLINT
  comp_stream_.avail_in = (uInt)*sourceLen;
  // Check for sourceLen (unsigned long) to fit into avail_in (unsigned int).
  if ((uLong)comp_stream_.avail_in != *sourceLen) return Z_BUF_ERROR;
  comp_stream_.next_out = dest;
  comp_stream_.avail_out = (uInt)*destLen;
  // Check for destLen (unsigned long) to fit into avail_out (unsigned int).
  if ((uLong)comp_stream_.avail_out != *destLen) return Z_BUF_ERROR;

  if (!first_chunk_)  // only need to set up stream the first time through
    return Z_OK;

  // Force full reinit if properties have changed in a way we can't adjust.
  if (comp_init_ &&
      (init_settings_.dictionary_ != settings_.dictionary_ ||
       init_settings_.dict_len_ != settings_.dict_len_ ||
       init_settings_.window_bits_ != settings_.window_bits_ ||
       init_settings_.mem_level_ != settings_.mem_level_ ||
       init_settings_.no_header_mode_ != settings_.no_header_mode_)) {
    deflateEnd(&comp_stream_);
    comp_init_ = false;
  }

  // Reuse if we've already initted the object.
  if (comp_init_) {  // we've already initted it
    err = deflateReset(&comp_stream_);
    if (err != Z_OK) {
      deflateEnd(&comp_stream_);
      comp_init_ = false;
    }
  }

  // If compression level has changed, try to reconfigure instead of reinit
  if (comp_init_ &&
      init_settings_.compression_level_ != settings_.compression_level_) {
    err = deflateParams(&comp_stream_, settings_.compression_level_,
                        Z_DEFAULT_STRATEGY);
    if (err == Z_OK) {
      init_settings_.compression_level_ = settings_.compression_level_;
    } else {
      deflateEnd(&comp_stream_);
      comp_init_ = false;
    }
  }

  // First use or previous state was not reusable with current settings.
  if (!comp_init_) {
    comp_stream_.zalloc = (alloc_func)0;
    comp_stream_.zfree = (free_func)0;
    comp_stream_.opaque = (voidpf)0;
    err = DeflateInit();
    if (err != Z_OK) return err;
    comp_init_ = true;
  }
  return Z_OK;
}

// In a perfect world we'd always have the full buffer to compress
// when the time came, and we could just call Compress().  Alas, we
// want to do chunked compression on our webserver.  In this
// application, we compress the header, send it off, then compress the
// results, send them off, then compress the footer.  Thus we need to
// use the chunked compression features of zlib.
int ZLib::CompressAtMostOrAll(Bytef* dest, uLongf* destLen, const Bytef* source,
                              uLong* sourceLen,
                              int flush_mode) {  // Z_FULL_FLUSH or Z_FINISH
  int err;

  if ((err = CompressInit(dest, destLen, source, sourceLen)) != Z_OK)
    return err;

  // This is used to figure out how many bytes we wrote *this chunk*
  int compressed_size = comp_stream_.total_out;

  // Some setup happens only for the first chunk we compress in a run
  if (first_chunk_) {
    // Append the gzip header before we start compressing
    if (settings_.gzip_header_mode_) {
      if ((err = WriteGzipHeader()) != Z_OK) return err;
      compressed_size -= sizeof(GZIP_HEADER) - 1;  // -= is right: adds to size
      crc_ = crc32(0, nullptr, 0);                 // initialize
    }

    // Initialize the dictionary just before we start compressing
    if (settings_.dictionary_) {
      err = deflateSetDictionary(&comp_stream_, settings_.dictionary_,
                                 settings_.dict_len_);
      if (err != Z_OK) return err;
      init_settings_.dictionary_ = settings_.dictionary_;
      init_settings_.dict_len_ = settings_.dict_len_;
    }

    uncompressed_size_ = 0;
    first_chunk_ = false;  // so we don't do this again
  }

  // flush_mode is Z_FINISH for all mode, Z_SYNC_FLUSH for incremental
  // compression.
  err = deflate(&comp_stream_, flush_mode);

  const uLong source_bytes_consumed = *sourceLen - comp_stream_.avail_in;
  *sourceLen = comp_stream_.avail_in;

  if ((err == Z_STREAM_END || err == Z_OK) && comp_stream_.avail_in == 0 &&
      comp_stream_.avail_out != 0) {
    // we processed everything ok and the output buffer was large enough.
    {}
  } else if (err == Z_STREAM_END && comp_stream_.avail_in > 0) {
    return Z_BUF_ERROR;  // should never happen
  } else if (err != Z_OK && err != Z_STREAM_END && err != Z_BUF_ERROR) {
    // an error happened
    CompressErrorInit();
    return err;
  } else if (comp_stream_.avail_out == 0) {  // not enough space
    err = Z_BUF_ERROR;
  }

  assert(err == Z_OK || err == Z_STREAM_END || err == Z_BUF_ERROR);
  if (err == Z_STREAM_END) err = Z_OK;

  // update the crc and other metadata
  uncompressed_size_ += source_bytes_consumed;
  compressed_size = comp_stream_.total_out - compressed_size;  // delta
  *destLen = compressed_size;
  if (settings_.gzip_header_mode_)  // don't bother with crc else
    crc_ = crc32(crc_, source, source_bytes_consumed);

  return err;
}

int ZLib::CompressChunkOrAll(Bytef* dest, uLongf* destLen, const Bytef* source,
                             uLong sourceLen,
                             int flush_mode) {  // Z_FULL_FLUSH or Z_FINISH
  const int ret =
      CompressAtMostOrAll(dest, destLen, source, &sourceLen, flush_mode);
  if (ret == Z_BUF_ERROR) CompressErrorInit();
  return ret;
}

int ZLib::CompressChunk(Bytef* dest, uLongf* destLen, const Bytef* source,
                        uLong sourceLen) {
  return CompressChunkOrAll(dest, destLen, source, sourceLen, Z_SYNC_FLUSH);
}

int ZLib::CompressAtMost(Bytef* dest, uLongf* destLen, const Bytef* source,
                         uLong* sourceLen) {
  return CompressAtMostOrAll(dest, destLen, source, sourceLen, Z_SYNC_FLUSH);
}

// This writes the gzip footer info, if necessary.
// No matter what, we call Reset() so we can compress Chunks again.
int ZLib::CompressChunkDone(Bytef* dest, uLongf* destLen) {
  // Make sure our buffer is of reasonable size.
  if (*destLen < MinFooterSize()) {
    *destLen = 0;
    return Z_BUF_ERROR;
  }

  // The underlying zlib library requires a non-nullptr source pointer, even if
  // the source length is zero, otherwise it will generate an (incorrect) zero-
  // valued CRC checksum.
  char dummy = '\0';
  int err;

  assert(!first_chunk_ && comp_init_);

  const uLongf orig_destLen = *destLen;
  // NOLINTNEXTLINE
  if ((err = CompressChunkOrAll(dest, destLen, (const Bytef*)&dummy, 0,
                                Z_FINISH)) != Z_OK) {
    Reset();  // we assume they won't retry on error
    return err;
  }

  // Make sure that when we exit, we can start a new round of chunks later
  // (This must be set after the call to CompressChunkOrAll() above.)
  Reset();

  // Write gzip footer if necessary.  They're explicitly in little-endian order
  if (settings_.gzip_header_mode_) {
    if ((err = WriteGzipFooter(dest + *destLen, orig_destLen - *destLen)) !=
        Z_OK)
      return err;
    *destLen += 8;  // zlib footer took up another 8 bytes
  }
  return Z_OK;  // stream_end is ok
}

// This routine only initializes the compression stream once.  Thereafter, it
// just does a deflateReset on the stream, which should be faster.
int ZLib::Compress(Bytef* dest, uLongf* destLen, const Bytef* source,
                   uLong sourceLen) {
  int err;
  const uLongf orig_destLen = *destLen;
  if ((err = CompressChunkOrAll(dest, destLen, source, sourceLen, Z_FINISH)) !=
      Z_OK)
    return err;
  Reset();  // reset for next call to Compress

  if (settings_.gzip_header_mode_) {
    if ((err = WriteGzipFooter(dest + *destLen, orig_destLen - *destLen)) !=
        Z_OK)
      return err;
    *destLen += 8;  // zlib footer took up another 8 bytes
  }

  return Z_OK;
}

// --------- UNCOMPRESS MODE

int ZLib::InflateInit() {
  // Use negative window bits size to indicate bare stream with no header.
  int wbits = (settings_.no_header_mode_ ? -MAX_WBITS : MAX_WBITS);
  int err = inflateInit2(&uncomp_stream_, wbits);
  if (err == Z_OK) {
    init_settings_.no_header_mode_ = settings_.no_header_mode_;
  }
  return err;
}

// Initialization method to be called if we hit an error while
// uncompressing. On hitting an error, call this method before
// returning the error.
void ZLib::UncompressErrorInit() {
  if (uncomp_init_) {
    inflateEnd(&uncomp_stream_);
    uncomp_init_ = false;
  }
  Reset();
}

int ZLib::UncompressInit(Bytef* dest, uLongf* destLen, const Bytef* source,
                         uLong* sourceLen) {
  int err;

  uncomp_stream_.next_in = (Bytef*)source;  // NOLINT
  uncomp_stream_.avail_in = (uInt)*sourceLen;
  // Check for sourceLen (unsigned long) to fit into avail_in (unsigned int).
  if ((uLong)uncomp_stream_.avail_in != *sourceLen) return Z_BUF_ERROR;

  uncomp_stream_.next_out = dest;
  uncomp_stream_.avail_out = (uInt)*destLen;
  // Check for destLen (unsigned long) to fit into avail_out (unsigned int).
  if ((uLong)uncomp_stream_.avail_out != *destLen) return Z_BUF_ERROR;

  if (!first_chunk_)  // only need to set up stream the first time through
    return Z_OK;

  // Force full reinit if properties have changed in a way we can't adjust.
  if (uncomp_init_ && (init_settings_.dictionary_ != settings_.dictionary_ ||
                       init_settings_.dict_len_ != settings_.dict_len_)) {
    inflateEnd(&uncomp_stream_);
    uncomp_init_ = false;
  }

  // Reuse if we've already initted the object.
  if (uncomp_init_) {
    // Use negative window bits size to indicate bare stream with no header.
    int wbits = (settings_.no_header_mode_ ? -MAX_WBITS : MAX_WBITS);
    err = inflateReset2(&uncomp_stream_, wbits);
    if (err == Z_OK) {
      init_settings_.no_header_mode_ = settings_.no_header_mode_;
    } else {
      UncompressErrorInit();
    }
  }

  // First use or previous state was not reusable with current settings.
  if (!uncomp_init_) {
    uncomp_stream_.zalloc = (alloc_func)0;
    uncomp_stream_.zfree = (free_func)0;
    uncomp_stream_.opaque = (voidpf)0;
    err = InflateInit();
    if (err != Z_OK) return err;
    uncomp_init_ = true;
  }
  return Z_OK;
}

// If you compressed your data a chunk at a time, with CompressChunk,
// you can uncompress it a chunk at a time with UncompressChunk.
// Only difference bewteen chunked and unchunked uncompression
// is the flush mode we use: Z_SYNC_FLUSH (chunked) or Z_FINISH (unchunked).
int ZLib::UncompressAtMostOrAll(Bytef* dest, uLongf* destLen,
                                const Bytef* source, uLong* sourceLen,
                                int flush_mode) {  // Z_SYNC_FLUSH or Z_FINISH
  int err = Z_OK;

  if (first_chunk_) {
    gzip_footer_bytes_ = -1;
    if (settings_.gzip_header_mode_) {
      // If we haven't read our first chunk of actual compressed data,
      // and we're expecting gzip headers, then parse some more bytes
      // from the gzip headers.
      const Bytef* bodyBegin = nullptr;
      GZipHeader::Status status = gzip_header_->ReadMore(
          reinterpret_cast<const char*>(source), *sourceLen,
          reinterpret_cast<const char**>(&bodyBegin));
      switch (status) {
        case GZipHeader::INCOMPLETE_HEADER:  // don't have the complete header
          *destLen = 0;
          *sourceLen = 0;  // GZipHeader used all the input
          return Z_OK;
        case GZipHeader::INVALID_HEADER:  // bogus header
          Reset();
          return Z_DATA_ERROR;
        case GZipHeader::COMPLETE_HEADER:      // we have the full header
          *sourceLen -= (bodyBegin - source);  // skip past header bytes
          source = bodyBegin;
          crc_ = crc32(0, nullptr, 0);  // initialize CRC
          break;
        default:
          LOG(FATAL) << "Unexpected gzip header parsing result: " << status;
      }
    }
  } else if (gzip_footer_bytes_ >= 0) {
    // We're now just reading the gzip footer. We already read all the data.
    if (gzip_footer_bytes_ + *sourceLen > sizeof(gzip_footer_) &&
        // When this flag is true, we allow some extra bytes after the
        // gzip footer.
        !should_be_flexible_with_gzip_footer_) {
      VLOG(1) << "UncompressChunkOrAll: Received "
              << (gzip_footer_bytes_ + *sourceLen - sizeof(gzip_footer_))
              << " extra bytes after gzip footer: "
              << std::string(reinterpret_cast<const char*>(source),
                             std::min(*sourceLen, 20UL));
      Reset();
      return Z_DATA_ERROR;
    }
    uLong len = sizeof(gzip_footer_) - gzip_footer_bytes_;
    if (len > *sourceLen) len = *sourceLen;
    if (len > 0) {
      memcpy(gzip_footer_ + gzip_footer_bytes_, source, len);
      gzip_footer_bytes_ += len;
    }
    *sourceLen -= len;
    *destLen = 0;
    return Z_OK;
  }

  if ((err = UncompressInit(dest, destLen, source, sourceLen)) != Z_OK) {
    LOG(WARNING) << "ZLib: UncompressInit: Error: " << err
                 << "SourceLen: " << *sourceLen;
    return err;
  }

  // This is used to figure out how many output bytes we wrote *this chunk*:
  const uLong old_total_out = uncomp_stream_.total_out;

  // This is used to figure out how many input bytes we read *this chunk*:
  const uLong old_total_in = uncomp_stream_.total_in;

  // Some setup happens only for the first chunk we compress in a run
  if (first_chunk_) {
    // Initialize the dictionary just before we start compressing
    if (settings_.gzip_header_mode_ || settings_.no_header_mode_) {
      // In no_header_mode, we can just set the dictionary, since no
      // checking is done to advance past header bits to get us in the
      // dictionary setting mode. In settings_.gzip_header_mode_ we've already
      // removed headers, so this code works too.
      if (settings_.dictionary_) {
        err = inflateSetDictionary(&uncomp_stream_, settings_.dictionary_,
                                   settings_.dict_len_);
        if (err != Z_OK) {
          LOG(WARNING) << "inflateSetDictionary: Error: " << err
                       << " dict_len: " << settings_.dict_len_;
          UncompressErrorInit();
          return err;
        }
        init_settings_.dictionary_ = settings_.dictionary_;
        init_settings_.dict_len_ = settings_.dict_len_;
      }
    }

    first_chunk_ = false;  // so we don't do this again

    // For the first chunk *only* (to avoid infinite troubles), we let
    // there be no actual data to uncompress.  This sometimes triggers
    // when the input is only the gzip header, say.
    if (*sourceLen == 0) {
      *destLen = 0;
      return Z_OK;
    }
  }

  // We'll uncompress as much as we can.  If we end OK great, otherwise
  // if we get an error that seems to be the gzip footer, we store the
  // gzip footer and return OK, otherwise we return the error.

  // flush_mode is Z_SYNC_FLUSH for chunked mode, Z_FINISH for all mode.
  err = inflate(&uncomp_stream_, flush_mode);
  if (settings_.dictionary_ && err == Z_NEED_DICT) {
    err = inflateSetDictionary(&uncomp_stream_, settings_.dictionary_,
                               settings_.dict_len_);
    if (err != Z_OK) {
      LOG(WARNING) << "UncompressChunkOrAll: failed in inflateSetDictionary : "
                   << err;
      UncompressErrorInit();
      return err;
    }
    init_settings_.dictionary_ = settings_.dictionary_;
    init_settings_.dict_len_ = settings_.dict_len_;
    err = inflate(&uncomp_stream_, flush_mode);
  }

  // Figure out how many bytes of the input zlib slurped up:
  const uLong bytes_read = uncomp_stream_.total_in - old_total_in;
  CHECK_LE(source + bytes_read, source + *sourceLen);
  *sourceLen = uncomp_stream_.avail_in;

  // Next we look at the footer, if any. Note that we might currently
  // have just part of the footer (eg, if this data is arriving over a
  // socket). After looking for a footer, log a warning if there is
  // extra cruft.
  if ((err == Z_STREAM_END) &&
      ((gzip_footer_bytes_ == -1) ||
       (gzip_footer_bytes_ < sizeof(gzip_footer_))) &&
      (uncomp_stream_.avail_in <= sizeof(gzip_footer_) ||
       // When this flag is true, we allow some extra bytes after the
       // zlib footer.
       should_be_flexible_with_gzip_footer_)) {
    // Due to a bug in old versions of zlibwrapper, we appended the gzip
    // footer even in non-gzip mode.  Thus we always allow a gzip footer
    // even if we're not in gzip mode, so we can continue to uncompress
    // the old data.  :-(

    // Store gzip footer bytes so we can check for footer consistency
    // in UncompressChunkDone(). (If we have the whole footer, we
    // could do the checking here, but we don't to keep consistency
    // with CompressChunkDone().)
    gzip_footer_bytes_ = std::min(static_cast<size_t>(uncomp_stream_.avail_in),
                                  sizeof(gzip_footer_));
    memcpy(gzip_footer_, source + bytes_read, gzip_footer_bytes_);
    *sourceLen -= gzip_footer_bytes_;
  } else if ((err == Z_STREAM_END || err == Z_OK)  // everything went ok
             && uncomp_stream_.avail_in == 0) {    // and we read it all
    {}
  } else if (err == Z_STREAM_END && uncomp_stream_.avail_in > 0) {
    VLOG(1) << "UncompressChunkOrAll: Received some extra data, bytes total: "
            << uncomp_stream_.avail_in << " bytes: "
            << std::string(
                   reinterpret_cast<const char*>(uncomp_stream_.next_in),
                   std::min(static_cast<int>(uncomp_stream_.avail_in), 20));
    UncompressErrorInit();
    return Z_DATA_ERROR;  // what's the extra data for?
  } else if (err != Z_OK && err != Z_STREAM_END && err != Z_BUF_ERROR) {
    // an error happened
    VLOG(1) << "UncompressChunkOrAll: Error: " << err
            << " avail_out: " << uncomp_stream_.avail_out;
    UncompressErrorInit();
    return err;
  } else if (uncomp_stream_.avail_out == 0) {
    err = Z_BUF_ERROR;
  }

  assert(err == Z_OK || err == Z_BUF_ERROR || err == Z_STREAM_END);
  if (err == Z_STREAM_END && !settings_.dont_hide_zstream_end_) err = Z_OK;

  // update the crc and other metadata
  uncompressed_size_ = uncomp_stream_.total_out;
  *destLen = uncomp_stream_.total_out - old_total_out;  // size for this call
  if (settings_.gzip_header_mode_) crc_ = crc32(crc_, dest, *destLen);

  return err;
}

int ZLib::UncompressChunkOrAll(Bytef* dest, uLongf* destLen,
                               const Bytef* source, uLong sourceLen,
                               int flush_mode) {  // Z_SYNC_FLUSH or Z_FINISH
  const int ret =
      UncompressAtMostOrAll(dest, destLen, source, &sourceLen, flush_mode);
  if (ret == Z_BUF_ERROR) UncompressErrorInit();
  return ret;
}

int ZLib::UncompressAtMost(Bytef* dest, uLongf* destLen, const Bytef* source,
                           uLong* sourceLen) {
  return UncompressAtMostOrAll(dest, destLen, source, sourceLen, Z_SYNC_FLUSH);
}

int ZLib::UncompressChunk(Bytef* dest, uLongf* destLen, const Bytef* source,
                          uLong sourceLen) {
  return UncompressChunkOrAll(dest, destLen, source, sourceLen, Z_SYNC_FLUSH);
}

// We make sure we've uncompressed everything, that is, the current
// uncompress stream is at a compressed-buffer-EOF boundary.  In gzip
// mode, we also check the gzip footer to make sure we pass the gzip
// consistency checks.  We RETURN true iff both types of checks pass.
bool ZLib::UncompressChunkDone() {
  if (first_chunk_ || !uncomp_init_) {
    return false;
  }
  // Make sure we're at the end-of-compressed-data point.  This means
  // if we call inflate with Z_FINISH we won't consume any input or
  // write any output
  Bytef dummyin, dummyout;
  uLongf dummylen = 0;
  if (UncompressChunkOrAll(&dummyout, &dummylen, &dummyin, 0, Z_FINISH) !=
      Z_OK) {
    return false;
  }

  // Make sure that when we exit, we can start a new round of chunks later
  Reset();

  // We don't need to check footer when this flag is true.
  if (should_be_flexible_with_gzip_footer_) {
    return true;
  }

  // Whether we were hoping for a gzip footer or not, we allow a gzip
  // footer.  (See the note above about bugs in old zlibwrappers.) But
  // by the time we've seen all the input, it has to be either a
  // complete gzip footer, or no footer at all.
  if ((gzip_footer_bytes_ != -1) && (gzip_footer_bytes_ != 0) &&
      (gzip_footer_bytes_ != sizeof(gzip_footer_)))
    return false;

  if (!settings_.gzip_header_mode_) return true;

  return IsGzipFooterValid();
}

bool ZLib::IsGzipFooterValid() const {
  // If we were expecting a gzip footer, and didn't get a full one,
  // that's an error.
  if (gzip_footer_bytes_ == -1 || gzip_footer_bytes_ < sizeof(gzip_footer_))
    return false;

  // The footer holds the lower four bytes of the length.
  uLong uncompressed_size = 0;
  uncompressed_size += static_cast<uLong>(gzip_footer_[7]) << 24;
  uncompressed_size += gzip_footer_[6] << 16;
  uncompressed_size += gzip_footer_[5] << 8;
  uncompressed_size += gzip_footer_[4] << 0;
  if (uncompressed_size != (uncompressed_size_ & 0xffffffff)) {
    return false;
  }

  uLong checksum = 0;
  checksum += static_cast<uLong>(gzip_footer_[3]) << 24;
  checksum += gzip_footer_[2] << 16;
  checksum += gzip_footer_[1] << 8;
  checksum += gzip_footer_[0] << 0;
  if (crc_ != checksum) return false;

  return true;
}

// Uncompresses the source buffer into the destination buffer.
// The destination buffer must be long enough to hold the entire
// decompressed contents.
//
// We only initialize the uncomp_stream once.  Thereafter, we use
// inflateReset2, which should be faster.
//
// Returns Z_OK on success, otherwise, it returns a zlib error code.
int ZLib::Uncompress(Bytef* dest, uLongf* destLen, const Bytef* source,
                     uLong sourceLen) {
  int err;
  if ((err = UncompressChunkOrAll(dest, destLen, source, sourceLen,
                                  Z_FINISH)) != Z_OK) {
    Reset();  // let us try to compress again
    return err;
  }
  if (!UncompressChunkDone())  // calls Reset()
    return Z_DATA_ERROR;
  return Z_OK;  // stream_end is ok
}

// read uncompress length from gzip footer
uLongf ZLib::GzipUncompressedLength(const Bytef* source, uLong len) {
  if (len <= 4) return 0;  // malformed data.

  return (static_cast<uLongf>(source[len - 1]) << 24) +
         (static_cast<uLongf>(source[len - 2]) << 16) +
         (static_cast<uLongf>(source[len - 3]) << 8) +
         (static_cast<uLongf>(source[len - 4]) << 0);
}

int ZLib::UncompressGzipAndAllocate(Bytef** dest, uLongf* destLen,
                                    const Bytef* source, uLong sourceLen) {
  *dest = nullptr;  // until we successfully allocate
  if (!settings_.gzip_header_mode_) return Z_VERSION_ERROR;  // *shrug*

  uLongf uncompress_length = GzipUncompressedLength(source, sourceLen);

  // Do not trust the uncompress size reported by the compressed buffer.
  if (uncompress_length > *destLen) {
    if (!HasGzipHeader(reinterpret_cast<const char*>(source), sourceLen)) {
      VLOG(1) << "Attempted to un-gzip data that is not gzipped.";
      return Z_DATA_ERROR;
    }
    VLOG(1) << "Uncompressed size " << uncompress_length
            << " exceeds maximum expected size " << *destLen;
    return Z_MEM_ERROR;  // probably a corrupted gzip buffer
  }

  *destLen = uncompress_length;

  *dest = (Bytef*)malloc(*destLen);  // NOLINT
  if (*dest == nullptr)              // probably a corrupted gzip buffer
    return Z_MEM_ERROR;

  const int retval = Uncompress(*dest, destLen, source, sourceLen);
  if (retval != Z_OK) {  // just to make life easier for them
    free(*dest);
    *dest = nullptr;
  }
  return retval;
}

// Convenience method to check if a bytestream has a header.  This
// is intended as a quick test: "Is this likely a GZip file?"
bool ZLib::HasGzipHeader(const char* source, int sourceLen) {
  GZipHeader gzh;
  const char* ptr = nullptr;
  return gzh.ReadMore(source, sourceLen, &ptr) == GZipHeader::COMPLETE_HEADER;
}

}  // namespace csrblocksparse
