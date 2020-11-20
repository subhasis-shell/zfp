#ifndef __BITSTRUCT_H__
#define __BITSTRUCT_H__

#include "zfp/types.h"

#ifdef BIT_STREAM_WORD_TYPE

typedef BIT_STREAM_WORD_TYPE word;

#else

typedef uint64 word;

#endif

#define wsize ((uint)(CHAR_BIT * sizeof(word)))

// bit stream structure  

struct bitstream {
  uint bits;   /* number of buffered bits (0 <= bits < wsize) */
  word buffer; /* buffer for incoming/outgoing bits (buffer < 2^bits) */
  word* ptr;   /* pointer to next word to be read/written */
  word* begin; /* beginning of stream */
  word* end;   /* end of stream (currently unused) */
#ifdef BIT_STREAM_STRIDED
  size_t mask;     /* one less the block size in number of words */
  ptrdiff_t delta; /* number of words between consecutive blocks */
#endif
};


#endif 
