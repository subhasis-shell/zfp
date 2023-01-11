#ifndef __BITSTRUCT_H__
#define __BITSTRUCT_H__

#include "zfp/types.h"

/* bit stream word/buffer type; granularity of stream I/O operations */
#ifdef BIT_STREAM_WORD_TYPE
  /* may be 8-, 16-, 32-, or 64-bit unsigned integer type */
  typedef BIT_STREAM_WORD_TYPE word;
#else
    /* use maximum word size by default for highest speed */
    typedef uint64 word;
#endif

    /* number of bits in a buffered word */

#define wsize ((uint)(CHAR_BIT * sizeof(word)))

    /* bit stream structure (opaque to caller) */
    
    struct bitstream {
      uint bits;   /* number of buffered bits (0 <= bits < wsize) */
      word buffer; /* buffer for incoming/outgoing bits (buffer < 2^bits) */
      word* ptr;   /* pointer to next word to be read/written */
      word* begin; /* beginning of stream */
      word* end;   /* end of stream (currently unused) */
      ushort *bitlengths; /* Individual block lengths (for variable bit rate) */

#ifdef BIT_STREAM_STRIDED
      size_t mask;     /* one less the block size in number of words */
      ptrdiff_t delta; /* number of words between consecutive blocks */
#endif
    };

#endif


