#ifdef _OPENMP

#if defined(WITH_IPP)
/*
* This source code file was modified with Intel(R) Integrated Performance Primitives library content
*/
#endif
/* compress 1d contiguous array in parallel */
static void
_t2(compress_omp, Scalar, 1)(zfp_stream* stream, const zfp_field* field)
{
  /* array metadata */
  const Scalar* data = (const Scalar*)field->data;
  uint16* length_table = NULL;
  if (stream->index){
    length_table = (uint16*)stream->index->data;
  }
  size_t nx = field->nx;

  /* number of omp threads, blocks, and chunks */
  size_t threads = thread_count_omp(stream);
  size_t blocks = (nx + 3) / 4;
  size_t chunks = chunk_count_omp(stream, blocks, threads);
  int chunk;

  /* allocate per-thread streams */
  bitstream** bs = compress_init_par(stream, field, chunks, blocks);
  if (!bs)
    return;

  /* compress chunks of blocks in parallel */
  #pragma omp parallel for num_threads(threads)
  for (chunk = 0; chunk < (int)chunks; chunk++) {
    /* determine range of block indices assigned to this thread */
    size_t bmin = chunk_offset(blocks, chunks, chunk + 0);
    size_t bmax = chunk_offset(blocks, chunks, chunk + 1);
    size_t block;
    /* set up thread-local bit stream */
    zfp_stream s = *stream;
    zfp_stream_set_bit_stream(&s, bs[chunk]);
    /* compress sequence of blocks */
    for (block = bmin; block < bmax; block++) {
      /* determine block origin x within array */
      const Scalar* p = data;
      size_t x = 4 * block;
      p += x;
      /* compress partial or full block */
      if (nx - x < 4u)
        if (length_table)
          length_table[block] = _t2(zfp_encode_partial_block_strided, Scalar, 1)(&s, p, MIN(nx - x, 4u), 1);
        else
          _t2(zfp_encode_partial_block_strided, Scalar, 1)(&s, p, MIN(nx - x, 4u), 1);
      else
        if (length_table)
          length_table[block] = _t2(zfp_encode_block, Scalar, 1)(&s, p);
        else
          _t2(zfp_encode_block, Scalar, 1)(&s, p);
    }
  }

  /* concatenate per-thread streams */
  compress_finish_par(stream, bs, chunks);
}

/* compress 1d strided array in parallel */
static void
_t2(compress_strided_omp, Scalar, 1)(zfp_stream* stream, const zfp_field* field)
{
  /* array metadata */
  const Scalar* data = (const Scalar*)field->data;
  uint16* length_table = NULL;
  if (stream->index){
    length_table = (uint16*)stream->index->data;
  }
  size_t nx = field->nx;
  ptrdiff_t sx = field->sx ? field->sx : 1;

  /* number of omp threads, blocks, and chunks */
  size_t threads = thread_count_omp(stream);
  size_t blocks = (nx + 3) / 4;
  size_t chunks = chunk_count_omp(stream, blocks, threads);
  int chunk;

  /* allocate per-thread streams */
  bitstream** bs = compress_init_par(stream, field, chunks, blocks);
  if (!bs)
    return;

  /* compress chunks of blocks in parallel */
  #pragma omp parallel for num_threads(threads)
  for (chunk = 0; chunk < (int)chunks; chunk++) {
    /* determine range of block indices assigned to this thread */
    size_t bmin = chunk_offset(blocks, chunks, chunk + 0);
    size_t bmax = chunk_offset(blocks, chunks, chunk + 1);
    size_t block;
    /* set up thread-local bit stream */
    zfp_stream s = *stream;
    zfp_stream_set_bit_stream(&s, bs[chunk]);
    /* compress sequence of blocks */
    for (block = bmin; block < bmax; block++) {
      /* determine block origin x within array */
      const Scalar* p = data;
      size_t x = 4 * block;
      p += sx * (ptrdiff_t)x;
      /* compress partial or full block */
      if (nx - x < 4u)
        if (length_table)
          length_table[block] = _t2(zfp_encode_partial_block_strided, Scalar, 1)(&s, p, MIN(nx - x, 4u), sx);
        else
          _t2(zfp_encode_partial_block_strided, Scalar, 1)(&s, p, MIN(nx - x, 4u), sx);
      else
        if (length_table)
          length_table[block] = _t2(zfp_encode_block_strided, Scalar, 1)(&s, p, sx);
        else
          _t2(zfp_encode_block_strided, Scalar, 1)(&s, p, sx);
    }
  }

  /* concatenate per-thread streams */
  compress_finish_par(stream, bs, chunks);
}

/* compress 2d strided array in parallel */
static void
_t2(compress_strided_omp, Scalar, 2)(zfp_stream* stream, const zfp_field* field)
{
  /* array metadata */
  const Scalar* data = (const Scalar*)field->data;
  uint16* length_table = NULL;
  if (stream->index){
    length_table = (uint16*)stream->index->data;
  }
  size_t nx = field->nx;
  size_t ny = field->ny;
  ptrdiff_t sx = field->sx ? field->sx : 1;
  ptrdiff_t sy = field->sy ? field->sy : (int)nx;

  /* number of omp threads, blocks, and chunks */
  size_t threads = thread_count_omp(stream);
  size_t bx = (nx + 3) / 4;
  size_t by = (ny + 3) / 4;
  size_t blocks = bx * by;
  size_t chunks = chunk_count_omp(stream, blocks, threads);
  int chunk;

  /* allocate per-thread streams */
  bitstream** bs = compress_init_par(stream, field, chunks, blocks);
  if (!bs)
    return;

  /* compress chunks of blocks in parallel */
  #pragma omp parallel for num_threads(threads)
  for (chunk = 0; chunk < (int)chunks; chunk++) {
    /* determine range of block indices assigned to this thread */
    size_t bmin = chunk_offset(blocks, chunks, chunk + 0);
    size_t bmax = chunk_offset(blocks, chunks, chunk + 1);
    size_t block;
    /* set up thread-local bit stream */
    zfp_stream s = *stream;
    zfp_stream_set_bit_stream(&s, bs[chunk]);
    /* compress sequence of blocks */
    for (block = bmin; block < bmax; block++) {
      /* determine block origin (x, y) within array */
      const Scalar* p = data;
      size_t b = block;
      size_t x, y;
      x = 4 * (b % bx); b /= bx;
      y = 4 * b;
      p += sx * (ptrdiff_t)x + sy * (ptrdiff_t)y;
      /* compress partial or full block */
      if (nx - x < 4u || ny - y < 4u)
        if (length_table)
          length_table[block] = _t2(zfp_encode_partial_block_strided, Scalar, 2)(&s, p, MIN(nx - x, 4u), MIN(ny - y, 4u), sx, sy);
        else
          _t2(zfp_encode_partial_block_strided, Scalar, 2)(&s, p, MIN(nx - x, 4u), MIN(ny - y, 4u), sx, sy);
      else
        if (length_table)
          length_table[block] = _t2(zfp_encode_block_strided, Scalar, 2)(&s, p, sx, sy);
        else
          _t2(zfp_encode_block_strided, Scalar, 2)(&s, p, sx, sy);
    }
  }

  /* concatenate per-thread streams */
  compress_finish_par(stream, bs, chunks);
}

#if defined(IPP_OPTIMIZATION_ENABLED) && !defined(_SET_TMP_BLOCK_FROM_)
#define _SET_TMP_BLOCK_FROM_
static void  CopyFromPartialBlock(const Ipp32f *pSrc, int stepY, int stepZ, int sizeX, int sizeY, int sizeZ, Ipp32f *pTmpBlock)
{
    Ipp32f    *pTmp;
    int       x, y, z, serIdx;
    int       copyX, copyY, copyZ;
    for (serIdx = z = 0; z < 4; z++) {
        copyZ = (z < sizeZ) ? z : sizeZ - 1;
        for (y = 0; y < 4; y++) {
            copyY = (y < sizeY) ? y : sizeY - 1;
            pTmp = (Ipp32f*)pSrc + copyZ * stepZ + copyY * stepY;
            for (x = 0; x < 4; x++) {
                copyX = (x < sizeX) ? x : sizeX - 1;
                pTmpBlock[serIdx++] = pTmp[copyX];
            }
        }
    }
}
#endif

/* compress 3d strided array in parallel */
static void
_t2(compress_strided_omp, Scalar, 3)(zfp_stream* stream, const zfp_field* field)
{
  /* array metadata */
  const Scalar* data = (const Scalar*)field->data;
  uint16* length_table = NULL;
  if (stream->index){
    length_table = (uint16*)stream->index->data;
  }
  size_t nx = field->nx;
  size_t ny = field->ny;
  size_t nz = field->nz;
  ptrdiff_t sx = field->sx ? field->sx : 1;
  ptrdiff_t sy = field->sy ? field->sy : (int)nx;
  ptrdiff_t sz = field->sz ? field->sz : (int)(nx * ny);

  /* number of omp threads, blocks, and chunks */
  size_t threads = thread_count_omp(stream);
  size_t bx = (nx + 3) / 4;
  size_t by = (ny + 3) / 4;
  size_t bz = (nz + 3) / 4;
  size_t blocks = bx * by * bz;
  size_t chunks = chunk_count_omp(stream, blocks, threads);
  int chunk;

  /* allocate per-thread streams */
  bitstream** bs = compress_init_par(stream, field, chunks, blocks);
  // if (!bs)
  //   return;

#if defined (IPP_OPTIMIZATION_ENABLED)

  IppEncodeZfpState_32f* pStates = NULL;
  Ipp64u* chunk_bit_lengths = (Ipp64u*)malloc(sizeof(Ipp64u)* chunks);
  int srcBlockLineStep = nx * sizeof(Ipp32f);
  int srcBlockPlaneStep = ny * srcBlockLineStep;
  uint min_bits, max_bits, max_prec;
  int min_exp;
  int sizeState = 0;
  if (!(REVERSIBLE(stream)))
  {
    zfp_stream_params(stream, &min_bits, &max_bits, &max_prec, &min_exp);
    ippsEncodeZfpGetStateSize_32f(&sizeState);
    pStates = (IppEncodeZfpState_32f*)ippsMalloc_8u(sizeState * threads);
  }
#endif

  /* compress chunks of blocks in parallel */
  int chunk;
#if !defined (IPP_OPTIMIZATION_ENABLED)
  #pragma omp parallel for num_threads(threads)
#else
  #pragma omp parallel num_threads(threads)
  {
    bitstream *pBitStream = NULL;
    IppEncodeZfpState_32f* pState = NULL;
    Ipp32f pTmpBlock[64];
    if (!(REVERSIBLE(stream)))
    {
      pState = (IppEncodeZfpState_32f*)((Ipp8u*)pStates + omp_get_thread_num() * sizeState);
    }
  #pragma omp for
#endif
  for (chunk = 0; chunk < (int)chunks; chunk++) {
    /* determine range of block indices assigned to this thread */
    size_t bmin = chunk_offset(blocks, chunks, chunk + 0);
    size_t bmax = chunk_offset(blocks, chunks, chunk + 1);
    size_t block;
    /* set up thread-local bit stream */
    zfp_stream s = *stream;
    zfp_stream_set_bit_stream(&s, bs[chunk]);

    #if defined (IPP_OPTIMIZATION_ENABLED)
    if (!(REVERSIBLE(stream)))
    {
      pBitStream = bs[chunk];
      ippsEncodeZfpInitLong_32f((Ipp8u*)stream_data(pBitStream), stream_capacity(pBitStream), pState);
      ippsEncodeZfpSet_32f(min_bits, max_bits, max_prec, min_exp, pState);
    }
    #endif

    /* compress sequence of blocks */
    for (block = bmin; block < bmax; block++) {
      /* determine block origin (x, y, z) within array */
      const Scalar* p = data;
      size_t b = block;
      size_t x, y, z;
      x = 4 * (b % bx); b /= bx;
      y = 4 * (b % by); b /= by;
      z = 4 * b;
      p += sx * (ptrdiff_t)x + sy * (ptrdiff_t)y + sz * (ptrdiff_t)z;
      /* compress partial or full block */
      if (nx - x < 4u || ny - y < 4u || nz - z < 4u)
      {
        if (length_table)
          length_table[block] = _t2(zfp_encode_partial_block_strided, Scalar, 3)(&s, p, MIN(nx - x, 4u), MIN(ny - y, 4u), MIN(nz - z, 4u), sx, sy, sz);
        else
          #if !defined(IPP_OPTIMIZATION_ENABLED)
            _t2(zfp_encode_partial_block_strided, Scalar, 3)(&s, p, MIN(nx - x, 4u), MIN(ny - y, 4u), MIN(nz - z, 4u), sx, sy, sz);
          #else
            if (!(REVERSIBLE(stream)))
            {
              CopyFromPartialBlock((const Ipp32f *)p, sy, sz, MIN(nx - x, 4u), MIN(ny - y, 4u), MIN(nz - z, 4u), pTmpBlock);
              ippsEncodeZfp444_32f(pTmpBlock, 4 * sizeof(Ipp32f), 4 * 4 * sizeof(Ipp32f), pState);
            }
            else
            {
              _t2(zfp_encode_partial_block_strided, Scalar, 3)(&s, p, MIN(nx - x, 4u), MIN(ny - y, 4u), MIN(nz - z, 4u), sx, sy, sz);
            }
          #endif
      }
      else
      {
        if (length_table)
          length_table[block] = _t2(zfp_encode_block_strided, Scalar, 3)(&s, p, sx, sy, sz);
        else
          #if !defined(IPP_OPTIMIZATION_ENABLED)
            _t2(zfp_encode_block_strided, Scalar, 3)(&s, p, sx, sy, sz);
          #else
            if (!(REVERSIBLE(stream)))
            {
              ippsEncodeZfp444_32f((const Ipp32f *)p, srcBlockLineStep, srcBlockPlaneStep, pState);
            }
            else 
            {
              _t2(zfp_encode_block_strided, Scalar, 3)(&s, p, sx, sy, sz);
            }
          #endif
      }
    } /* block loop end */

#if defined (IPP_OPTIMIZATION_ENABLED)
    if (!(REVERSIBLE(stream)) && pState != NULL)
    {
      Ipp64u chunk_compr_length;
      ippsEncodeZfpGetCompressedBitSize_32f(pState, &chunk_bit_lengths[chunk]);
      ippsEncodeZfpFlush_32f(pState);
      chunk_compr_length = (size_t)((chunk_bit_lengths[chunk] + 7) >> 3);
      stream_set_eos(pBitStream, chunk_compr_length);
    }
  #endif
  } /* chunk loop end */

#if defined (IPP_OPTIMIZATION_ENABLED)
}//The end of pragma omp parallel block

  /* concatenate per-thread streams */
  if (!(REVERSIBLE(stream)) && pStates != NULL)
    {
        compress_finish_par_opt(stream, bs, chunks, chunk_bit_lengths);
        free(chunk_bit_lengths);
        ippsFree(pStates);
        return;
    }
#else
  compress_finish_par(stream, bs, chunks);
#endif
}

/* compress 4d strided array in parallel */
static void
_t2(compress_strided_omp, Scalar, 4)(zfp_stream* stream, const zfp_field* field)
{
  /* array metadata */
  const Scalar* data = field->data;
  uint16* length_table = NULL;
  if (stream->index){
    length_table = (uint16*)stream->index->data;
  }
  size_t nx = field->nx;
  size_t ny = field->ny;
  size_t nz = field->nz;
  size_t nw = field->nw;
  ptrdiff_t sx = field->sx ? field->sx : 1;
  ptrdiff_t sy = field->sy ? field->sy : (int)nx;
  ptrdiff_t sz = field->sz ? field->sz : (int)(nx * ny);
  ptrdiff_t sw = field->sw ? field->sw : (int)(nx * ny * nz);

  /* number of omp threads, blocks, and chunks */
  size_t threads = thread_count_omp(stream);
  size_t bx = (nx + 3) / 4;
  size_t by = (ny + 3) / 4;
  size_t bz = (nz + 3) / 4;
  size_t bw = (nw + 3) / 4;
  size_t blocks = bx * by * bz * bw;
  size_t chunks = chunk_count_omp(stream, blocks, threads);
  int chunk;

  /* allocate per-thread streams */
  bitstream** bs = compress_init_par(stream, field, chunks, blocks);
  if (!bs)
    return;

  /* compress chunks of blocks in parallel */
  #pragma omp parallel for num_threads(threads)
  for (chunk = 0; chunk < (int)chunks; chunk++) {
    /* determine range of block indices assigned to this thread */
    size_t bmin = chunk_offset(blocks, chunks, chunk + 0);
    size_t bmax = chunk_offset(blocks, chunks, chunk + 1);
    size_t block;
    /* set up thread-local bit stream */
    zfp_stream s = *stream;
    zfp_stream_set_bit_stream(&s, bs[chunk]);
    /* compress sequence of blocks */
    for (block = bmin; block < bmax; block++) {
      /* determine block origin (x, y, z, w) within array */
      const Scalar* p = data;
      size_t b = block;
      size_t x, y, z, w;
      x = 4 * (b % bx); b /= bx;
      y = 4 * (b % by); b /= by;
      z = 4 * (b % bz); b /= bz;
      w = 4 * b;
      p += sx * (ptrdiff_t)x + sy * (ptrdiff_t)y + sz * (ptrdiff_t)z + sw * (ptrdiff_t)w;
      /* compress partial or full block */
      if (nx - x < 4u || ny - y < 4u || nz - z < 4u || nw - w < 4u)
        if (length_table)
          length_table[block] = _t2(zfp_encode_partial_block_strided, Scalar, 4)(&s, p, MIN(nx - x, 4u), MIN(ny - y, 4u), MIN(nz - z, 4u), MIN(nw - w, 4u), sx, sy, sz, sw);
        else
          _t2(zfp_encode_partial_block_strided, Scalar, 4)(&s, p, MIN(nx - x, 4u), MIN(ny - y, 4u), MIN(nz - z, 4u), MIN(nw - w, 4u), sx, sy, sz, sw);
      else
        if (length_table)
          length_table[block] = _t2(zfp_encode_block_strided, Scalar, 4)(&s, p, sx, sy, sz, sw);
        else
          _t2(zfp_encode_block_strided, Scalar, 4)(&s, p, sx, sy, sz, sw);
    }
  }

  /* concatenate per-thread streams */
  compress_finish_par(stream, bs, chunks);
}

#endif
