#ifdef _OPENMP

/* decompress 1d contiguous array in parallel */
static void
_t2(decompress_omp, Scalar, 1)(zfp_stream* stream, zfp_field* field)
{
  Scalar* data = (Scalar*)field->data;
  const uint nx = field->nx;
  const uint threads = thread_count_omp(stream);
  const zfp_mode mode = zfp_stream_compression_mode(stream);

  /* calculate the number of blocks and chunks */
  const uint blocks = (nx + 3) / 4;
  uint index_granularity;
  if (mode == zfp_mode_fixed_accuracy || mode == zfp_mode_fixed_precision) {
    if (stream->index == NULL)
      return;
    else {
      /* TODO: support variable index granularity! 
      Ideally this should be readable from stream->index->granularity, or similar
      Allowing this number to be >>1 is extremely beneficial */
      index_granularity = 1;
      /* TODO: support more types
      current implementation only supports OpenMP decompression with an offset table */
      if (stream->index->type != zfp_index_offset)
        return;
    }
  }
  const uint chunks = (blocks + index_granularity - 1) / index_granularity;

  /* allocate per-thread streams */
  bitstream** bs = decompress_init_par(stream, field, chunks, blocks);
  if (!bs)
    return;

  /* decompress chunks of blocks in parallel */
  int chunk;
  #pragma omp parallel for num_threads(threads)
  for (chunk = 0; chunk < (int)chunks; chunk++) {
    /* determine range of block indices assigned to this thread */
    const uint bmin = chunk * index_granularity;
    const uint bmax = MIN(bmin + index_granularity, blocks);
    uint block;

    /* set up thread-local bit stream */
    zfp_stream s = *stream;
    zfp_stream_set_bit_stream(&s, bs[chunk]);

    /* decode all blocks in the chunk sequentially */
    uint x;
    Scalar * block_data;

    for (block = bmin; block < bmax; block++) {
      x = block * 4;
      block_data = data + x;
      if (nx - x < 4)
        _t2(zfp_decode_partial_block_strided, Scalar, 1)(&s, block_data, nx - x, 1);
      else
        _t2(zfp_decode_block, Scalar, 1)(&s, block_data);
    }
  }
  decompress_finish_par(bs, chunks);
  /* TODO: find a better solution
  this workaround reads a bit from the bitstream, because the bitstream pointer is checked to see if decompression was succesful */
  stream_read_bit(stream->stream);
}

/* decompress 1d strided array in parallel */
static void
_t2(decompress_strided_omp, Scalar, 1)(zfp_stream* stream, zfp_field* field)
{
  Scalar* data = (Scalar*)field->data;
  const uint nx = field->nx;
  const int sx = field->sx ? field->sx : 1;
  const uint threads = thread_count_omp(stream);
  const zfp_mode mode = zfp_stream_compression_mode(stream);

  /* calculate the number of blocks and chunks */
  const uint blocks = (nx + 3) / 4;
  uint index_granularity = 1;
  if (mode == zfp_mode_fixed_accuracy || mode == zfp_mode_fixed_precision) {
    if (!stream->index)
      return;
    else {
      /* TODO: support variable index granularity! 
      Ideally this should be readable from stream->index->granularity, or similar
      Allowing this number to be >>1 is extremely beneficial */
      index_granularity = 1;
      /* TODO: support more types
      current implementation only supports OpenMP decompression with an offset table */
      if (stream->index->type != zfp_index_offset)
        return;
    }
  }
  const uint chunks = (blocks + index_granularity - 1) / index_granularity;

  /* allocate per-thread streams */
  bitstream** bs = decompress_init_par(stream, field, chunks, blocks);
  if (!bs)
    return;

  /* decompress chunks of blocks in parallel */
  int chunk;
  #pragma omp parallel for num_threads(threads)
  for (chunk = 0; chunk < (int)chunks; chunk++) {
    /* determine range of block indices assigned to this thread */
    const uint bmin = chunk * index_granularity;
    const uint bmax = MIN(bmin + index_granularity, blocks);
    uint block;

    /* set up thread-local bit stream */
    zfp_stream s = *stream;
    zfp_stream_set_bit_stream(&s, bs[chunk]);

    /* decode all blocks in the chunk sequentially */
    uint x;
    Scalar * block_data;

    for (block = bmin; block < bmax; block++) {
      x = block * 4;
      block_data = data + sx * x;
      if (nx - x < 4)
        _t2(zfp_decode_partial_block_strided, Scalar, 1)(&s, block_data, nx - x, 1);
      else
        _t2(zfp_decode_block_strided, Scalar, 1)(&s, block_data, sx);
    }
  }
  decompress_finish_par(bs, chunks);
  /* TODO: find a better solution
  this workaround reads a bit from the bitstream, because the bitstream pointer is checked to see if decompression was succesful */
  stream_read_bit(stream->stream);
}

/* decompress 2d strided array in parallel */
static void
_t2(decompress_strided_omp, Scalar, 2)(zfp_stream* stream, zfp_field* field)
{
  Scalar* data = (Scalar*)field->data;
  const uint nx = field->nx;
  const uint ny = field->ny;
  const int sx = field->sx ? field->sx : 1;
  const int sy = field->sy ? field->sy : nx;
  const uint threads = thread_count_omp(stream);
  const zfp_mode mode = zfp_stream_compression_mode(stream);

  /* calculate the number of blocks and chunks */
  const uint bx = (nx + 3) / 4;
  const uint by = (ny + 3) / 4;
  const uint blocks = bx * by;
  uint index_granularity = 1;
  if (mode == zfp_mode_fixed_accuracy || mode == zfp_mode_fixed_precision) {
    if (!stream->index)
      return;
    else {
      /* TODO: support variable index granularity! 
      Ideally this should be readable from stream->index->granularity, or similar
      Allowing this number to be >>1 is extremely beneficial */
      index_granularity = 1;
      /* TODO: support more types
      current implementation only supports OpenMP decompression with an offset table */
      if (stream->index->type != zfp_index_offset)
        return;
    }
  }
  const uint chunks = (blocks + index_granularity - 1) / index_granularity;

  /* allocate per-thread streams */
  bitstream** bs = decompress_init_par(stream, field, chunks, blocks);
  if (!bs)
    return;

  /* decompress chunks of blocks in parallel */
  int chunk;
  #pragma omp parallel for num_threads(threads)
  for (chunk = 0; chunk < (int)chunks; chunk++) {
    /* determine range of block indices assigned to this thread */
    const uint bmin = chunk * index_granularity;
    const uint bmax = MIN(bmin + index_granularity, blocks);
    uint block;

    /* set up thread-local bit stream */
    zfp_stream s = *stream;
    zfp_stream_set_bit_stream(&s, bs[chunk]);

    /* decode all blocks in the chunk sequentially */
    uint x, y;
    Scalar * block_data;

    for (block = bmin; block < bmax; block++) {
      x = 4 * (block % bx);
      y = 4 * (block / bx);
      block_data = data + y * sy + x * sx;
      if (nx - x < 4 || ny - y < 4)
        _t2(zfp_decode_partial_block_strided, Scalar, 2)(&s, block_data, MIN(nx - x, 4u), MIN(ny - y, 4u), sx, sy);
      else
        _t2(zfp_decode_block_strided, Scalar, 2)(&s, block_data, sx, sy);
    }
  }
  decompress_finish_par(bs, chunks);
  /* TODO: find a better solution
  this workaround reads a bit from the bitstream, because the bitstream pointer is checked to see if decompression was succesful */
  stream_read_bit(stream->stream);
}

/* same function as in serial decompression for enabling a copy (when non aligned blocks are there) */
#if defined(IPP_OPTIMIZATION_ENABLED) && !defined(_SET_TMP_BLOCK_TO_)
#define _SET_TMP_BLOCK_TO_
static void CopyToPartialBlock(Ipp32f *pDst, int stepY, int stepZ, int sizeX, int sizeY, int sizeZ, const Ipp32f *pTmpBlock)
{
    int       x, y, z;
    for(z = 0; z < sizeZ; z++)
        for(y = 0; y < sizeY; y++)
            for (x = 0; x < sizeX; x++)
            {
                int idx = x + stepY * y + stepZ * z;
                pDst[idx] = pTmpBlock[x + 4 * y + 4 * 4 * z];
            }
}
#endif

/* decompress 3d strided array in parallel */
static void
_t2(decompress_strided_omp, Scalar, 3)(zfp_stream* stream, zfp_field* field)
{
  Scalar* data = (Scalar*)field->data;
  const uint nx = field->nx;
  const uint ny = field->ny;
  const uint nz = field->nz;
  const int sx = field->sx ? field->sx : 1;
  const int sy = field->sy ? field->sy : nx;
  const int sz = field->sz ? field->sz : nx * ny;
  const uint threads = thread_count_omp(stream);
  const zfp_mode mode = zfp_stream_compression_mode(stream);

  /* calculate the number of blocks and chunks */
  const uint bx = (nx + 3) / 4;
  const uint by = (ny + 3) / 4;
  const uint bz = (nz + 3) / 4;
  const uint blocks = bx * by * bz;
  uint index_granularity = 1; /* @aniruddha - does not support index granularity */

  /* TODO: other zfp decompress modes except fixed precision + fixed accuracy */
  if (mode == zfp_mode_fixed_accuracy || mode == zfp_mode_fixed_precision) {
    if (!stream->index)
      return;
    else {
      /* TODO: support variable index granularity! 
      Ideally this should be readable from stream->index->granularity, or similar
      Allowing this number to be >>1 is extremely beneficial */
      index_granularity = 1;
      /* TODO: support more types
      current implementation only supports OpenMP decompression with an offset table */
      if (stream->index->type != zfp_index_offset)
        return;
    }
  }

  /* @aniruddha - chunk calculation is not the same as it was for ompcompression */
  const uint chunks = (blocks + index_granularity - 1) / index_granularity;

  /* allocate per-thread streams */
  bitstream** bs = decompress_init_par(stream, field, chunks, blocks);
  // if (!bs)
  //   return;

#if defined (IPP_OPTIMIZATION_ENABLED)
  IppDecodeZfpState_32f* pStates = NULL;
  Ipp64u* chunk_bit_lengths = (Ipp64u*)malloc(sizeof(Ipp64u)* chunks);
  int srcBlockLineStep = nx * sizeof(Ipp32f);
  int srcBlockPlaneStep = ny * srcBlockLineStep;
  uint min_bits, max_bits, max_prec;
  int min_exp;
  int sizeState = 0;
  if (!(REVERSIBLE(stream)))
  {
    zfp_stream_params(stream, &min_bits, &max_bits, &max_prec, &min_exp);
    ippsDecodeZfpGetStateSize_32f(&sizeState);
    pStates = (IppDecodeZfpState_32f*)ippsMalloc_8u(sizeState * threads);
  }
#endif

  /* decompress chunks of blocks in parallel */
  int chunk;

#if !defined (IPP_OPTIMIZATION_ENABLED)
  #pragma omp parallel for num_threads(threads)
#else
#pragma omp parallel num_threads(threads)
{
  bitstream *pBitStream = NULL;
  IppDecodeZfpState_32f* pState = NULL;
  Ipp32f pTmpBlock[64];
  if (!(REVERSIBLE(stream)))
  {
    //int bytesPerChunk = stream->maxbits >> 3;
    int threadIndex = omp_get_thread_num();
    pState = (IppDecodeZfpState_32f*)((Ipp8u*)pStates + threadIndex * sizeState);
  }
  #pragma omp for
#endif
  for (chunk = 0; chunk < (int)chunks; chunk++) {
    /* determine range of block indices assigned to this thread */
    const uint bmin = chunk * index_granularity;
    const uint bmax = MIN(bmin + index_granularity, blocks);
    uint block;

    /* set up thread-local bit stream */
    zfp_stream s = *stream;
    zfp_stream_set_bit_stream(&s, bs[chunk]);

#if defined (IPP_OPTIMIZATION_ENABLED)
    if (!(REVERSIBLE(stream)))
    {
      pBitStream = bs[chunk];
      int bytesPerChunk = (stream->maxbits >> 3) * index_granularity;
      Ipp8u* pInData =(Ipp8u*)stream_data(stream->stream) + chunk * bytesPerChunk;
      //Ipp8u* pInData =stream_data(pBitStream);
      //int bytesPerChunk =stream_capacity(pBitStream);
      ippsDecodeZfpInit_32f(pInData, bytesPerChunk, pState);
      ippsDecodeZfpSet_32f(min_bits, max_bits, max_prec, min_exp, pState);
    }
#endif

    /* decode all blocks in the chunk sequentially */
    uint x, y, z;
    Scalar * block_data;
    IppStatus status;

    for (block = bmin; block < bmax; block++) {
      x = 4 * (block % bx);
      y = 4 * ((block / bx) % by);
      z = 4 * (block / (bx * by));
      block_data = data + x * sx + y * sy + z * sz;
      if (nx - x < 4 || ny - y < 4 || nz - z < 4)
      {
        #if !defined(IPP_OPTIMIZATION_ENABLED)
            _t2(zfp_decode_partial_block_strided, Scalar, 3)(&s, block_data, MIN(nx - x, 4u), MIN(ny - y, 4u), MIN(nz - z, 4u), sx, sy, sz);
        #else
          if (!(REVERSIBLE(stream)))
          {
            status = ippsDecodeZfp444_32f(pState, (Ipp32f*)pTmpBlock, 4 * sizeof(Ipp32f), 4 * 4 * sizeof(Ipp32f));
            CopyToPartialBlock((Ipp32f *)block_data, sy, sz, MIN(nx - x, 4u), MIN(ny - y, 4u), MIN(nz - z, 4u), (const Ipp32f*)pTmpBlock);
          }
          else
          {
            _t2(zfp_decode_partial_block_strided, Scalar, 3)(&s, block_data, MIN(nx - x, 4u), MIN(ny - y, 4u), MIN(nz - z, 4u), sx, sy, sz);
          }
        #endif
      }
      else
            {
        #if !defined(IPP_OPTIMIZATION_ENABLED)
          _t2(zfp_decode_block_strided, Scalar, 3)(&s, block_data, sx, sy, sz);
        #else
          if (!(REVERSIBLE(stream)))
          {
            status = ippsDecodeZfp444_32f(pState, (Ipp32f *)block_data, srcBlockLineStep, srcBlockPlaneStep);
          }
          else 
          {
            _t2(zfp_decode_block_strided, Scalar, 3)(&s, block_data, sx, sy, sz);
          }
        #endif        
      }
    } /* block loop ends */
// #if defined(IPP_OPTIMIZATION_ENABLED)
//     if (!(REVERSIBLE(stream)) && pState != NULL)
//     {
//       Ipp64u chunk_decompr_length;
//       //ippsDecodeZfpGetDecompressedBitSize_32f(pState, &chunk_bit_lengths[chunk]);
//       ippsDecodeZfpGetDecompressedSize_32f(pState, &chunk_bit_lengths[chunk]);
//       ippsFree(pState);
//       chunk_decompr_length = (size_t)((chunk_bit_lengths[chunk] + 7) >> 3);
//       stream_set_eos(pBitStream, chunk_decompr_length);
//     }
// #endif    
  } /* chunk loop ends */
#if defined (IPP_OPTIMIZATION_ENABLED)
} /* The end of pragma omp parallel block */
//free(chunk_bit_lengths);
ippsFree(pStates);
#endif
  decompress_finish_par(bs, chunks);
  /* TODO: find a better solution
  this workaround reads a bit from the bitstream, because the bitstream pointer is checked to see if decompression was succesful */
  stream_read_bit(stream->stream);
}

/* decompress 4d strided array in parallel */
static void
_t2(decompress_strided_omp, Scalar, 4)(zfp_stream* stream, zfp_field* field)
{
  Scalar* data = (Scalar*)field->data;
  uint nx = field->nx;
  uint ny = field->ny;
  uint nz = field->nz;
  uint nw = field->nw;
  int sx = field->sx ? field->sx : 1;
  int sy = field->sy ? field->sy : nx;
  int sz = field->sz ? field->sz : (ptrdiff_t)nx * ny;
  int sw = field->sw ? field->sw : (ptrdiff_t)nx * ny * nz;
  const uint threads = thread_count_omp(stream);
  const zfp_mode mode = zfp_stream_compression_mode(stream);

  /* calculate the number of blocks and chunks */
  const uint bx = (nx + 3) / 4;
  const uint by = (ny + 3) / 4;
  const uint bz = (nz + 3) / 4;
  const uint bw = (nw + 3) / 4;
  const uint blocks = bx * by * bz * bw;
  uint index_granularity = 1;
  if (mode == zfp_mode_fixed_accuracy || mode == zfp_mode_fixed_precision) {
    if (!stream->index)
      return;
    else {
      /* TODO: support variable index granularity! 
      Ideally this should be readable from stream->index->granularity, or similar
      Allowing this number to be >>1 is extremely beneficial */
      index_granularity = 1;
      /* TODO: support more types
      current implementation only supports OpenMP decompression with an offset table */
      if (stream->index->type != zfp_index_offset)
        return;
    }
  }
  const uint chunks = (blocks + index_granularity - 1) / index_granularity;

  /* allocate per-thread streams */
  bitstream** bs = decompress_init_par(stream, field, chunks, blocks);
  if (!bs)
    return;

  /* decompress chunks of blocks in parallel */
  int chunk;
  #pragma omp parallel for num_threads(threads)
  for (chunk = 0; chunk < (int)chunks; chunk++) {
    /* determine range of block indices assigned to this thread */
    const uint bmin = chunk * index_granularity;
    const uint bmax = MIN(bmin + index_granularity, blocks);
    uint block;

    /* set up thread-local bit stream */
    zfp_stream s = *stream;
    zfp_stream_set_bit_stream(&s, bs[chunk]);

    /* decode all blocks in the chunk sequentially */
    uint x, y, z, w;
    Scalar * block_data;

    for (block = bmin; block < bmax; block++) {
      x = 4 * (block % bx);
      y = 4 * ((block / bx) % by);
      z = 4 * ((block / (bx * by)) % bz);
      w = 4 * (block / (bx * by * bz));
      block_data = data + x * sx + y * sy + z * sz + sw * w;
      if (nx - x < 4 || ny - y < 4 || nz - z < 4 || nw - w < 4)
        _t2(zfp_decode_partial_block_strided, Scalar, 4)(&s, block_data, MIN(nx - x, 4u), MIN(ny - y, 4u), MIN(nz - z, 4u), MIN(nw - w, 4u), sx, sy, sz, sw);
      else
        _t2(zfp_decode_block_strided, Scalar, 4)(&s, block_data, sx, sy, sz, sw);
    }
  }
  decompress_finish_par(bs, chunks);
  /* TODO: find a better solution
  this workaround reads a bit from the bitstream, because the bitstream pointer is checked to see if decompression was succesful */
  stream_read_bit(stream->stream);
}

#endif