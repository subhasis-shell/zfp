#include <assert.h>

#include "cuZFP.h"

#include "encode1.cuh"
#include "encode2.cuh"
#include "encode3.cuh"

#include "decode1.cuh"
#include "decode2.cuh"
#include "decode3.cuh"

#include "ErrorCheck.h"

#include "pointers.cuh"
#include "type_info.cuh"
#include <iostream>
#include <assert.h>

// we need to know about bitstream, but we don't 
// want duplicate symbols.
#ifndef inline_
  #define inline_ inline
#endif

#include "../inline/bitstream.c"

namespace internal 
{ 
  
bool is_contigous3d(const uint dims[3], const int3 &stride, long long int &offset)
{
  typedef long long int int64;
  int64 idims[3];
  idims[0] = dims[0];
  idims[1] = dims[1];
  idims[2] = dims[2];

  int64 imin = std::min(stride.x,0) * (idims[0] - 1) + 
               std::min(stride.y,0) * (idims[1] - 1) + 
               std::min(stride.z,0) * (idims[2] - 1);

  int64 imax = std::max(stride.x,0) * (idims[0] - 1) + 
               std::max(stride.y,0) * (idims[1] - 1) + 
               std::max(stride.z,0) * (idims[2] - 1);
  offset = imin;
  int64 ns = idims[0] * idims[1] * idims[2];

  return (imax - imin + 1 == ns);
}

bool is_contigous2d(const uint dims[3], const int3 &stride, long long int &offset)
{
  typedef long long int int64;
  int64 idims[2];
  idims[0] = dims[0];
  idims[1] = dims[1];

  int64 imin = std::min(stride.x,0) * (idims[0] - 1) + 
               std::min(stride.y,0) * (idims[1] - 1);

  int64  imax = std::max(stride.x,0) * (idims[0] - 1) + 
                std::max(stride.y,0) * (idims[1] - 1); 

  offset = imin;
  return (imax - imin + 1) == (idims[0] * idims[1]);
}

bool is_contigous1d(uint dim, const int &stride, long long int &offset)
{
  offset = 0;
  if(stride < 0) offset = stride * (int(dim) - 1);
  return std::abs(stride) == 1;
}

bool is_contigous(const uint dims[3], const int3 &stride, long long int &offset)
{
  int d = 0;
  
  if(dims[0] != 0) d++;
  if(dims[1] != 0) d++;
  if(dims[2] != 0) d++;

  if(d == 3)
  {
    return is_contigous3d(dims, stride, offset);
  }
  else if(d == 2)
  {
   return is_contigous2d(dims, stride, offset);
  }
  else
  {
    return is_contigous1d(dims[0], stride.x, offset);
  } 

}
//
// encode expects device pointers
//
template<typename T>
size_t encode(uint dims[3], int3 stride, int bits_per_block, T *d_data, Word *d_stream)
{

  int d = 0;
  size_t len = 1;
  for(int i = 0; i < 3; ++i)
  {
    if(dims[i] != 0)
    {
      d++;
      len *= dims[i];
    }
  }

  ErrorCheck errors;
  size_t stream_size = 0;
  if(d == 1)
  {
    int dim = dims[0];
    int sx = stride.x;
    stream_size = cuZFP::encode1<T>(dim, sx, d_data, d_stream, bits_per_block); 
  }
  else if(d == 2)
  {
    uint2 ndims = make_uint2(dims[0], dims[1]);
    int2 s;
    s.x = stride.x; 
    s.y = stride.y; 
    stream_size = cuZFP::encode2<T>(ndims, s, d_data, d_stream, bits_per_block); 
  }
  else if(d == 3)
  {
    int3 s;
    s.x = stride.x; 
    s.y = stride.y; 
    s.z = stride.z; 
    uint3 ndims = make_uint3(dims[0], dims[1], dims[2]);
    stream_size = cuZFP::encode3<T>(ndims, s, d_data, d_stream, bits_per_block); 
  }

  errors.chk("Encode");
  
  return stream_size; 
}

template<typename T>
size_t decode(uint ndims[3], int3 stride, int bits_per_block, Word *stream, T *out)
{

  int d = 0;
  size_t out_size = 1;
  size_t stream_bytes = 0;
  for(int i = 0; i < 3; ++i)
  {
    if(ndims[i] != 0)
    {
      d++;
      out_size *= ndims[i];
    }
  }

  if(d == 3)
  {
    uint3 dims = make_uint3(ndims[0], ndims[1], ndims[2]);

    int3 s;
    s.x = stride.x; 
    s.y = stride.y; 
    s.z = stride.z; 

    stream_bytes = cuZFP::decode3<T>(dims, s, stream, out, bits_per_block); 
  }
  else if(d == 1)
  {
    uint dim = ndims[0];
    int sx = stride.x;

    stream_bytes = cuZFP::decode1<T>(dim, sx, stream, out, bits_per_block); 

  }
  else if(d == 2)
  {
    uint2 dims;
    dims.x = ndims[0];
    dims.y = ndims[1];

    int2 s;
    s.x = stride.x; 
    s.y = stride.y; 

    stream_bytes = cuZFP::decode2<T>(dims, s, stream, out, bits_per_block); 
  }
  else std::cerr <<" d ==  "<<d<<" not implemented\n";
 
  return stream_bytes;
}


// Implementation for stream: additional functions

// encode expects device pointers
//
template<typename T>
size_t encodestream(uint dims[3], int3 stride, int bits_per_block, T *d_data, Word *d_stream,
              cudaStream_t custream)
{

  int d = 0;
  size_t len = 1;
  for(int i = 0; i < 3; ++i)
  {
    if(dims[i] != 0)
    {
      d++;
      len *= dims[i];
    }
  }

  ErrorCheck errors;
  size_t stream_size = 0;

  if(d == 3)
  {
    int3 s;
    s.x = stride.x; 
    s.y = stride.y; 
    s.z = stride.z; 
    uint3 ndims = make_uint3(dims[0], dims[1], dims[2]);
    stream_size = cuZFP::encode3stream<T>(ndims, s, d_data, d_stream, bits_per_block, custream); 
  }
  else {
    fprintf(stderr, "Only 3-D is supported in CUDA stream\n");
    exit(0);
  }

  errors.chk("Encode");
  
  return stream_size; 
}

template<typename T>
size_t decodestream(uint ndims[3], int3 stride, int bits_per_block, Word *stream, T *out, 
                    cudaStream_t custream)
{

  int d = 0;
  size_t out_size = 1;
  size_t stream_bytes = 0;
  for(int i = 0; i < 3; ++i)
  {
    if(ndims[i] != 0)
    {
      d++;
      out_size *= ndims[i];
    }
  }

  if(d == 3)
  {
    uint3 dims = make_uint3(ndims[0], ndims[1], ndims[2]);

    int3 s;
    s.x = stride.x; 
    s.y = stride.y; 
    s.z = stride.z; 

    std::cout << "Calling cuZFP decode3 CUDA stream" << std::endl;
    stream_bytes = cuZFP::decode3stream<T>(dims, s, stream, out, bits_per_block, custream); 
  }
  else 
    std::cerr <<" d ==  "<<d<<" is not implemented in CUDA stream \n";
 
  return stream_bytes;
}


Word *setup_device_stream_compress(zfp_stream *stream, const zfp_field *field)
{
  bool stream_device = cuZFP::is_gpu_ptr(stream->stream->begin);
  assert(sizeof(word) == sizeof(Word)); // "CUDA version currently only supports 64bit words");

  if(stream_device)
  {
    return (Word*) stream->stream->begin;
  }

  Word *d_stream = NULL;
  size_t max_size = zfp_stream_maximum_size(stream, field);
   
  if (field->cuda_malloc_func) {
    checkCudaError(field->cuda_malloc_func((void**) &d_stream, max_size));
  }
  else {
    checkCudaError(cudaMalloc(&d_stream, max_size));
  }

  // fprintf(stderr, "Compress - device stream - place compressed data = %p\n",
  //        d_stream);
  
  return d_stream;
}

Word *setup_device_stream_decompress(zfp_stream *stream,const zfp_field *field)
{
  bool stream_device = cuZFP::is_gpu_ptr(stream->stream->begin);
  assert(sizeof(word) == sizeof(Word)); // "CUDA version currently only supports 64bit words");

  if(stream_device)
  {
    return (Word*) stream->stream->begin;
  }

  Word *d_stream = NULL;
  //TODO: change maximum_size to compressed stream size
  size_t size = zfp_stream_maximum_size(stream, field);
  if (field->cuda_malloc_func) {
    checkCudaError(field->cuda_malloc_func((void**) &d_stream, size));
  }
  else {
    checkCudaError(cudaMalloc(&d_stream, size));
  }
  
  // fprintf(stderr, "Print from cuZFP = size = %d, dest addr = %p, address src comp = %p\n",
  //        size, d_stream, stream->stream->begin);
  checkCudaError(cudaMemcpy(d_stream, stream->stream->begin, size, cudaMemcpyHostToDevice));
  return d_stream;
}

void * offset_void(zfp_type type, void *ptr, long long int offset)
{
  void * offset_ptr = NULL;
  if(type == zfp_type_float)
  {
    float* data = (float*) ptr;
    offset_ptr = (void*)(&data[offset]);
  }
  else if(type == zfp_type_double)
  {
    double* data = (double*) ptr;
    offset_ptr = (void*)(&data[offset]);
  }
  else if(type == zfp_type_int32)
  {
    int * data = (int*) ptr;
    offset_ptr = (void*)(&data[offset]);
  }
  else if(type == zfp_type_int64)
  {
    long long int * data = (long long int*) ptr;
    offset_ptr = (void*)(&data[offset]);
  }

  return offset_ptr;

}

void *setup_device_field_compress(const zfp_field *field, const int3 &stride, long long int &offset)
{
  bool field_device = cuZFP::is_gpu_ptr(field->data);

  if(field_device)
  {
    offset = 0;
    return field->data;
  }
  
  uint dims[3];
  dims[0] = field->nx;
  dims[1] = field->ny;
  dims[2] = field->nz;

  size_t type_size = zfp_type_size(field->type);

  size_t field_size = 1;
  for(int i = 0; i < 3; ++i)
  {
    if(dims[i] != 0)
    {
      field_size *= dims[i];
    }
  }

  bool contig = internal::is_contigous(dims, stride, offset);
  
  void * host_ptr = offset_void(field->type, field->data, offset);

  void *d_data = NULL;

  size_t field_bytes = type_size * field_size;
  
  if(contig)
  {
     if (field->cuda_malloc_func) {
       checkCudaError(field->cuda_malloc_func((void**) &d_data, field_bytes));
     }
     else {
       checkCudaError(cudaMalloc(&d_data, field_bytes));
     }
    checkCudaError(cudaMemcpy(d_data, host_ptr, field_bytes, cudaMemcpyHostToDevice));
  }
  
  return offset_void(field->type, d_data, -offset);
}

void *setup_device_field_decompress(const zfp_field *field, const int3 &stride, long long int &offset)
{
  bool field_device = cuZFP::is_gpu_ptr(field->data);

  if(field_device)
  {
    offset = 0;
    return field->data;
  }

  uint dims[3];
  dims[0] = field->nx;
  dims[1] = field->ny;
  dims[2] = field->nz;

  size_t type_size = zfp_type_size(field->type);

  size_t field_size = 1;
  for(int i = 0; i < 3; ++i)
  {
    if(dims[i] != 0)
    {
      field_size *= dims[i];
    }
  }

  bool contig = internal::is_contigous(dims, stride, offset);

  void *d_data = NULL;
  if(contig)
  {
    size_t field_bytes = type_size * field_size;
  
    if (field->cuda_malloc_func) {
      checkCudaError(field->cuda_malloc_func((void**) &d_data, field_bytes));
    }
    else {
      checkCudaError(cudaMalloc(&d_data, field_bytes));
    }
  }

  return offset_void(field->type, d_data, -offset);

}

void cleanup_device_ptr(const zfp_field *field, void *orig_ptr, void *d_ptr, 
                        size_t bytes, long long int offset, zfp_type type)
{
  bool device = cuZFP::is_gpu_ptr(orig_ptr);
  if(device)
  {
    return;
  }
  // from whence it came
  void *d_offset_ptr = offset_void(type, d_ptr, offset);
  void *h_offset_ptr = offset_void(type, orig_ptr, offset);

  if(bytes > 0)
  {
    checkCudaError(cudaMemcpy(h_offset_ptr, d_offset_ptr, bytes, cudaMemcpyDeviceToHost));
  }

   if (field && field->cuda_free_func) {
     checkCudaError(field && field->cuda_free_func(d_offset_ptr));
   }
   else {
     checkCudaError(cudaFree(d_offset_ptr));
   }

}

} // namespace internal

size_t
cuda_compress(zfp_stream *stream, const zfp_field *field)
{
  uint dims[3];
  dims[0] = field->nx;
  dims[1] = field->ny;
  dims[2] = field->nz;

  int3 stride;  
  stride.x = field->sx ? field->sx : 1;
  stride.y = field->sy ? field->sy : field->nx;
  stride.z = field->sz ? field->sz : field->nx * field->ny;
  
  size_t stream_bytes = 0;
  long long int offset = 0; 
  void *d_data = internal::setup_device_field_compress(field, stride, offset);

  if(d_data == NULL)
  {
    // null means the array is non-contiguous host mem which is not supported
    return 0;
  }

  Word *d_stream = internal::setup_device_stream_compress(stream, field);

  if(field->type == zfp_type_float)
  {
    float* data = (float*) d_data;
    stream_bytes = internal::encodestream<float>(dims, stride, (int)stream->maxbits, 
                                                 data, d_stream, field->cuStream);
  }
  else if(field->type == zfp_type_double)
  {
    double* data = (double*) d_data;
    stream_bytes = internal::encodestream<double>(dims, stride, (int)stream->maxbits, 
                                                  data, d_stream, field->cuStream);
  }
  else if(field->type == zfp_type_int32)
  {
    int * data = (int*) d_data;
    stream_bytes = internal::encodestream<int>(dims, stride, (int)stream->maxbits, 
                                               data, d_stream, field->cuStream);
  }
  else if(field->type == zfp_type_int64)
  {
    long long int * data = (long long int*) d_data;
    stream_bytes = internal::encodestream<long long int>(dims, stride, (int)stream->maxbits, 
                                                         data, d_stream, field->cuStream);
  }

  internal::cleanup_device_ptr(field, stream->stream->begin, d_stream, stream_bytes, 0, field->type);
  internal::cleanup_device_ptr(field, field->data, d_data, 0, offset, field->type);

  // zfp wants to flush the stream.
  // set bits to wsize because we already did that.
  size_t compressed_size = stream_bytes / sizeof(Word);
  stream->stream->bits = wsize;
  // set stream pointer to end of stream
  stream->stream->ptr = stream->stream->begin + compressed_size;

  return stream_bytes;
}
  
void 
cuda_decompress(zfp_stream *stream, zfp_field *field)
{
  uint dims[3];
  dims[0] = field->nx;
  dims[1] = field->ny;
  dims[2] = field->nz;
   
  int3 stride;  
  stride.x = field->sx ? field->sx : 1;
  stride.y = field->sy ? field->sy : field->nx;
  stride.z = field->sz ? field->sz : field->nx * field->ny;

  size_t decoded_bytes = 0;
  long long int offset = 0;
  void *d_data = internal::setup_device_field_decompress(field, stride, offset);
  
  if(d_data == NULL)
  {
    // null means the array is non-contiguous host mem which is not supported
    return;
  }

  Word *d_stream = internal::setup_device_stream_decompress(stream, field);

  if(field->type == zfp_type_float)
  {
    float *data = (float*) d_data;
    decoded_bytes = internal::decodestream(dims, stride, (int)stream->maxbits, d_stream, 
                                           data, field->cuStream);
    d_data = (void*) data;
  }

  else if(field->type == zfp_type_double)
  {
    double *data = (double*) d_data;
    decoded_bytes = internal::decodestream(dims, stride, (int)stream->maxbits, d_stream, 
                                           data, field->cuStream);
    d_data = (void*) data;
  }
  else if(field->type == zfp_type_int32)
  {
    int *data = (int*) d_data;
    decoded_bytes = internal::decodestream(dims, stride, (int)stream->maxbits, d_stream, 
                                           data, field->cuStream);
    d_data = (void*) data;
  }
  else if(field->type == zfp_type_int64)
  {
    long long int *data = (long long int*) d_data;
    decoded_bytes = internal::decodestream(dims, stride, (int)stream->maxbits, d_stream, 
                                           data, field->cuStream);
    d_data = (void*) data;
  }
  else
  {
    std::cerr<<"Cannot decompress: type unknown\n";
  }

   
  size_t type_size = zfp_type_size(field->type);

  size_t field_size = 1;
  for(int i = 0; i < 3; ++i)
  {
    if(dims[i] != 0)
    {
      field_size *= dims[i];
    }
  }
  
  size_t bytes = type_size * field_size;
  internal::cleanup_device_ptr(field, stream->stream->begin, d_stream, 0, 0, field->type);
  internal::cleanup_device_ptr(field, field->data, d_data, bytes, offset, field->type);
  
  // this is how zfp determins if this was a success
  size_t words_read = decoded_bytes / sizeof(Word);
  stream->stream->bits = wsize;
  // set stream pointer to end of stream
  stream->stream->ptr = stream->stream->begin + words_read;
}

// Exposing CUDA compress and decompress API

size_t zfpEncodeGpu(zfp_stream *zpstream, zfp_field *field,
                           ext_zfp_field *exfield)
{
  uint dims[3];
  dims[0] = field->nx;
  dims[1] = field->ny;
  dims[2] = field->nz;

  int3 stride;  
  stride.x = field->sx ? field->sx : 1;
  stride.y = field->sy ? field->sy : field->nx;
  stride.z = field->sz ? field->sz : field->nx * field->ny;

  size_t stream_bytes = 0;
  
  if(field->type == zfp_type_float) {
    float *local_data = (float*) exfield->device_data;
    stream_bytes = internal::encode<float>(dims, stride, (int)zpstream->maxbits, 
                  local_data, exfield->device_stream);
  }
  else if(field->type == zfp_type_double) {
    double* local_data = (double*) exfield->device_data;
    stream_bytes = internal::encode<double>(dims, stride, (int)zpstream->maxbits, 
                   local_data, exfield->device_stream);
  }
  else if(field->type == zfp_type_int32) {
    int * local_data = (int*) exfield->device_data;
    stream_bytes = internal::encode<int>(dims, stride, (int)zpstream->maxbits, 
                   local_data, exfield->device_stream);
  }
  else if(field->type == zfp_type_int64) {
    long long int * local_data = (long long int*) exfield->device_data;
    stream_bytes = internal::encode<long long int>(dims, stride, (int)zpstream->maxbits, 
                   local_data, exfield->device_stream);
  }
  else
  {
    std::cerr<<"Cannot decompress: type unknown\n";
  }

  // zfp wants to flush the stream.
  // set bits to wsize because we already did that.
  
  size_t compressed_size = stream_bytes / sizeof(Word);
  zpstream->stream->bits = wsize;
  
  // set stream pointer to end of stream
  zpstream->stream->ptr = zpstream->stream->begin + compressed_size;
  return stream_bytes;

}

size_t zfpDecodeGpu(zfp_stream *zpstream, zfp_field *field,
                    ext_zfp_field *exfield)
{
  uint dims[3];
  dims[0] = field->nx;
  dims[1] = field->ny;
  dims[2] = field->nz;

  int3 stride;  
  stride.x = field->sx ? field->sx : 1;
  stride.y = field->sy ? field->sy : field->nx;
  stride.z = field->sz ? field->sz : field->nx * field->ny;

  size_t decoded_bytes = 0;
  
  if(field->type == zfp_type_float) {
    float *local_data = (float*) exfield->device_data;
    decoded_bytes = internal::decode<float>(dims, stride, (int)zpstream->maxbits, 
                    exfield->device_stream, local_data);
    exfield->device_data = (void*) local_data;
  }
  else if(field->type == zfp_type_double) {
    double* local_data = (double*) exfield->device_data;
    decoded_bytes = internal::decode<double>(dims, stride, (int)zpstream->maxbits, 
                    exfield->device_stream, local_data);
    exfield->device_data = (void*) local_data;
  }
  else if(field->type == zfp_type_int32) {
    int * local_data = (int*) exfield->device_data;
    decoded_bytes = internal::decode<int>(dims, stride, (int)zpstream->maxbits, 
                   exfield->device_stream, local_data);
    exfield->device_data = (void*) local_data;
  }
  else if(field->type == zfp_type_int64) {
    long long int * local_data = (long long int*) exfield->device_data;
    decoded_bytes = internal::decode<long long int>(dims, stride, (int)zpstream->maxbits, 
                   exfield->device_stream, local_data);
    exfield->device_data = (void*) local_data;
  }
  else
  {
    std::cerr<<"Cannot decompress: type unknown\n";
  }
   
  size_t type_size = zfp_type_size(field->type);

  size_t field_size = 1;
  for(int i = 0; i < 3; ++i)
  {
    if(dims[i] != 0)
    {
      field_size *= dims[i];
    }
  }
 
  // this is how zfp determins if this was a success
  size_t words_read = decoded_bytes / sizeof(Word);
  zpstream->stream->bits = wsize;
  // set stream pointer to end of stream
  zpstream->stream->ptr = zpstream->stream->begin + words_read;

  return(stream_size(zpstream->stream)); 
}

// CUDA compress and decompress API with Streams

size_t zfpEncodeGpuStream(zfp_stream *zpstream, zfp_field *field,
                          ext_zfp_field *exfield, cudaStream_t custream)
{
  uint dims[3];
  dims[0] = field->nx;
  dims[1] = field->ny;
  dims[2] = field->nz;

  int3 stride;  
  stride.x = field->sx ? field->sx : 1;
  stride.y = field->sy ? field->sy : field->nx;
  stride.z = field->sz ? field->sz : field->nx * field->ny;

  size_t stream_bytes = 0;
  
  if(field->type == zfp_type_float) {
    float *local_data = (float*) exfield->device_data;
    stream_bytes = internal::encodestream<float>(dims, stride, (int)zpstream->maxbits, 
                  local_data, exfield->device_stream, custream);
  }
  else if(field->type == zfp_type_double) {
    double* local_data = (double*) exfield->device_data;
    stream_bytes = internal::encodestream<double>(dims, stride, (int)zpstream->maxbits, 
                   local_data, exfield->device_stream, custream);
  }
  else if(field->type == zfp_type_int32) {
    int * local_data = (int*) exfield->device_data;
    stream_bytes = internal::encodestream<int>(dims, stride, (int)zpstream->maxbits, 
                   local_data, exfield->device_stream, custream);
  }
  else if(field->type == zfp_type_int64) {
    long long int * local_data = (long long int*) exfield->device_data;
    stream_bytes = internal::encodestream<long long int>(dims, stride, (int)zpstream->maxbits, 
                   local_data, exfield->device_stream, custream);
  }
  else
  {
    std::cerr<<"Cannot decompress: type unknown\n";
  }

  // zfp wants to flush the stream.
  // set bits to wsize because we already did that.
  
  size_t compressed_size = stream_bytes / sizeof(Word);
  zpstream->stream->bits = wsize;
  
  // set stream pointer to end of stream
  zpstream->stream->ptr = zpstream->stream->begin + compressed_size;
  return stream_bytes;

}

size_t zfpDecodeGpuStream(zfp_stream *zpstream, zfp_field *field,
                          ext_zfp_field *exfield, cudaStream_t custream)
{
  uint dims[3];
  dims[0] = field->nx;
  dims[1] = field->ny;
  dims[2] = field->nz;

  int3 stride;  
  stride.x = field->sx ? field->sx : 1;
  stride.y = field->sy ? field->sy : field->nx;
  stride.z = field->sz ? field->sz : field->nx * field->ny;

  size_t decoded_bytes = 0;
  
  if(field->type == zfp_type_float) {
    float *local_data = (float*) exfield->device_data;
    decoded_bytes = internal::decodestream<float>(dims, stride, (int)zpstream->maxbits, 
                    exfield->device_stream, local_data, custream);
    exfield->device_data = (void*) local_data;
  }
  else if(field->type == zfp_type_double) {
    double* local_data = (double*) exfield->device_data;
    decoded_bytes = internal::decodestream<double>(dims, stride, (int)zpstream->maxbits, 
                    exfield->device_stream, local_data, custream);
    exfield->device_data = (void*) local_data;
  }
  else if(field->type == zfp_type_int32) {
    int * local_data = (int*) exfield->device_data;
    decoded_bytes = internal::decodestream<int>(dims, stride, (int)zpstream->maxbits, 
                   exfield->device_stream, local_data, custream);
    exfield->device_data = (void*) local_data;
  }
  else if(field->type == zfp_type_int64) {
    long long int * local_data = (long long int*) exfield->device_data;
    decoded_bytes = internal::decodestream<long long int>(dims, stride, (int)zpstream->maxbits, 
                   exfield->device_stream, local_data, custream);
    exfield->device_data = (void*) local_data;
  }
  else
  {
    std::cerr<<"Cannot decompress: type unknown\n";
  }
   
  size_t type_size = zfp_type_size(field->type);

  size_t field_size = 1;
  for(int i = 0; i < 3; ++i)
  {
    if(dims[i] != 0)
    {
      field_size *= dims[i];
    }
  }
 
  // this is how zfp determins if this was a success
  size_t words_read = decoded_bytes / sizeof(Word);
  zpstream->stream->bits = wsize;
  // set stream pointer to end of stream
  zpstream->stream->ptr = zpstream->stream->begin + words_read;

  return(stream_size(zpstream->stream)); 
}
