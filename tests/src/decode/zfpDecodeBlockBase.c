#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "utils/testMacros.h"
#include "utils/zfpChecksums.h"
#include "utils/zfpHash.h"

struct setupVars {
  Scalar* dataArr;
  void* buffer;
  size_t bufsizeBytes;
  zfp_stream* stream;
};

static void
populateInitialArray(Scalar** dataArrPtr)
{
  *dataArrPtr = malloc(sizeof(Scalar) * BLOCK_SIZE);
  assert_non_null(*dataArrPtr);

  int i;
  for (i = 0; i < BLOCK_SIZE; i++) {
#ifdef FL_PT_DATA
    (*dataArrPtr)[i] = nextSignedRandFlPt();
#else
    (*dataArrPtr)[i] = nextSignedRandInt();
#endif
  }

}

static void
populateInitialArraySpecial(Scalar* dataArr, int index)
{
  // IEEE-754 special values
  static const uint32 special_float_values[] = {
    0x00000000u, // +0
    0x80000000u, // -0
    0x00000001u, // +FLT_TRUE_MIN
    0x80000001u, // -FLT_TRUE_MIN
    0x7f7fffffu, // +FLT_MAX
    0xff7fffffu, // -FLT_MAX
    0x7f800000u, // +infinity
    0xff800000u, // -infinity
    0x7fc00000u, // qNaN
    0x7fa00000u, // sNaN
  };
  static const uint64 special_double_values[] = {
    UINT64C(0x0000000000000000), // +0
    UINT64C(0x8000000000000000), // -0
    UINT64C(0x0000000000000001), // +DBL_TRUE_MIN
    UINT64C(0x8000000000000001), // -DBL_TRUE_MIN
    UINT64C(0x7fefffffffffffff), // +DBL_MAX
    UINT64C(0xffefffffffffffff), // -DBL_MAX
    UINT64C(0x7ff0000000000000), // +infinity
    UINT64C(0xfff0000000000000), // -infinity
    UINT64C(0x7ff8000000000000), // qNaN
    UINT64C(0x7ff4000000000000), // sNaN
  };

  size_t i;
  for (i = 0; i < BLOCK_SIZE; i++) {
#ifdef FL_PT_DATA
    // generate special values
    if ((i & 3u) == 0) {
      switch(ZFP_TYPE) {
        case zfp_type_float:
          memcpy(dataArr + i, &special_float_values[index], sizeof(Scalar));
          break;
        case zfp_type_double:
          memcpy(dataArr + i, &special_double_values[index], sizeof(Scalar));
          break;
      }
    }
    else
      dataArr[i] = 0;
#else
    dataArr[i] = nextSignedRandInt();
#endif
  }
}

static void
setupZfpStream(struct setupVars* bundle)
{
  zfp_type type = ZFP_TYPE;
  zfp_field* field;
  switch(DIMS) {
    case 1:
      field = zfp_field_1d(bundle->dataArr, type, BLOCK_SIDE_LEN);
      break;
    case 2:
      field = zfp_field_2d(bundle->dataArr, type, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN);
      break;
    case 3:
      field = zfp_field_3d(bundle->dataArr, type, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN);
      break;
    case 4:
      field = zfp_field_4d(bundle->dataArr, type, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN);
      break;
  }

  zfp_stream* stream = zfp_stream_open(NULL);
  zfp_stream_set_rate(stream, ZFP_RATE_PARAM_BITS, type, DIMS, 0);

  size_t bufsizeBytes = zfp_stream_maximum_size(stream, field);
  char* buffer = calloc(bufsizeBytes, sizeof(char));
  assert_non_null(buffer);

  bitstream* s = stream_open(buffer, bufsizeBytes);
  assert_non_null(s);

  zfp_stream_set_bit_stream(stream, s);
  zfp_stream_rewind(stream);
  zfp_field_free(field);

  bundle->buffer = buffer;
  bundle->stream = stream;
}

static void
setupZfpStreamSpecial(struct setupVars* bundle)
{
  zfp_type type = ZFP_TYPE;
  zfp_field* field;
  switch(DIMS) {
    case 1:
      field = zfp_field_1d(bundle->dataArr, type, BLOCK_SIDE_LEN);
      break;
    case 2:
      field = zfp_field_2d(bundle->dataArr, type, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN);
      break;
    case 3:
      field = zfp_field_3d(bundle->dataArr, type, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN);
      break;
    case 4:
      field = zfp_field_4d(bundle->dataArr, type, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN);
      break;
  }

  zfp_stream* stream = zfp_stream_open(NULL);
  zfp_stream_set_reversible(stream);

  size_t bufsizeBytes = zfp_stream_maximum_size(stream, field);
  char* buffer = calloc(bufsizeBytes, sizeof(char));
  assert_non_null(buffer);

  bitstream* s = stream_open(buffer, bufsizeBytes);
  assert_non_null(s);

  zfp_stream_set_bit_stream(stream, s);
  zfp_stream_rewind(stream);
  zfp_field_free(field);

  bundle->bufsizeBytes = bufsizeBytes;
  bundle->buffer = buffer;
  bundle->stream = stream;
}

static int
setup(void **state)
{
  struct setupVars *bundle = malloc(sizeof(struct setupVars));
  assert_non_null(bundle);

  resetRandGen();
  populateInitialArray(&bundle->dataArr);
  setupZfpStream(bundle);

  *state = bundle;

  return 0;
}

static int
setupSpecial(void **state)
{
  struct setupVars *bundle = malloc(sizeof(struct setupVars));
  assert_non_null(bundle);

  bundle->dataArr = malloc(sizeof(Scalar) * BLOCK_SIZE);
  assert_non_null(bundle->dataArr);

  resetRandGen();
  setupZfpStreamSpecial(bundle);

  *state = bundle;

  return 0;
}

static int
teardown(void **state)
{
  struct setupVars *bundle = *state;

  stream_close(bundle->stream->stream);
  zfp_stream_close(bundle->stream);
  free(bundle->buffer);
  free(bundle->dataArr);
  free(bundle);

  return 0;
}

static void
when_seededRandomDataGenerated_expect_ChecksumMatches(void **state)
{
  struct setupVars *bundle = *state;
  UInt checksum = _catFunc2(hashArray, SCALAR_BITS)((const UInt*)bundle->dataArr, BLOCK_SIZE, 1);
  uint64 expectedChecksum = getChecksumOriginalDataBlock(DIMS, ZFP_TYPE);
  assert_int_equal(checksum, expectedChecksum);
}

static void
_catFunc3(given_, DIM_INT_STR, Block_when_DecodeBlock_expect_ReturnValReflectsNumBitsReadFromBitstream)(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  bitstream* s = zfp_stream_bit_stream(stream);

  _t2(zfp_encode_block, Scalar, DIMS)(stream, bundle->dataArr);
  zfp_stream_flush(stream);
  zfp_stream_rewind(stream);

  uint returnValBits = _t2(zfp_decode_block, Scalar, DIMS)(stream, bundle->dataArr);

  assert_int_equal(returnValBits, stream_rtell(s));
}

static void
_catFunc3(given_, DIM_INT_STR, Block_when_DecodeBlock_expect_ArrayChecksumMatches)(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  _t2(zfp_encode_block, Scalar, DIMS)(stream, bundle->dataArr);
  zfp_stream_flush(stream);
  zfp_stream_rewind(stream);

  Scalar* decodedDataArr = calloc(BLOCK_SIZE, sizeof(Scalar));
  assert_non_null(decodedDataArr);
  _t2(zfp_decode_block, Scalar, DIMS)(stream, decodedDataArr);

  UInt checksum = _catFunc2(hashArray, SCALAR_BITS)((const UInt*)decodedDataArr, BLOCK_SIZE, 1);
  free(decodedDataArr);

  uint64 expectedChecksum = getChecksumDecodedBlock(DIMS, ZFP_TYPE);
  assert_int_equal(checksum, expectedChecksum);
}

static void
_catFunc3(given_, DIM_INT_STR, Block_when_DecodeSpecialBlocks_expect_ArraysMatchBitForBit)(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  int failures = 0;
  int i;
  for (i = 0; i < 10; i++) {
    populateInitialArraySpecial(bundle->dataArr, i);

    _t2(zfp_encode_block, Scalar, DIMS)(stream, bundle->dataArr);
    zfp_stream_flush(stream);
    zfp_stream_rewind(stream);

    Scalar* decodedDataArr = calloc(BLOCK_SIZE, sizeof(Scalar));
    assert_non_null(decodedDataArr);
    _t2(zfp_decode_block, Scalar, DIMS)(stream, decodedDataArr);

    if (memcmp(bundle->dataArr, decodedDataArr, BLOCK_SIZE * sizeof(Scalar)) != 0) {
      printf("Decode special Block testcase %d failed\n", i);
      failures++;
    }

    free(decodedDataArr);

    // reset/zero bitstream, rewind for next iteration
    memset(bundle->buffer, bundle->bufsizeBytes, 0);
    zfp_stream_rewind(stream);
  }

  if (failures > 0) {
    fail_msg("At least 1 special block testcase failed\n");
  }
}
