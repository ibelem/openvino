// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for openvino/src/inference/src/single_file_storage.cpp:162-166:
// a ConstantMeta TLV record whose const_type byte is out of range (>= enum_types_size)
// must be rejected by build_content_index instead of being stored as an invalid Type_t.
// Pre-fix: the invalid enum is stored and later get_type_info() (element_type.cpp:131) throws/UB.
// Post-fix: build_content_index returns false / the record is rejected.
//
// TODO: This needs a crafted serialized single-file cache blob (TLV stream) as a fixture.
// TODO: Confirm the exact test target for src/inference (e.g. ov_inference_unit_tests) and the
//       header that exposes SingleFileStorage::build_content_index for direct testing; it may be
//       internal, in which case drive it through the public model-cache load API with a poisoned file.
#include <gtest/gtest.h>
#include <sstream>
#include <cstdint>
// #include "single_file_storage.hpp"  // TODO: correct relative include for the internal header

TEST(SingleFileStorageConstantMeta, RejectsOutOfRangeElementType) {
    // TODO: build a minimal TLV stream containing one ConstantMeta record:
    //   source_id(u64), then [const_id(u64), const_offset(u64), const_size(u64), const_type(u8)=0x1A]
    // const_type = 26 (== enum_types_size) is the first invalid Type_t value.
    std::stringstream crafted;
    // TODO: serialize a valid header + ConstantMeta tag/length + payload above into `crafted`.

    // SingleFileStorage storage;
    // const bool ok = storage.build_content_index(crafted);
    // EXPECT_FALSE(ok) << "out-of-range const_type byte must be rejected";
    // ASSERT_TRUE(storage_weight_registry_empty(storage));
    GTEST_SKIP() << "TODO: provide crafted TLV blob fixture and internal accessor";
}
