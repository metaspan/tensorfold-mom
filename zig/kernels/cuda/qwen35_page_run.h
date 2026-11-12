// A block's monotone logical-key walk reads each encountered page ID once; page IDs must remain stable.
#pragma once
#include <stdint.h>

#ifdef __CUDACC__
#define TF_Q35_PAGE_INLINE __device__ __forceinline__
#else
#define TF_Q35_PAGE_INLINE static inline
#endif

typedef struct {
    const int* ids;
    int tokens;
    int logical;
    uint64_t stride;
    uint64_t base;
} TfQ35PageRun;

TF_Q35_PAGE_INLINE uint64_t tf_q35_page_base(TfQ35PageRun* run, int pos) {
    // Stateless: the MoM shared view calls two runs interleaved for the same walk
    // (shared pages via the view's skvi, tail pages via the thin's pt), and a walk
    // is not monotone when the slot scan replays pending kept rows first -- a
    // cached base then serves earlier logical pages the last page's id (2026-11-12
    // toy repro: deterministic garbage past page 0 with the audit green).
    const int logical = pos / run->tokens;
    return (uint64_t)run->ids[logical] * run->stride;
}
#undef TF_Q35_PAGE_INLINE
