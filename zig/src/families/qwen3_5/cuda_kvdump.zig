//! TF_MOM_KVDUMP=<dir>: the S1 (docs/12) debug writer. Every kept-state checkpoint the B5 host store
//! saves (request starts under TF_QWEN_REQUEST_STARTS=1, chunk marks otherwise) also lands its
//! full-attention K/V rows, token ids and engine identity under <dir>/<model>/ckpt<N>.bin plus a
//! ckpt<N>.json sidecar, for experiments/s1_fit.py. Debug-only: the switch is off by default, and a
//! failed dump never fails the save (cuda_reuse.zig's save wraps this in a catch). The row arithmetic,
//! header and sidecar live in the pure kvdump_format.zig (host tests: server/mom_kvdump_test.zig).
const std = @import("std");
const cuda = @import("cuda");
const engine = @import("cuda_engine.zig");
const kvdump = @import("kvdump_format.zig");

const Engine = engine.Engine;

/// The dump directory once read: TF_MOM_KVDUMP unset or empty keeps the writer off.
pub fn dumpDir() ?[]const u8 {
    const S = struct {
        var on: ??[]const u8 = null;
    };
    if (S.on) |v| return v;
    const v = if (std.c.getenv("TF_MOM_KVDUMP")) |x| blk: {
        const s = std.mem.span(x);
        break :blk if (s.len == 0) null else s;
    } else null;
    S.on = v;
    return v;
}

/// One checkpoint's dump: the first `at` prompt ids and the kept-state copy's key/value region —
/// its host bytes (host save), or `dev`'s bytes, downloaded here (an integrated GPU keeps states on
/// the device). GDN recurrent state is out of scope for S1: the harness assumes it is recomputed.
pub fn dump(
    e: *Engine,
    io: std.Io,
    ids: []const u32,
    at: usize,
    page_tokens: u32,
    host: []const u8,
    dev: ?cuda.DeviceBuffer,
) !kvdump.Written {
    const base = dumpDir() orelse return error.KvdumpOff;
    const g = e.g;
    if (at == 0 or ids.len < at) return error.KvdumpPromptShort;
    const attn = g.attentions();
    if (attn == 0) return error.KvdumpNoAttention;
    const src: kvdump.Src = if (e.kv8) .fp8 else .bf16;
    const need = kvdump.copyBytes(src, g.kv_heads, attn, at);
    var region: []const u8 = undefined;
    var owned: ?[]u8 = null;
    defer if (owned) |b| e.gpa.free(b);
    if (dev) |b| {
        const full = try e.gpa.alloc(u8, b.len);
        errdefer e.gpa.free(full);
        try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(full.ptr, b.ptr, b.len, e.stream.handle), "cuMemcpyDtoHAsync");
        try e.stream.synchronize();
        owned = full;
        const sb = g.stateBytes();
        if (full.len < sb + need) return error.KvdumpCopyShort;
        region = full[sb .. sb + need];
    } else region = host;
    if (region.len < need) return error.KvdumpCopyShort;
    const layer_indices = try e.gpa.alloc(u32, attn);
    defer e.gpa.free(layer_indices);
    var n: usize = 0;
    for (0..g.layers) |i| if (!g.linear[i]) {
        layer_indices[n] = @intCast(i);
        n += 1;
    };
    if (n != attn) return error.KvdumpLayers;
    const m: kvdump.Meta = .{
        .model = std.fs.path.basename(e.dir),
        .model_dir = e.dir,
        .ckpt = 0, // writeDump numbers checkpoints by scanning the model's directory
        .src = src,
        .elem = if (e.kv8) .f32 else .bf16,
        .kv_heads = @intCast(g.kv_heads),
        .head_dim = kvdump.head_dim,
        .layer_indices = layer_indices,
        .token_ids = ids[0..at],
        .page_tokens = page_tokens,
        .model_layers = @intCast(g.layers),
    };
    return kvdump.writeDump(io, e.gpa, base, m, region);
}
