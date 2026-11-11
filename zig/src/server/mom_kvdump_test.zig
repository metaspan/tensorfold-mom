//! Host tests for the S1 KV-dump format (docs/12): the pure serializer, the kept-state copy's piece
//! packing, the fp8 decode, the sidecar schema and the checkpoint files themselves. The family's device
//! writer (cuda_kvdump.zig) hands these functions the bytes; the exe build compiles it. Filter: "kvdump".
const std = @import("std");
const kvdump = @import("kvdump_format");

const testing = std.testing;

/// A deterministic pattern so a test can point at any (layer, kv, head, position, dim) byte.
fn pattern(buf: []u8, seed: usize) void {
    var x: u64 = seed *% 6364136223846793005 +% 1442695040888963407;
    for (buf, 0..) |*b, i| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        b.* = @truncate(x +% i);
    }
}

fn metaOf(a: std.mem.Allocator, src: kvdump.Src, layers: []const u32, seq: usize, kv_heads: usize) !kvdump.Meta {
    const ids = try a.alloc(u32, seq);
    for (ids, 0..) |*t, i| t.* = @intCast(1000 + i);
    return .{
        .model = "qwen35-2b", // the engine's checkpoint folder name, as cuda_kvdump.zig passes it
        .model_dir = "/models/qwen35-2b",
        .ckpt = 5,
        .src = src,
        .elem = if (src == .fp8) .f32 else .bf16,
        .kv_heads = @intCast(kv_heads),
        .layer_indices = layers,
        .token_ids = ids,
        .page_tokens = 0,
        .model_layers = @intCast(layers[layers.len - 1] + 1),
    };
}

test "kvdump: kept-state copy pieces tile the region, rows then scales for fp8, no gaps or overlaps" {
    for ([_]kvdump.Src{ .bf16, .fp8 }) |src| for ([_]usize{ 1, 2, 4 }) |kvh| for ([_]usize{ 1, 3, 17 }) |at| {
        const attn: usize = 2;
        const n = kvdump.pieceCount(src, kvh, attn);
        var off: usize = 0;
        for (0..n) |k| {
            const p = kvdump.copyPiece(src, at, k);
            try testing.expectEqual(off, p.off); // packed: the save writes pieces back to back
            off += p.len;
        }
        try testing.expectEqual(off, kvdump.copyBytes(src, kvh, attn, at));
        // a bf16 block is at rows of head_dim bf16; an fp8 block is its rows then its one-scale-a-row scales
        const blk = if (src == .bf16) at * kvdump.head_dim * 2 else at * (kvdump.head_dim + 4);
        try testing.expectEqual(attn * 2 * kvh * blk, off);
    };
}

test "kvdump: a bf16 dump round-trips its header, tables and rows; short copies and mixed kinds refuse" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const m = try metaOf(a, .bf16, &.{ 3, 7, 11 }, 5, 2);
    const need = kvdump.copyBytes(.bf16, 2, 3, 5);
    const copy = try a.alloc(u8, need + 99); // room past the pieces: only the pieces may be read
    pattern(copy, 7);
    const bytes = try kvdump.buildBin(a, m, copy);
    const h = try kvdump.parseHeader(bytes);
    try testing.expectEqual(@as(u32, 3), h.n_layers);
    try testing.expectEqual(@as(u32, 2), h.kv_heads);
    try testing.expectEqual(@as(u32, kvdump.head_dim), h.head_dim);
    try testing.expectEqual(@as(u32, 5), h.seq_len);
    try testing.expectEqual(@as(u32, 5), h.ckpt);
    try testing.expectEqual(@as(u32, 0), h.elem);
    try testing.expectEqual(@as(u32, @intCast(kvdump.headerBytes(3, 5))), h.header_bytes);
    try testing.expectEqual(@as(usize, 0), h.header_bytes % 8);
    const t = kvdump.tables(h, bytes);
    try testing.expectEqualSlices(u32, &.{ 3, 7, 11 }, t.layers);
    try testing.expectEqual(@as(usize, 5), t.tokens.len);
    try testing.expectEqual(@as(u32, 1000), t.tokens[0]);
    try testing.expectEqual(@as(u32, 1004), t.tokens[4]);
    // bf16 rows keep the copy's save order: the data region is the pieces verbatim
    try testing.expectEqualSlices(u8, copy[0..need], bytes[h.header_bytes..][0..need]);
    try testing.expectEqual(need, kvdump.dataBytes(3, 2, kvdump.head_dim, 5, .bf16));
    // truncated leads, a broken magic and an inconsistent kind are refused
    try testing.expectError(error.KvdumpShort, kvdump.parseHeader(bytes[0 .. h.header_bytes + need - 1]));
    var broken = try a.dupe(u8, bytes);
    broken[3] ^= 0xFF;
    try testing.expectError(error.KvdumpMagic, kvdump.parseHeader(broken));
    broken = try a.dupe(u8, bytes);
    broken[8] = 9; // version
    try testing.expectError(error.KvdumpVersion, kvdump.parseHeader(broken));
    var mixed = m;
    mixed.elem = .f32; // bf16 rows pass through; they are never claimed as decoded f32
    try testing.expectError(error.KvdumpKind, kvdump.buildBin(a, mixed, copy));
    try testing.expectError(error.KvdumpCopyShort, kvdump.buildBin(a, m, copy[0 .. need - 1]));
}

test "kvdump: an fp8 cache's e4m3 rows decode through their per-row scales into the data region" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const m = try metaOf(a, .fp8, &.{ 3 }, 3, 1); // one attention layer, one head, three positions
    const copy = try a.alloc(u8, kvdump.copyBytes(.fp8, 1, 1, 3));
    @memset(copy, 0);
    const rows = kvdump.copyPiece(.fp8, 3, 0);
    const scales = kvdump.copyPiece(.fp8, 3, 1);
    copy[rows.off] = 0x38; // +1.0
    copy[rows.off + kvdump.head_dim] = 0x00; // +0
    copy[rows.off + 2 * kvdump.head_dim] = 0xB8; // -1.0
    for (0..3) |t| std.mem.writeInt(u32, copy[scales.off + t * 4 ..][0..4], @bitCast(@as(f32, 2.0)), .little);
    const bytes = try kvdump.buildBin(a, m, copy);
    const h = try kvdump.parseHeader(bytes);
    try testing.expectEqual(@as(u32, 1), h.elem); // rows land as f32
    const f: [*]const f32 = @alignCast(@ptrCast(bytes.ptr + h.header_bytes));
    try testing.expectEqual(@as(f32, 2.0), f[0]);
    try testing.expectEqual(@as(f32, 0.0), f[kvdump.head_dim]);
    try testing.expectEqual(@as(f32, -2.0), f[2 * kvdump.head_dim]);
    // every other element of this copy is a zero row times a 2.0 scale
    for (0 .. 3 * kvdump.head_dim) |i| {
        const want: f32 = if (i == 0) 2.0 else if (i == 2 * kvdump.head_dim) -2.0 else 0.0;
        try testing.expectEqual(want, f[i]);
    }
}

test "kvdump: e4m3 and bf16 element decodes match their formats exactly" {
    try testing.expectEqual(@as(f32, 0), kvdump.fp8ToF32(0x00));
    try testing.expectEqual(@as(f32, 1), kvdump.fp8ToF32(0x38));
    try testing.expectEqual(@as(f32, -1), kvdump.fp8ToF32(0xB8));
    try testing.expectEqual(@as(f32, 448), kvdump.fp8ToF32(0x7E)); // e4m3fn's largest finite
    try testing.expectEqual(@as(f32, -448), kvdump.fp8ToF32(0xFE));
    try testing.expectEqual(@as(f32, 0.001953125), kvdump.fp8ToF32(0x01)); // 2^-9, the smallest subnormal
    try testing.expectEqual(@as(f32, 0.00390625), kvdump.fp8ToF32(0x02)); // 2^-8
    try testing.expect(std.math.isNan(kvdump.fp8ToF32(0x7F)));
    try testing.expectEqual(@as(f32, 1), kvdump.bf16ToF32(0x3F80));
    try testing.expectEqual(@as(f32, -1), kvdump.bf16ToF32(0xBF80));
    try testing.expectEqual(@as(f32, 0.5), kvdump.bf16ToF32(0x3F00));
}

test "kvdump: decodeFp8ToBytes agrees with decodeFp8 element for element" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const rows = try a.alloc(u8, 2 * kvdump.head_dim);
    const scales = try a.alloc(u8, 2 * 4);
    for (0..rows.len) |i| rows[i] = if (i < kvdump.head_dim) 0x38 else 0xB8; // +1, -1
    for (0..2) |t_| std.mem.writeInt(u32, scales[t_ * 4 ..][0..4], @bitCast(@as(f32, 0.5)), .little);
    const via_f32 = try a.alloc(f32, rows.len);
    const via_bytes = try a.alloc(u8, rows.len * 4);
    kvdump.decodeFp8(rows, scales, via_f32);
    kvdump.decodeFp8ToBytes(rows, scales, via_bytes);
    for (0..rows.len) |i| {
        const got = std.mem.readInt(u32, via_bytes[i * 4 ..][0..4], .little);
        try testing.expectEqual(@as(u32, @bitCast(via_f32[i])), got);
    }
}

test "kvdump: buildBin survives the device build's SafeAllocator (alignedAlloc free-mismatch regression)" {
    // The device crash: alignedAlloc'd bytes freed plainly under SafeAllocator. Host tests
    // on the testing allocator cannot see that class — run the real path under SafeAllocator.
    var safe: std.heap.SafeAllocator = .init(testing.allocator, .{ .block_size_log2 = 12, .bucket_size_log2 = 12 });
    defer _ = safe.deinit();
    const a = safe.allocator();
    const m = try metaOf(a, .fp8, &.{ 3 }, 3, 1);
    const copy = try a.alloc(u8, kvdump.copyBytes(.fp8, 1, 1, 3));
    @memset(copy, 0);
    const bytes = try kvdump.buildBin(a, m, copy);
    _ = try kvdump.parseHeader(bytes);
    a.free(bytes); // must match the allocation's recorded alignment
    a.free(copy);
    a.free(m.token_ids); // layer_indices is borrowed (a stack literal), as in the writer
}

test "kvdump: the sidecar names the model, checkpoint, layers and element kind as JSON" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const m = try metaOf(a, .fp8, &.{ 3, 7, 11, 15, 19, 23 }, 4, 2);
    const text = try kvdump.sidecar(a, m);
    const doc = try std.json.parseFromSlice(std.json.Value, a, text, .{});
    defer doc.deinit();
    const o = doc.value.object;
    try testing.expectEqualStrings(kvdump.sidecar_format, o.get("format").?.string);
    try testing.expectEqual(@as(i64, 1), o.get("version").?.integer);
    try testing.expectEqualStrings("qwen35-2b", o.get("model").?.string);
    try testing.expectEqualStrings("/models/qwen35-2b", o.get("model_dir").?.string);
    try testing.expectEqual(@as(i64, 5), o.get("ckpt").?.integer);
    try testing.expectEqual(@as(i64, 4), o.get("seq_len").?.integer);
    try testing.expectEqual(@as(i64, 4), o.get("at").?.integer);
    try testing.expectEqualStrings("f32", o.get("elem").?.string);
    try testing.expectEqualStrings("fp8", o.get("cache_format").?.string);
    try testing.expectEqualStrings("ckpt5.bin", o.get("bin").?.string);
    try testing.expectEqual(@as(i64, 6), @as(i64, @intCast(o.get("layers").?.array.items.len)));
    try testing.expectEqual(@as(i64, 3), o.get("layers").?.array.items[0].integer);
    try testing.expectEqual(@as(i64, 23), o.get("layers").?.array.items[5].integer);
    try testing.expectEqual(@as(i64, 2), o.get("kv_heads").?.integer);
    try testing.expectEqual(@as(i64, kvdump.head_dim), o.get("head_dim").?.integer);
    try testing.expectEqual(@as(i64, 24), o.get("model_layers").?.integer);
    try testing.expectEqual(
        kvdump.dataBytes(6, 2, kvdump.head_dim, 4, .f32),
        @as(usize, @intCast(o.get("data_bytes").?.integer)),
    );
    try testing.expectEqual(kvdump.tokenHash(m.token_ids), @as(u64, @intCast(o.get("token_hash").?.integer)));
    // a fingerprint only for alignment hints: different ids hash differently
    var other = m;
    var ids = try a.dupe(u32, m.token_ids);
    ids[0] = 999;
    other.token_ids = ids;
    try testing.expect(kvdump.tokenHash(m.token_ids) != kvdump.tokenHash(other.token_ids));
}

test "kvdump: checkpoint names, model directory names and the base are read defensively" {
    try testing.expectEqual(@as(?u32, 7), kvdump.parseCkptName("ckpt7.bin"));
    try testing.expectEqual(@as(?u32, 0), kvdump.parseCkptName("ckpt0.bin"));
    try testing.expectEqual(@as(?u32, null), kvdump.parseCkptName("ckpt3.json"));
    try testing.expectEqual(@as(?u32, null), kvdump.parseCkptName("ckptx.bin"));
    try testing.expectEqual(@as(?u32, null), kvdump.parseCkptName("ckpt.bin"));
    try testing.expectEqual(@as(?u32, null), kvdump.parseCkptName("keep.bin"));
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try testing.expectEqualStrings("qwen35-2b", try kvdump.modelDirName(a, "/home/derek/models/tensorfold-mom/qwen35-2b"));
    try testing.expectEqualStrings("qwen36-35b-a3b-nvfp4", try kvdump.modelDirName(a, "/models/qwen36-35b-a3b-nvfp4/"));
    try testing.expectEqualStrings("d_", try kvdump.modelDirName(a, "a/b c/d!"));
    try testing.expectError(error.KvdumpModelName, kvdump.modelDirName(a, "x/."));
    try testing.expectError(error.KvdumpModelName, kvdump.modelDirName(a, "x/.."));
    try testing.expectError(error.KvdumpModelName, kvdump.modelDirName(a, "/"));
}

test "kvdump: writeDump numbers checkpoints per model directory and lands both files whole" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const base = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const cwd = std.Io.Dir.cwd();

    const m = try metaOf(a, .bf16, &.{ 3, 7 }, 6, 2);
    const need = kvdump.copyBytes(.bf16, 2, 2, 6);
    const copy = try a.alloc(u8, need);
    pattern(copy, 11);

    const w0 = try kvdump.writeDump(io, a, base, m, copy);
    try testing.expectEqual(@as(u32, 0), w0.ckpt);
    try testing.expectEqualStrings(try std.fs.path.join(a, &.{ base, "qwen35-2b" }), w0.dir);
    const w1 = try kvdump.writeDump(io, a, base, m, copy);
    try testing.expectEqual(@as(u32, 1), w1.ckpt); // sequence advanced past the first checkpoint

    // another model's directory counts from its own zero
    var m2 = m;
    m2.model = "/models/other 35B";
    const w2 = try kvdump.writeDump(io, a, base, m2, copy);
    try testing.expectEqual(@as(u32, 0), w2.ckpt);

    const bin_path = try std.fs.path.join(a, &.{ base, "qwen35-2b", "ckpt0.bin" });
    const json_path = try std.fs.path.join(a, &.{ base, "qwen35-2b", "ckpt0.json" });
    const bin = try cwd.readFileAlloc(io, bin_path, a, .limited(1 << 24));
    const json = try cwd.readFileAlloc(io, json_path, a, .limited(1 << 16));
    const h = try kvdump.parseHeader(bin);
    try testing.expectEqual(@as(u32, 0), h.ckpt);
    try testing.expectEqualSlices(u8, copy[0..need], bin[h.header_bytes..][0..need]);
    const doc = try std.json.parseFromSlice(std.json.Value, a, json, .{});
    defer doc.deinit();
    try testing.expectEqual(@as(i64, 0), doc.value.object.get("ckpt").?.integer);
    try testing.expectEqualStrings("ckpt0.bin", doc.value.object.get("bin").?.string);

    // a second process's writer scans the directory: the next file is ckpt2, never a clobber
    const w3 = try kvdump.writeDump(io, a, base, m, copy);
    try testing.expectEqual(@as(u32, 2), w3.ckpt);
    const again = try cwd.readFileAlloc(io, bin_path, a, .limited(1 << 24));
    try testing.expectEqualSlices(u8, bin, again);
}
