//! S1 (docs/12): the KV-dump wire format for the cross-model translated-cache study. Pure (std only):
//! a checkpoint's full-attention K/V rows, their token ids and metadata serialize to a fixed binary
//! header plus a JSON sidecar, so one offline harness can read two engines' dumps of the same token
//! prefix. The device writer (cuda_kvdump.zig) feeds `buildBin` the key/value region of a B5 kept-state
//! copy (cuda_reuse.zig's Layout packing, pieces back to back); the host tests
//! (server/mom_kvdump_test.zig, filter "kvdump") exercise this file directly.
//!
//! .bin layout (little-endian):
//!   Header (44 bytes: magic "TFKVDMP1", version, elem, n_layers, kv_heads, head_dim, seq_len, ckpt,
//!   page_tokens, header_bytes) — header_bytes is 8-aligned and where the data region starts.
//!   layer_indices: n_layers x u32 (the model's full-attention layer indices, ascending)
//!   token_ids:     seq_len  x u32
//!   data:          for each dumped layer, K then V, for each kv head, for each position:
//!                  head_dim elements — u16 bf16 bits (elem=bf16) or f32 (elem=f32: an fp8 cache's
//!                  e4m3 rows scaled by their per-row fp32 scale, decoded at write time).
//! The sidecar ckpt<N>.json repeats the metadata (schema "format": "tfmom-kvdump", version 1) for tools.
const std = @import("std");

pub const magic = "TFKVDMP1";
pub const version: u32 = 1;
pub const sidecar_format = "tfmom-kvdump";

/// qwen3_5's fixed attention width (cuda_geometry.zig's head_dim).
pub const head_dim = 256;

/// How rows sit in the .bin data region.
pub const Elem = enum(u32) { bf16 = 0, f32 = 1 };
/// The engine's cache format the copy came from (mirrors cuda_reuse.zig's Format).
pub const Src = enum(u32) { bf16 = 0, fp8 = 1 };

pub fn elemBytes(e: Elem) usize {
    return if (e == .bf16) 2 else 4;
}

/// The .bin's fixed lead; all fields little-endian u32 after the 8-byte magic.
pub const Header = extern struct {
    magic: [8]u8 = std.mem.zeroes([8]u8),
    version: u32 = 1,
    elem: u32 = 0,
    n_layers: u32 = 0,
    kv_heads: u32 = 0,
    head_dim: u32 = 0,
    seq_len: u32 = 0,
    ckpt: u32 = 0,
    page_tokens: u32 = 0,
    header_bytes: u32 = 0,
};
comptime {
    if (@sizeOf(Header) != 44) @compileError("kvdump Header must stay 44 bytes");
}

/// The .bin's total lead: Header, the layer-index table and the token table, 8-aligned.
pub fn headerBytes(n_layers: usize, seq_len: usize) usize {
    return std.mem.alignForward(usize, @sizeOf(Header) + 4 * n_layers + 4 * seq_len, 8);
}

/// The .bin's data region: one head_dim-element row per (layer, K/V, head, position).
pub fn dataBytes(n_layers: usize, kv_heads: usize, head_dim_: usize, seq_len: usize, elem: Elem) usize {
    return n_layers * 2 * kv_heads * seq_len * head_dim_ * elemBytes(elem);
}

/// One key/value piece of a kept-state copy (cuda_reuse.zig's Layout, past its state piece): pieces
/// lie packed in save order — per attention layer the key heads then the value heads, bf16 one rows
/// piece, fp8 a rows piece then its scales. `k` counts key/value pieces from 0, `at` the rows each holds.
pub const CopyPiece = struct { off: usize, len: usize };

pub fn copyPiece(src: Src, at: usize, k: usize) CopyPiece {
    if (src == .bf16) return .{ .off = k * at * head_dim * 2, .len = at * head_dim * 2 };
    const stride = at * (head_dim + 4);
    return if (k % 2 == 0) // even k: a block's e4m3 rows, odd k: the same block's fp32 scales
        .{ .off = (k / 2) * stride, .len = at * head_dim }
    else
        .{ .off = (k / 2) * stride + at * head_dim, .len = at * 4 };
}

/// Pieces a copy of `attentions` layers holds (bf16 one per layer-K/V/head, fp8 two: rows and scales).
pub fn pieceCount(src: Src, kv_heads: usize, attentions: usize) usize {
    return attentions * 2 * kv_heads * (if (src == .fp8) @as(usize, 2) else @as(usize, 1));
}

/// The bytes such a copy's key/value pieces hold in total.
pub fn copyBytes(src: Src, kv_heads: usize, attentions: usize, at: usize) usize {
    var n: usize = 0;
    for (0..pieceCount(src, kv_heads, attentions)) |k| n += copyPiece(src, at, k).len;
    return n;
}

/// Everything one checkpoint's files record besides the copied rows themselves.
pub const Meta = struct {
    model: []const u8, // the engine's checkpoint folder name (its dump subdirectory)
    model_dir: []const u8, // the engine's checkpoint dir as opened
    ckpt: u32,
    src: Src,
    elem: Elem,
    kv_heads: u32,
    head_dim: u32 = head_dim,
    layer_indices: []const u32, // the model's full-attention layer indices, ascending
    token_ids: []const u32,
    page_tokens: u32 = 0, // the cache's page size, 0: contiguous
    model_layers: u32 = 0, // the model's whole layer count (provenance)
};

pub fn fp8ToF32(b: u8) f32 {
    const e: u8 = (b >> 3) & 0xF;
    const m: u8 = b & 7;
    var v: f32 = undefined;
    if (e == 0) {
        v = @as(f32, @floatFromInt(m)) * 0.001953125; // m * 2^-9: the e4m3 subnormals and zero
    } else if (e == 0xF and m == 7) {
        return std.math.nan(f32);
    } else {
        const mant: f32 = 1.0 + @as(f32, @floatFromInt(m)) * 0.125;
        v = mant * std.math.pow(f32, 2.0, @as(f32, @floatFromInt(e)) - 7.0);
    }
    return if (b & 0x80 != 0) -v else v;
}

pub fn bf16ToF32(bits: u16) f32 {
    return @bitCast(@as(u32, @as(u16, bits)) << 16);
}

/// One head's `rows` e4m3 rows (rows.len = rows_n*head_dim) with one fp32 scale a row, decoded.
pub fn decodeFp8(rows: []const u8, scales: []const u8, out: []f32) void {
    std.debug.assert(rows.len % head_dim == 0);
    std.debug.assert(scales.len == rows.len / head_dim * 4);
    std.debug.assert(out.len == rows.len);
    for (0..rows.len / head_dim) |t| {
        const scale = std.mem.readInt(u32, scales[t * 4 ..][0..4], .little);
        const s: f32 = @bitCast(scale);
        for (0..head_dim) |d| out[t * head_dim + d] = fp8ToF32(rows[t * head_dim + d]) * s;
    }
}

/// A checkpoint's .bin bytes: header, tables, then the rows in dump order from a kept-state copy's
/// key/value region (what cuda_reuse.zig packs after its state piece).
pub fn buildBin(a: std.mem.Allocator, m: Meta, copy: []const u8) ![]u8 {
    const n = m.layer_indices.len;
    const seq = m.token_ids.len;
    if (m.head_dim != head_dim) return error.KvdumpHeadDim;
    if (m.kv_heads == 0 or n == 0 or seq == 0) return error.KvdumpEmpty;
    if (@intFromEnum(m.src) > 1 or @intFromEnum(m.elem) > 1) return error.KvdumpKind;
    if (m.src == .bf16 and m.elem != .bf16) return error.KvdumpKind; // bf16 rows pass through
    const want = copyBytes(m.src, m.kv_heads, n, seq);
    if (copy.len < want) return error.KvdumpCopyShort;
    const hb = headerBytes(n, seq);
    const total = hb + dataBytes(n, m.kv_heads, head_dim, seq, m.elem);
    const bytes = try a.alignedAlloc(u8, .of(f32), total); // the f32 data region is written through f32 pointers
    errdefer a.free(bytes);
    var h: Header = .{
        .elem = @intFromEnum(m.elem),
        .n_layers = @intCast(n),
        .kv_heads = m.kv_heads,
        .head_dim = m.head_dim,
        .seq_len = @intCast(seq),
        .ckpt = m.ckpt,
        .page_tokens = m.page_tokens,
        .header_bytes = @intCast(hb),
    };
    @memcpy(h.magic[0..], magic);
    @memcpy(bytes[0..@sizeOf(Header)], std.mem.asBytes(&h));
    var at: usize = @sizeOf(Header);
    for (m.layer_indices) |li| {
        std.mem.writeInt(u32, bytes[at..][0..4], li, .little);
        at += 4;
    }
    for (m.token_ids) |t| {
        std.mem.writeInt(u32, bytes[at..][0..4], t, .little);
        at += 4;
    }
    std.debug.assert(at <= hb);
    @memset(bytes[at..hb], 0); // the lead's 8-alignment pad: zeros, so dumps are byte-reproducible
    var out = bytes[hb..];
    for (0..n) |layer| for (0..2) |kv| for (0..m.kv_heads) |head| {
        // the copy's piece k for this row group: layer-major, keys then values, head ascending
        const block = (layer * 2 + kv) * m.kv_heads + head;
        const k = if (m.src == .bf16) block else 2 * block;
        const rows = copyPiece(m.src, seq, k);
        const row_bytes: usize = if (m.src == .fp8) head_dim else head_dim * 2;
        std.debug.assert(rows.len == seq * row_bytes);
        if (m.elem == .bf16) {
            @memcpy(out[0..rows.len], copy[rows.off..][0..rows.len]);
            out = out[rows.len..];
        } else {
            const scales = copyPiece(m.src, seq, k + 1);
            std.debug.assert(scales.len == seq * 4);
            const dst = out.ptr[0 .. seq * head_dim * 4];
            decodeFp8(copy[rows.off..][0..rows.len], copy[scales.off..][0..scales.len], @as([*]f32, @alignCast(@ptrCast(dst)))[0 .. seq * head_dim]);
            out = out[dst.len..];
        }
    };
    std.debug.assert(out.len == 0);
    return bytes;
}

/// Read back and validate a .bin's lead; the caller owns `bytes`.
pub fn parseHeader(bytes: []const u8) !Header {
    if (bytes.len < @sizeOf(Header)) return error.KvdumpShort;
    var h: Header = undefined;
    @memcpy(std.mem.asBytes(&h), bytes[0..@sizeOf(Header)]);
    if (!std.mem.eql(u8, &h.magic, magic)) return error.KvdumpMagic;
    if (h.version != version) return error.KvdumpVersion;
    if (h.n_layers == 0 or h.kv_heads == 0 or h.seq_len == 0) return error.KvdumpFields;
    if (h.head_dim != head_dim) return error.KvdumpHeadDim;
    if (h.elem > 1) return error.KvdumpKind;
    if (h.header_bytes != headerBytes(h.n_layers, h.seq_len)) return error.KvdumpHeaderBytes;
    const need = h.header_bytes + dataBytes(h.n_layers, h.kv_heads, h.head_dim, h.seq_len, @enumFromInt(h.elem));
    if (bytes.len < need) return error.KvdumpShort;
    return h;
}

/// Layer indices and token ids read back from a .bin lead (valid once parseHeader accepted `bytes`).
/// `bytes` must be at least 4-aligned (a GPA file buffer is); the tables start at 4-aligned offsets.
pub fn tables(h: Header, bytes: []const u8) struct { layers: []const u32, tokens: []const u32 } {
    const ls = bytes[@sizeOf(Header)..][0 .. h.n_layers * 4];
    const ts = bytes[@sizeOf(Header) + h.n_layers * 4 ..][0 .. h.seq_len * 4];
    return .{ .layers = @alignCast(@ptrCast(ls)), .tokens = @alignCast(@ptrCast(ts)) };
}

/// The ckpt<N>.json sidecar: the same metadata as the header, JSON-addressable for the harness.
pub fn sidecar(a: std.mem.Allocator, m: Meta) ![]u8 {
    const model = try escapeJson(a, m.model);
    defer a.free(model);
    const dir = try escapeJson(a, m.model_dir);
    defer a.free(dir);
    var layers: std.ArrayList(u8) = .empty;
    defer layers.deinit(a);
    try layers.append(a, '[');
    for (m.layer_indices, 0..) |li, i| {
        if (i > 0) try layers.append(a, ',');
        try layers.print(a, "{d}", .{li});
    }
    try layers.append(a, ']');
    const bin = try fileName(a, m.ckpt, "bin");
    defer a.free(bin);
    return std.fmt.allocPrint(a,
        \\{{"format":"{s}","version":1,"model":"{s}","model_dir":"{s}","ckpt":{d},"at":{d},"seq_len":{d},"layers":{s},"kv_heads":{d},"head_dim":{d},"elem":"{s}","cache_format":"{s}","page_tokens":{d},"bin":"{s}","data_bytes":{d},"token_hash":{d},"model_layers":{d}}}
    , .{
        sidecar_format, model, dir, m.ckpt, m.token_ids.len, m.token_ids.len,
        layers.items,   m.kv_heads, m.head_dim,
        if (m.elem == .bf16) "bf16" else "f32",
        if (m.src == .bf16) "bf16" else "fp8",
        m.page_tokens,  bin,
        dataBytes(m.layer_indices.len, m.kv_heads, head_dim, m.token_ids.len, m.elem),
        tokenHash(m.token_ids),
        m.model_layers,
    });
}

/// A stable 64-bit fingerprint of a checkpoint's token ids (harness alignment hint; ids are truth).
pub fn tokenHash(ids: []const u32) u64 {
    return std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(ids));
}

pub fn fileName(a: std.mem.Allocator, ckpt: u32, ext: []const u8) ![]u8 {
    return std.fmt.allocPrint(a, "ckpt{d}.{s}", .{ ckpt, ext });
}

/// `ckpt<N>.bin` names the sequence; every other name is ignored.
pub fn parseCkptName(name: []const u8) ?u32 {
    if (!std.mem.startsWith(u8, name, "ckpt") or !std.mem.endsWith(u8, name, ".bin")) return null;
    return std.fmt.parseInt(u32, name[4 .. name.len - 4], 10) catch null;
}

/// The dump subdirectory an engine's checkpoint dir names it: basename, path-safe characters only.
pub fn modelDirName(a: std.mem.Allocator, dir: []const u8) ![]u8 {
    var base = dir;
    while (base.len > 0 and base[base.len - 1] == '/') base = base[0 .. base.len - 1];
    if (std.mem.lastIndexOfScalar(u8, base, '/')) |s| base = base[s + 1 ..];
    if (base.len == 0) return error.KvdumpModelName;
    const out = try a.alloc(u8, base.len);
    for (base, 0..) |c, i| out[i] = if (safeNameChar(c)) c else '_';
    if (std.mem.eql(u8, out[0..], ".") or std.mem.eql(u8, out[0..], "..")) return error.KvdumpModelName;
    return out;
}

fn safeNameChar(c: u8) bool {
    return (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or (c >= '0' and c <= '9') or
        c == '.' or c == '_' or c == '-';
}

fn escapeJson(a: std.mem.Allocator, s: []const u8) ![]u8 {
    var list: std.ArrayList(u8) = .empty;
    errdefer list.deinit(a);
    for (s) |c| {
        switch (c) {
            '"' => try list.appendSlice(a, "\\\""),
            '\\' => try list.appendSlice(a, "\\\\"),
            '\n' => try list.appendSlice(a, "\\n"),
            '\r' => try list.appendSlice(a, "\\r"),
            '\t' => try list.appendSlice(a, "\\t"),
            else => if (c < 0x20) try list.appendSlice(a, "\\u0000") else try list.append(a, c),
        }
    }
    return list.toOwnedSlice(a);
}

pub const Written = struct { ckpt: u32, dir: []const u8, bin_bytes: usize };

/// Write ckpt<N>.bin and ckpt<N>.json under <base>/<model>/, N the next free sequence in that
/// directory (existing ckpt*.bin scanned, so restarts keep numbering). Files land whole.
pub fn writeDump(io: std.Io, a: std.mem.Allocator, base: []const u8, meta: Meta, copy: []const u8) !Written {
    if (base.len == 0) return error.KvdumpBase;
    const model = try modelDirName(a, meta.model);
    defer a.free(model);
    var m = meta;
    m.model = model;
    const dir = try std.fs.path.join(a, &.{ base, model });
    defer a.free(dir);
    std.Io.Dir.cwd().createDirPath(io, dir) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    };
    var next: u32 = 0;
    if (std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true })) |d| {
        defer d.close(io);
        var it = d.iterate();
        while (try it.next(io)) |e| {
            if (parseCkptName(e.name)) |n| next = @max(next, n + 1);
        }
    } else |_| {}
    m.ckpt = next;
    const bin = try buildBin(a, m, copy);
    defer a.free(bin); // buildBin's lead is 8-aligned; the f32 region is 4-aligned by its allocator
    const json = try sidecar(a, m);
    defer a.free(json);
    const bin_name = try std.fs.path.join(a, &.{ dir, try fileName(a, next, "bin") });
    defer a.free(bin_name);
    const json_name = try std.fs.path.join(a, &.{ dir, try fileName(a, next, "json") });
    defer a.free(json_name);
    const cwd = std.Io.Dir.cwd();
    try cwd.writeFile(io, .{ .sub_path = bin_name, .data = bin });
    try cwd.writeFile(io, .{ .sub_path = json_name, .data = json });
    return .{ .ckpt = next, .dir = try a.dupe(u8, dir), .bin_bytes = bin.len }; // dir outlives this frame
}
