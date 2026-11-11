//! Qwen3.5 CUDA prompt states for the shared cache, pinned on the host or on an integrated GPU's device memory.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const geometry = @import("cuda_geometry.zig");
const engine = @import("cuda_engine.zig");
const caches = @import("cuda_cache.zig");
const kvdumpWriter = @import("cuda_kvdump.zig");

const Engine = engine.Engine;
const Cache = engine.Cache;
const Geometry = geometry.Geometry;
const head_dim = geometry.head_dim;

/// A cache's attention rows: bf16, or e4m3 with an fp32 scale each (--kv-cache fp8, where the engine has it).
pub const Format = enum(u8) { bf16, fp8 };

/// The engine's cache format; an engine without fp8 caches keeps bf16 rows.
pub fn formatOf(e: *const Engine) Format {
    if (comptime @hasField(Engine, "kv8")) return if (e.kv8) .fp8 else .bf16;
    return .bf16;
}

/// An image a kept state's prompt holds: the row its placeholders start at, its identity (core/picture/identity.zig).
pub const Seen = struct { at: u32, id: u64 };

/// One kept state: its bytes, the position it stands at, its rotary shift, its cache format and images.
pub const Copy = struct {
    host: cuda.HostBuffer,
    dev: ?cuda.DeviceBuffer = null, // the bytes in device memory (Target.device): `host` is empty then
    len: u64,
    at: usize,
    delta: i32,
    format: Format,
    images: []Seen,
    reading: bool = false, // a restore's queued copies may still read `host`: a drop keeps it until they are done

    /// The bytes the copy holds room for, where they are.
    pub fn room(c: *const Copy) usize {
        return if (c.dev) |b| b.len else c.host.bytes.len;
    }
};

/// The engine, the streams' caches a save copies, and their prompts' images (cuda_native.zig's Lanes owns the maps).
pub const Target = struct {
    e: *Engine,
    caches: *std.AutoHashMapUnmanaged(*lanes.Stream, *Cache),
    seen: *std.AutoHashMapUnmanaged(*lanes.Stream, []Seen),
    ctx: ?*const cuda.Context = null, // --learn-conversations: made current where the disk tier allocates copies
    device: bool = false, // an integrated GPU: saves go to device memory, unless the disk tier reads them (ctx)
    /// TF_MOM_KVDUMP's files are written through this (cuda_kvdump.zig, docs/12 S1).
    io: std.Io,
    /// MoM M3 (docs/13): when set (the parent side, TF_MOM_ADOPT), `save` records its last
    /// kept state here so the paired thin engine can adopt it; `drop` retires the record.
    adopt_sink: ?*AdoptSink = null,
};

/// MoM M3: the last state a parent Target saved, named by the tokens it stands after.
/// The `copy` belongs to the store: `live` means it has not been dropped since the save.
pub const AdoptSink = struct {
    copy: ?*anyopaque = null,
    at: u32 = 0,
    tokens: []const u32 = &.{},
    live: bool = false,

    /// Take ownership of a fresh token copy standing after `at` tokens of `copy`.
    pub fn record(self: *AdoptSink, a: std.mem.Allocator, copy: *anyopaque, at: u32, prompt: []const u32) !void {
        const tok = try a.alloc(u32, prompt.len);
        @memcpy(tok, prompt);
        self.clear(a);
        self.* = .{ .copy = copy, .at = at, .tokens = tok, .live = true };
    }

    /// Retire the record (a drop of the recorded copy, or a take-over by `record`).
    pub fn clear(self: *AdoptSink, a: std.mem.Allocator) void {
        if (self.tokens.len != 0) a.free(self.tokens);
        self.* = .{};
    }
};

/// One piece of a stream's state: its offset in the state or key/value buffer, its length.
pub const Piece = struct { kv: bool, off: usize, len: usize };

/// A state's pieces by index, any geometry: the state, then each (layer, key/value, head)'s first `at` rows.
pub const Layout = struct {
    g: Geometry,
    format: Format,
    cap: usize,
    at: usize,

    fn parts(l: Layout) usize {
        return if (l.format == .fp8) 2 else 1;
    }

    pub fn count(l: Layout) usize {
        return 1 + l.g.attentions() * 2 * l.g.kv_heads * l.parts();
    }

    /// Piece `i`: qwen35.cu's layout (a layer's keys then values, kv_heads blocks of cap rows), or qwen35_kv8.cu's.
    pub fn piece(l: Layout, i: usize) Piece {
        if (i == 0) return .{ .kv = false, .off = 0, .len = l.g.stateBytes() };
        const j = (i - 1) / l.parts();
        const kvh = l.g.kv_heads;
        const block = j % (2 * kvh); // key heads, then value heads
        const layer = j / (2 * kvh);
        return switch (l.format) {
            .bf16 => .{ .kv = true, .off = (layer * 2 * kvh + block) * l.cap * head_dim * 2, .len = l.at * head_dim * 2 },
            .fp8 => blk: {
                const base = layer * 2 * kvh * l.cap * (head_dim + 4);
                break :blk if ((i - 1) % 2 == 0)
                    .{ .kv = true, .off = base + block * l.cap * head_dim, .len = l.at * head_dim }
                else
                    .{ .kv = true, .off = base + 2 * kvh * l.cap * head_dim + block * l.cap * 4, .len = l.at * 4 };
            },
        };
    }

    /// The bytes a save at `at` takes.
    pub fn total(l: Layout) u64 {
        var n: u64 = 0;
        for (0..l.count()) |i| n += l.piece(i).len;
        return n;
    }

    /// The key/value buffer's bytes at `cap` (Geometry.kvBytes).
    fn kvBytes(l: Layout) usize {
        return l.g.attentions() * 2 * l.g.kv_heads * l.cap * @as(usize, if (l.format == .fp8) head_dim + 4 else head_dim * 2);
    }
};

/// A piece's device spans: one, or with a page table a run a page (the host copy keeps the contiguous order).
const Runs = struct {
    base: u64,
    pages: ?caches.Paged,
    page_bytes: usize,
    in_page: usize,
    span: usize = 0, // a full page's run of the piece
    len: usize,
    page: usize = 0,
    done: usize = 0,

    const Run = struct { dev: u64, at: usize, len: usize };

    fn of(cache: *const Cache, l: Layout, i: usize, p: Piece) Runs {
        if (!p.kv or cache.paged == null) return .{ .base = (if (p.kv) cache.kv.ptr else cache.state.ptr) + p.off, .pages = null, .page_bytes = 0, .in_page = 0, .len = p.len };
        const pg = cache.paged.?;
        // the piece's offset inside a page: its layout at cap = tokens (Layout.piece), a run of tokens rows a page
        const in_page: Layout = .{ .g = l.g, .format = l.format, .cap = pg.tokens, .at = pg.tokens };
        const q = in_page.piece(i);
        return .{ .base = pg.base, .pages = pg, .page_bytes = in_page.kvBytes(), .in_page = q.off, .span = q.len, .len = p.len };
    }

    fn next(r: *Runs) ?Run {
        if (r.done >= r.len) return null;
        const pg = r.pages orelse {
            r.done = r.len;
            return .{ .dev = r.base, .at = 0, .len = r.len };
        };
        const n = @min(r.span, r.len - r.done);
        const run: Run = .{ .dev = r.base + pg.ids[r.page] * r.page_bytes + r.in_page, .at = r.done, .len = n };
        r.page += 1;
        r.done += n;
        return run;
    }
};

/// Whether a cache's keys and values go a page a copy: bf16 pages, whose pieces lie back to back a page's rows apart.
fn wholePages(cache: *const Cache, l: Layout) bool {
    return cache.paged != null and l.format == .bf16 and l.at > 0;
}

/// Every bf16 piece's rows from `at`'s pages, one 2D copy a page (`tokens` rows apart there, `at` apart in `copy`).
fn pageCopies(e: *Engine, cache: *const Cache, l: Layout, copy: *const Copy, off: usize, way: enum { save, restore }) !void {
    const pg = cache.paged.?;
    const row = head_dim * 2;
    const page_bytes = (Layout{ .g = l.g, .format = l.format, .cap = pg.tokens, .at = pg.tokens }).kvBytes();
    const pieces = l.count() - 1;
    var p: usize = 0;
    while (p * pg.tokens < l.at) : (p += 1) {
        const rows = @min(pg.tokens, l.at - p * pg.tokens);
        const dev = pg.base + pg.ids[p] * page_bytes;
        const o = off + p * pg.tokens * row;
        // the kept side: device memory on an integrated GPU (copy.dev), else the pinned host copy
        const kept: cuda.abi.DevicePtr = if (copy.dev) |b| b.ptr + o else 0;
        const host: ?*u8 = if (copy.dev == null) &copy.host.bytes.ptr[o] else null;
        const kind: c_int = @intCast(if (copy.dev == null) cuda.abi.memory_type_host else cuda.abi.memory_type_device);
        const m: cuda.abi.Memcpy2D = switch (way) {
            .save => .{ .src = dev, .src_pitch = pg.tokens * row, .dst_type = kind, .dst_host = host, .dst = kept, .dst_pitch = l.at * row, .width = rows * row, .height = pieces },
            .restore => .{ .src_type = kind, .src_host = host, .src = kept, .src_pitch = l.at * row, .dst = dev, .dst_pitch = pg.tokens * row, .width = rows * row, .height = pieces },
        };
        try e.d.check(e.d.api.cuMemcpy2DAsync_v2(&m, e.stream.handle), "cuMemcpy2DAsync");
    }
}

/// Whether a request's images match a kept state's: the same identities at the same rows, for every image before `at`.
pub fn sameImages(kept: []const Seen, request: []const Seen, at: usize) bool {
    var n: usize = 0;
    for (request) |r| {
        if (r.at >= at) continue;
        if (n >= kept.len or kept[n].at != r.at or kept[n].id != r.id) return false;
        n += 1;
    }
    return n == kept.len;
}

/// The pieces a host copy at `at` holds in the engine's cache format.
pub fn layoutOf(e: *const Engine, at: usize) Layout {
    return .{ .g = e.g, .format = formatOf(e), .cap = 0, .at = at };
}

/// Host bytes a save at `at` takes.
pub fn bytes(ptr: *anyopaque, at: u32) u64 {
    const t: *Target = @ptrCast(@alignCast(ptr));
    return (Layout{ .g = t.e.g, .format = formatOf(t.e), .cap = 0, .at = at }).total();
}

/// Copy the stream's state after `at` prompt tokens; its prompt pass stands there.
pub fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
    const t: *Target = @ptrCast(@alignCast(ptr));
    const s: *lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoStream));
    const cache = t.caches.get(s) orelse return error.UnknownStream;
    // MoM M3 diagnostic: TF_MOM_ADOPT_HOST keeps the records host-readable for the byte audit.
    const keep_host = adoptEnabled() and std.c.getenv("TF_MOM_ADOPT_HOST") != null;
    if (keep_host and std.c.getenv("TF_MOM_AUDIT_COLD") != null)
        std.log.info("qwen3_5: adopt audit save at {d} cuts {any}", .{ at, s.chunks });
    const copy = try saveCache(t.e, cache, at, t.seen.get(s) orelse &.{}, t.device and t.ctx == null and !keep_host);
    // the audit fires either after an adoption (kv pieces only: the state block was resumed and mutated)
    // or, when TF_MOM_AUDIT_COLD is set, on this engine's own cold save (no resume: the state block is
    // byte-fresh, so a match proves the map copies exactly what the engine itself writes).
    if (takeAudit()) |aa| {
        var a = aa;
        defer a.map.free(t.e.gpa);
        if (t.e.g.layers < a.parent_layers) diffAudit(a, copy, false);
    } else if (keep_host and std.c.getenv("TF_MOM_AUDIT_COLD") != null and cache.len == @as(usize, at)) {
        scan: for (open_targets) |o| {
            const p = o orelse continue;
            if (p == t or p.e.g.layers <= t.e.g.layers) continue;
            const sink = p.adopt_sink orelse continue;
            if (!sink.live) continue;
            const pc: *Copy = @ptrCast(@alignCast(sink.copy orelse continue));
            if (pc.at != at or pc.images.len != 0) continue;
            if (AdoptMap.buildParams(t.e.gpa, p.e.g, t.e.g, formatOf(t.e), at)) |dm| {
                var dmm = dm;
                defer dmm.free(t.e.gpa);
                diffAudit(.{ .parent_copy = pc, .map = dmm, .parent_layers = p.e.g.layers, .pg = p.e.g }, copy, true);
            }
            break :scan;
        }
    }
    // MoM M3 (docs/13): the paired thin engine adopts from this record; a bookkeeping OOM just keeps the previous record.
    if (t.adopt_sink) |sink| {
        sink.record(t.e.gpa, copy, at, s.prompt()) catch {};
        if (sink.live) std.log.info("qwen3_5: adopt sink at {d} of {d} prompt tokens", .{ sink.at, sink.tokens.len });
    }
    // TF_MOM_KVDUMP (docs/12 S1): the same checkpoint lands in the dump directory; a failed dump never fails the save.
    if (kvdumpWriter.dumpDir() != null) {
        const e = t.e;
        const sb = e.g.stateBytes();
        const region: []const u8 = if (copy.dev == null and copy.host.bytes.len >= sb) copy.host.bytes[sb..] else &.{};
        const pt: u32 = if (cache.paged) |p| @intCast(p.tokens) else 0;
        if (kvdumpWriter.dump(e, t.io, s.prompt(), at, pt, region, copy.dev)) |w| {
            std.log.warn("qwen3_5: kvdump ckpt{d} at {d} tokens ({d} attention layers) -> {s}", .{ w.ckpt, at, e.g.attentions(), w.dir });
            e.gpa.free(w.dir);
        } else |err| {
            std.log.warn("qwen3_5: kvdump checkpoint at {d} tokens skipped: {s}", .{ at, @errorName(err) });
        }
    }
    return @ptrCast(copy);
}

/// A copy of `cache` at `at` rows, on the host or the `device`, naming the prompt's images that start before it.
pub fn saveCache(e: *Engine, cache: *Cache, at: usize, seen: []const Seen, device: bool) !*Copy {
    if (cache.len != at or cache.last != null or at > cache.cap) return error.NotAtMark;
    try cache.flush(e); // the copy holds the state with the pending kept rows landed
    const l: Layout = .{ .g = e.g, .format = formatOf(e), .cap = cache.cap, .at = at };
    var n: usize = 0;
    for (seen) |im| n += @intFromBool(im.at < at);
    const copy = try e.gpa.create(Copy);
    errdefer e.gpa.destroy(copy);
    const images = try e.gpa.alloc(Seen, n);
    errdefer e.gpa.free(images);
    n = 0;
    for (seen) |im| if (im.at < at) {
        images[n] = im;
        n += 1;
    };
    // a cuMemAlloc takes ~4 ms where pinning the same 650 MB takes ~180: on shared memory nothing needs pinning
    const len = l.total();
    const dev = if (device) try cuda.DeviceBuffer.alloc(e.d, len) else null;
    const host: cuda.HostBuffer = if (dev) |_| .{ .d = e.d, .bytes = &.{} } else try cuda.HostBuffer.alloc(e.d, len);
    copy.* = .{
        .host = host,
        .dev = dev,
        .len = len,
        .at = at,
        .delta = cache.delta,
        .format = l.format,
        .images = images,
    };
    errdefer drainThenFree(e, copy);
    var off: usize = 0;
    for (0..l.count()) |i| {
        const p = l.piece(i);
        if (i == 1 and wholePages(cache, l)) {
            try pageCopies(e, cache, l, copy, off, .save);
            break;
        }
        var it = Runs.of(cache, l, i, p);
        while (it.next()) |r| {
            const s, const o = .{ e.stream.handle, off + r.at };
            if (dev) |b|
                try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(b.ptr + o, r.dev, r.len, s), "cuMemcpyDtoDAsync")
            else
                try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(host.bytes.ptr + o, r.dev, r.len, s), "cuMemcpyDtoHAsync");
        }
        off += p.len;
    }
    try e.stream.synchronize();
    return copy;
}

/// A failed save's buffer: earlier copies may still be queued into it, so it is freed once the stream drains.
fn drainThenFree(e: *Engine, copy: *Copy) void {
    if (copy.dev) |*b| {
        cuda.completion.drain(e.d, e.stream, "qwen3_5: a failed save's device copy did not drain");
        return b.free();
    }
    e.stream.synchronize() catch {
        std.log.warn("qwen3_5: a failed save's host copy goes to the copy-source quarantine: its stream did not drain", .{});
        cuda.copy_source.retainPinned(e.d, copy.host, e.stream);
        return;
    };
    copy.host.free();
}

/// Whether `cache` can take the kept state: the same cache format and bytes, a position inside it, the same images.
pub fn restorable(e: *const Engine, cache: *const Cache, copy: *const Copy, seen: []const Seen) bool {
    if (copy.format != formatOf(e) or copy.at >= cache.cap) return false;
    const l: Layout = .{ .g = e.g, .format = copy.format, .cap = cache.cap, .at = copy.at };
    return copy.len == l.total() and copy.room() >= copy.len and sameImages(copy.images, seen, copy.at);
}

/// Make `cache` the kept state's: its next prompt chunk starts at the copy's position; refused before any copy.
pub fn restoreCache(e: *Engine, cache: *Cache, saved: *anyopaque, seen: []const Seen) !void {
    const copy: *Copy = @ptrCast(@alignCast(saved));
    if (copy.format != formatOf(e)) return error.KeptCacheFormat;
    if (!restorable(e, cache, copy, seen)) return error.BadReuse;
    const l: Layout = .{ .g = e.g, .format = copy.format, .cap = cache.cap, .at = copy.at };
    // the copies read the kept state until the stream drains: a restore that fails before then leaves it marked
    copy.reading = true;
    errdefer if (e.stream.synchronize()) |_| {
        copy.reading = false;
    } else |_| std.log.warn("qwen3_5: a failed restore's stream did not drain: its kept state's copy stays", .{});
    const host = copy.host.bytes.ptr;
    var off: usize = 0;
    for (0..l.count()) |i| {
        const p = l.piece(i);
        if (i == 1 and wholePages(cache, l)) {
            try pageCopies(e, cache, l, copy, off, .restore);
            break;
        }
        var it = Runs.of(cache, l, i, p);
        while (it.next()) |r| {
            const s, const o = .{ e.stream.handle, off + r.at };
            if (copy.dev) |b|
                try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(r.dev, b.ptr + o, r.len, s), "cuMemcpyDtoDAsync")
            else
                try e.d.check(e.d.api.cuMemcpyHtoDAsync_v2(r.dev, host + o, r.len, s), "cuMemcpyHtoDAsync");
        }
        off += p.len;
    }
    try e.stream.synchronize(); // the copy may be dropped once this returns
    copy.reading = false;
    cache.len = copy.at;
    cache.written = copy.at;
    cache.last = null;
    cache.pend = 0; // kept rows pending in the replaced state go with it
    cache.delta = copy.delta;
}

/// The store's restore: the owner's cache made the kept state's.
pub fn restore(ptr: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) anyerror!void {
    const t: *Target = @ptrCast(@alignCast(ptr));
    const s: *lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoStream));
    return restoreCache(t.e, t.caches.get(s) orelse return error.UnknownStream, saved, t.seen.get(s) orelse &.{});
}

/// Free a kept state's copy.
pub fn drop(ptr: *anyopaque, saved: *anyopaque) void {
    const t: *Target = @ptrCast(@alignCast(ptr));
    if (t.adopt_sink) |sink| if (sink.live and sink.copy.? == saved) sink.clear(t.e.gpa); // MoM M3: a retired copy is never adopted
    dropCopy(t.e, @ptrCast(@alignCast(saved)));
}

/// Frees a kept state's copy; after a failed restore whose copies no wait proves done, quarantine takes it.
pub fn dropCopy(e: *Engine, copy: *Copy) void {
    if (copy.reading) if (e.stream.synchronize()) |_| {
        copy.reading = false;
    } else |_| {};
    if (copy.dev) |*b| {
        const why = "qwen3_5: a kept state's device copy stays allocated: a restore's copies did not drain";
        if (copy.reading) std.log.warn(why, .{}) else b.free();
    } else if (copy.reading) {
        std.log.warn("qwen3_5: a kept state's host copy goes to the copy-source quarantine: a restore's copies did not drain", .{});
        cuda.copy_source.retainPinned(e.d, copy.host, e.stream);
    } else copy.host.free();
    e.gpa.free(copy.images);
    e.gpa.destroy(copy);
}

/// Cross-engine adoption (docs/13 M3 v1): a parent engine's kept state, saved
/// at `at`, re-laid into a nested engine's cache -- legal only where the two
/// engines are byte-identical (the nested checkpoint is a byte-copy of the
/// parent's layer prefix; the open-time weight check is the gate, not this).
/// The parent copy stores contiguous pieces in parent piece order; adoption
/// reads the prefix subset (state regions of the shared linear layers, then
/// the shared attention layers' kv pieces) and writes each through the
/// nested cache's own Runs/pageCopies addressing.
pub const AdoptMap = struct {
    /// One write: `src` bytes into the parent copy, `piece` index in the nested layout.
    pub const Write = struct { src: usize, len: usize, piece: usize };
    writes: []const Write,
    at: usize,

    /// The subset map over plain values (host-testable), or null where the
    /// geometries are not parent-prefix-identical: same format and kv_heads,
    /// the nested's layer kinds are the parent's first ones verbatim, state
    /// regions uniform and equal-sized.
    pub fn buildParams(gpa: std.mem.Allocator, pg: Geometry, ng: Geometry, fmt: Format, at: usize) ?AdoptMap {
        if (pg.kv_heads != ng.kv_heads) return null;
        if (ng.layers > pg.layers) return null;
        if (ng.deltas() > pg.deltas() or ng.attentions() > pg.attentions()) return null;
        for (0..ng.layers) |i| if (ng.linear[i] != pg.linear[i]) return null;
        const dn = ng.deltas();
        const dp = pg.deltas();
        if (dn == 0 or dp == 0) return null;
        if (pg.stateBytes() % dp != 0) return null;
        const region = pg.stateBytes() / dp;
        if (ng.stateBytes() != dn * region) return null;
        const pl: Layout = .{ .g = pg, .format = fmt, .cap = 0, .at = at };
        const nl: Layout = .{ .g = ng, .format = fmt, .cap = 0, .at = at };
        var w = std.ArrayList(Write).initCapacity(gpa, nl.count()) catch return null;
        w.appendAssumeCapacity(.{ .src = 0, .len = dn * region, .piece = 0 });
        var off: usize = pg.stateBytes(); // pieces after 0 lie past the state block
        for (1..nl.count()) |i| {
            const np = nl.piece(i);
            w.appendAssumeCapacity(.{ .src = off, .len = np.len, .piece = i });
            off += pl.piece(i).len; // same format/at/heads: parent's piece i has the same length
        }
        return .{ .writes = w.items, .at = at };
    }

    /// The engine pair's convenience: same format required, geometries from the engines.
    pub fn build(pe: *const Engine, ne: *const Engine, at: usize, cap_n: usize) ?AdoptMap {
        if (formatOf(pe) != formatOf(ne)) return null;
        _ = cap_n; // the nested cache sizes its own pieces; the map is addressing-independent
        return buildParams(ne.gpa, pe.g, ne.g, formatOf(ne), at);
    }

    pub fn free(m: *AdoptMap, gpa: std.mem.Allocator) void {
        gpa.free(m.writes);
        m.* = undefined;
    }
};

/// The parent copy can seed `cache` in the nested engine.
pub fn adoptable(pe: *const Engine, ne: *const Engine, copy: *const Copy, cap_n: usize) ?AdoptMap {
    if (copy.images.len != 0) return null; // v1 adopts text-only prefixes
    if (copy.at == 0 or copy.at >= cap_n) return null;
    const m = AdoptMap.build(pe, ne, copy.at, cap_n) orelse return null;
    var need: usize = 0;
    for (m.writes) |w| need = @max(need, w.src + w.len);
    if (need > copy.room()) {
        var mm = m;
        mm.free(ne.gpa);
        return null;
    }
    return m;
}

/// Write the adopted bytes into `cache` and make it stand at `at`, exactly as
/// a same-engine restore would. The parent copy is NOT owned here: it stays
/// in the parent's reuse store and may still be restored by its own engine.
pub fn adoptCache(ne: *Engine, cache: *Cache, copy: *Copy, m: *const AdoptMap) !void {
    const l: Layout = .{ .g = ne.g, .format = copy.format, .cap = cache.cap, .at = m.at };
    if (l.count() != m.writes.len or m.writes.len == 0) return error.BadAdoption;
    copy.reading = true;
    errdefer if (ne.stream.synchronize()) |_| {
        copy.reading = false;
    } else |_| std.log.warn("qwen3_5: a failed adopt's stream did not drain: the parent copy stays", .{});
    // the state block, piece by piece (never paged)
    {
        const p0 = l.piece(0);
        var it = Runs.of(cache, l, 0, p0);
        while (it.next()) |r| {
            const s, const o = .{ ne.stream.handle, m.writes[0].src + r.at };
            if (copy.dev) |b|
                try ne.d.check(ne.d.api.cuMemcpyDtoDAsync_v2(r.dev, b.ptr + o, r.len, s), "cuMemcpyDtoDAsync")
            else
                try ne.d.check(ne.d.api.cuMemcpyHtoDAsync_v2(r.dev, copy.host.bytes.ptr + o, r.len, s), "cuMemcpyHtoDAsync");
        }
    }
    if (wholePages(cache, l)) {
        // bf16 pages: one 2D copy a page over every kv piece at once -- pageCopies'
        // restore branch with the kept side read at the parent copy's kv offset
        const kv_off = m.writes[1].src;
        const pg = cache.paged.?;
        const row = head_dim * 2;
        const page_bytes = (Layout{ .g = l.g, .format = l.format, .cap = pg.tokens, .at = pg.tokens }).kvBytes();
        const pieces = l.count() - 1;
        var p: usize = 0;
        while (p * pg.tokens < l.at) : (p += 1) {
            const rows = @min(pg.tokens, l.at - p * pg.tokens);
            const dev = pg.base + pg.ids[p] * page_bytes;
            const o = kv_off + p * pg.tokens * row;
            const kept: cuda.abi.DevicePtr = if (copy.dev) |b| b.ptr + o else 0;
            const host: ?*u8 = if (copy.dev == null) &copy.host.bytes.ptr[o] else null;
            const kind: c_int = @intCast(if (copy.dev == null) cuda.abi.memory_type_host else cuda.abi.memory_type_device);
            const m2: cuda.abi.Memcpy2D = .{
                .src_type = kind, .src_host = host, .src = kept, .src_pitch = l.at * row,
                .dst = dev, .dst_pitch = pg.tokens * row, .width = rows * row, .height = pieces,
            };
            try ne.d.check(ne.d.api.cuMemcpy2DAsync_v2(&m2, ne.stream.handle), "cuMemcpy2DAsync");
        }
    } else for (m.writes[1..], 1..) |w, j| {
        const i = j + 1;
        const p = l.piece(i);
        var it = Runs.of(cache, l, i, p);
        while (it.next()) |r| {
            const s, const o = .{ ne.stream.handle, w.src + r.at };
            if (copy.dev) |b|
                try ne.d.check(ne.d.api.cuMemcpyDtoDAsync_v2(r.dev, b.ptr + o, r.len, s), "cuMemcpyDtoDAsync")
            else
                try ne.d.check(ne.d.api.cuMemcpyHtoDAsync_v2(r.dev, copy.host.bytes.ptr + o, r.len, s), "cuMemcpyHtoDAsync");
        }
    }
    try ne.stream.synchronize();
    copy.reading = false;
    cache.len = m.at;
    cache.written = m.at;
    cache.last = null;
    cache.pend = 0;
    cache.delta = copy.delta;
}

// ---- MoM M3: the cross-engine adoption trigger (docs/13) ----

/// Live qwen3_5 lane targets (registered by cuda_native.zig); the pair is found at adopt time, never wired at open.
var open_targets: [4]?*Target = .{ null, null, null, null };

/// The adoption's byte audit (diagnostic): the adopted parent copy and its map, awaiting the nested
/// engine's own save of the same prompt, which is the ground truth of what the engine would have
/// written. KV rows are immutable once written, so the adopted kv pieces must match bit for bit; the
/// state block is mutated by every forward and is judged only by elimination. Set by `adoptMiss`,
/// consumed by the next save on the nested target. TF_MOM_ADOPT_HOST keeps saves host-backed so the
/// bytes are readable.
pub const AdoptAudit = struct {
    parent_copy: *Copy,
    map: AdoptMap,
    parent_layers: usize,
    pg: Geometry,
};
var audit: ?AdoptAudit = null;

pub fn takeAudit() ?AdoptAudit {
    const a = audit;
    audit = null;
    return a;
}

pub fn register(t: *Target) void {
    for (&open_targets) |*slot| if (slot.* == null) {
        slot.* = t;
        return;
    };
}

pub fn unregister(t: *Target) void {
    for (&open_targets) |*slot| if (slot.* == t) {
        slot.* = null;
    };
}

/// TF_MOM_ADOPT: the trigger is on for every qwen3_5 target; roles (parent vs nested) are decided per call by layer count.
pub fn adoptEnabled() bool {
    return std.c.getenv("TF_MOM_ADOPT") != null;
}

const Candidate = struct { from: u32, copy: *Copy, map: AdoptMap, pe: *const Engine };

/// The core's cold-miss path (prompt_cache.beginUser) asks: does a registered, deeper engine keep this prompt's
/// prefix? Takes the deepest candidate whose kept copy passes `adoptable` (same family shape, text-only, fits).
pub fn adoptMiss(ptr: *anyopaque, owner: ?*anyopaque, prompt: []const u32) anyerror!?u32 {
    if (!adoptEnabled()) return null;
    if (std.c.getenv("TF_MOM_ADOPT_AUDIT") != null) return null; // audit-only boots: the sink and host copies stay, the adopt does not
    const t: *Target = @ptrCast(@alignCast(ptr));
    const s: ?*lanes.Stream = if (owner) |o| @ptrCast(@alignCast(o)) else null;
    if (s == null) {
        std.log.info("qwen3_5: adopt miss skipped: no stream owner", .{});
        return null;
    }
    const cache = t.caches.get(s.?) orelse {
        std.log.info("qwen3_5: adopt miss skipped: stream has no cache yet", .{});
        return null;
    };
    if (cache.len != 0 or cache.last != null) {
        std.log.info("qwen3_5: adopt miss skipped: cache not fresh (len {d})", .{cache.len});
        return null;
    }
    var best: ?Candidate = null;
    for (open_targets) |o| {
        const p = o orelse continue;
        if (p == t or p.e.g.layers <= t.e.g.layers) continue; // the parent is the deeper stack
        const sink = p.adopt_sink orelse continue;
        if (!sink.live) continue;
        const copy: *Copy = @ptrCast(@alignCast(sink.copy orelse continue));
        const at: usize = copy.at;
        if (at + 1 > prompt.len or at + 1 > sink.tokens.len) {
            std.log.info("qwen3_5: adopt miss skipped: sink at {d} too deep for {d}-token prompt", .{ at, prompt.len });
            continue;
        }
        if (!std.mem.eql(u32, sink.tokens[0 .. at + 1], prompt[0 .. at + 1])) {
            std.log.info("qwen3_5: adopt miss skipped: token prefix mismatch at {d}", .{at});
            continue;
        }
        const m = adoptable(p.e, t.e, copy, cache.cap) orelse {
            std.log.info("qwen3_5: adopt miss skipped: adoptable refused (at {d})", .{at});
            continue;
        };
        if (best) |b| {
            if (m.at <= b.map.at) {
                var mm = m;
                mm.free(t.e.gpa);
                continue;
            }
            var old = b;
            old.map.free(t.e.gpa);
        }
        best = .{ .from = @intCast(m.at), .copy = copy, .map = m, .pe = p.e };
    }
    var b = best orelse return null;
    defer b.map.free(t.e.gpa);
    try adoptCache(t.e, cache, b.copy, &b.map);
    // arm the byte audit: the nested engine's next save of this prompt is the ground truth (diagnostic)
    if (AdoptMap.buildParams(t.e.gpa, b.pe.g, t.e.g, formatOf(t.e), b.map.at)) |dm| {
        audit = .{ .parent_copy = b.copy, .map = dm, .parent_layers = b.pe.g.layers, .pg = b.pe.g };
    }
    std.log.warn("qwen3_5: adopted the paired engine's kept state at {d} tokens", .{b.map.at});
    return b.from;
}

/// Compare an adopted parent copy against the nested engine's own save of the same prompt, piece by
/// piece. KV rows are immutable, so an adopted kv piece must be bit-identical to what the engine
/// itself writes; the state block was mutated by the resumed forward and is skipped.
fn diffAudit(a: AdoptAudit, copy: *Copy, compare_state: bool) void {
    if (copy.dev != null or a.parent_copy.dev != null) {
        std.log.warn("qwen3_5: adopt audit skipped: a copy is device-backed (set TF_MOM_ADOPT_HOST)", .{});
        return;
    }
    var off: usize = 0;
    var bad: usize = 0;
    for (a.map.writes, 0..) |w, i| {
        defer off += w.len;
        if (i == 0 and !compare_state) continue; // after a resume the state block holds the forward's writes, not the copy's
        const src = a.parent_copy.host.bytes[w.src..][0..w.len];
        const dst = copy.host.bytes[off..][0..w.len];
        if (!std.mem.eql(u8, src, dst)) {
            bad += 1;
            var k: usize = 0;
            while (k < w.len and src[k] == dst[k]) : (k += 1) {}
            std.log.warn("qwen3_5: adopt audit piece {d} DIFFERS: first byte {d} of {d}", .{ i, k, w.len });
            if (i == 0) { // the state block: name the regions and the head (region = recurrence + conv tail)
                const gp = a.pg;
                const region = gp.recBytes() + gp.convBytes();
                const per_head = gp.v_heads * gp.dk * gp.dk * 4 / gp.v_heads;
                const reg = k / region;
                std.log.warn("qwen3_5: adopt audit state: region {d} of {d} (rec {d} + conv {d}), byte {d} in region (head ~{d} of {d})", .{
                    reg, a.map.writes[0].len / region, gp.recBytes(), gp.convBytes(), k % region, (k % region) / per_head, gp.v_heads,
                });
            }
        }
    }
    const n = a.map.writes.len - 1 - if (compare_state) @as(usize, 0) else 1;
    std.log.warn("qwen3_5: adopt audit: {d} of {d} pieces bit-identical ({s})", .{ n - bad, n, if (compare_state) "cold" else "adopt" });
}

test "adoption map: the nested prefix takes the state head and the parent's first kv pieces" {
    const gpa = std.testing.allocator;
    const testing = std.testing;
    const pg = testGeometry(8, 4, 2); // attention at 3,7; six linear layers
    const ng = testGeometry(4, 4, 2); // attention at 3; three linear layers: the parent's first layers
    var m = AdoptMap.buildParams(gpa, pg, ng, .bf16, 5) orelse return error.TestUnexpectedResult;
    defer m.free(gpa);
    const nl: Layout = .{ .g = ng, .format = .bf16, .cap = 0, .at = 5 };
    try testing.expectEqual(nl.count(), m.writes.len);
    try testing.expectEqual(@as(usize, 0), m.writes[0].src);
    try testing.expectEqual(ng.deltas() * (pg.stateBytes() / pg.deltas()), m.writes[0].len);
    var off: usize = pg.stateBytes();
    for (m.writes[1..], 1..) |w, i| {
        try testing.expectEqual(off, w.src);
        try testing.expectEqual(nl.piece(i).len, w.len);
        try testing.expectEqual(i, w.piece);
        off += w.len;
    }
}

test "adoption map: refuses different heads, longer nested, and mismatched layer kinds" {
    const gpa = std.testing.allocator;
    const testing = std.testing;
    const pg = testGeometry(8, 4, 2);
    const heads = testGeometry(4, 4, 4);
    try testing.expect(AdoptMap.buildParams(gpa, pg, heads, .bf16, 5) == null);
    const longer = testGeometry(16, 4, 2);
    try testing.expect(AdoptMap.buildParams(gpa, pg, longer, .bf16, 5) == null);
    var kinds = testGeometry(4, 4, 2);
    kinds.linear[3] = true; // the parent's layer 3 is an attention layer
    try testing.expect(AdoptMap.buildParams(gpa, pg, kinds, .bf16, 5) == null);
    const dense = testGeometry(4, 1, 2); // all-attention parent slice: no linear layers to adopt
    try testing.expect(AdoptMap.buildParams(gpa, pg, dense, .bf16, 5) == null);
}

fn testGeometry(layers: usize, attention_every: usize, kv_heads: usize) Geometry {
    var g: Geometry = undefined;
    g.hidden = 5120;
    g.k_heads = 16;
    g.v_heads = 48;
    g.q_heads = 32;
    g.kv_heads = kv_heads;
    g.layers = layers;
    @memset(&g.linear, false);
    for (0..layers) |i| g.linear[i] = i % attention_every != attention_every - 1;
    return g;
}

test "a kept state holds the recurrences and tails whole and each attention head's rows it read" {
    const g = testGeometry(8, 4, 4); // the 27B's pattern: three linear layers, then attention
    const l: Layout = .{ .g = g, .format = .bf16, .cap = 2048, .at = 1000 };
    try std.testing.expectEqual(@as(usize, 1 + 2 * 2 * 4), l.count());
    try std.testing.expectEqual(@as(u64, g.stateBytes()), (Layout{ .g = g, .format = .bf16, .cap = 2048, .at = 0 }).total());
    try std.testing.expectEqual(@as(u64, g.stateBytes() + 2 * 2 * 4 * 1000 * head_dim * 2), l.total());
    // fp8: e4m3 rows and a scale each, 260 bytes a position and head
    const l8: Layout = .{ .g = g, .format = .fp8, .cap = 2048, .at = 1000 };
    try std.testing.expectEqual(@as(usize, 1 + 2 * 2 * 2 * 4), l8.count());
    try std.testing.expectEqual(@as(u64, g.stateBytes() + 2 * 2 * 4 * 1000 * (head_dim + 4)), l8.total());
}

test "every geometry's pieces lie inside its allocation, apart, with no fixed limit on layers or heads" {
    const a = std.testing.allocator;
    // 33 attention layers of 32 key heads passed the old fixed array; the widest the engine reads is 128 layers
    for ([_][3]usize{ .{ 8, 4, 4 }, .{ 99, 3, 32 }, .{ 33, 1, 32 }, .{ 128, 1, 32 }, .{ 24, 4, 2 } }) |shape| {
        const g = testGeometry(shape[0], shape[1], shape[2]);
        for ([_]Format{ .bf16, .fp8 }) |f| for ([_]usize{ 1, 7, 64 }) |at| {
            const l: Layout = .{ .g = g, .format = f, .cap = 64, .at = at };
            const end = l.kvBytes();
            if (comptime @typeInfo(@TypeOf(Geometry.kvBytes)).@"fn".param_types.len == 3) try std.testing.expectEqual(g.kvBytes(64, f == .fp8), end);
            const spans = try a.alloc([2]usize, l.count() - 1);
            defer a.free(spans);
            try std.testing.expectEqual(@as(usize, g.stateBytes()), l.piece(0).len);
            for (spans, 1..) |*sp, i| {
                const p = l.piece(i);
                try std.testing.expect(p.kv and p.len > 0 and p.off + p.len <= end);
                sp.* = .{ p.off, p.off + p.len };
            }
            std.mem.sort([2]usize, spans, {}, struct {
                fn lt(_: void, x: [2]usize, y: [2]usize) bool {
                    return x[0] < y[0];
                }
            }.lt);
            for (spans[1..], spans[0 .. spans.len - 1]) |sp, prev| try std.testing.expect(sp[0] >= prev[1]);
        };
    }
}

test "fp8 pieces follow qwen35_kv8.cu: a layer's key then value rows, then their scales" {
    const g = testGeometry(4, 4, 2); // one attention layer
    const l: Layout = .{ .g = g, .format = .fp8, .cap = 100, .at = 10 };
    // (key, head 1): rows at 1 * cap * 256, its scales after both blocks' rows, at 2 * 2 * cap * 256 + 1 * cap * 4
    try std.testing.expectEqual(Piece{ .kv = true, .off = 100 * 256, .len = 10 * 256 }, l.piece(1 + 2 * 1));
    try std.testing.expectEqual(Piece{ .kv = true, .off = 4 * 100 * 256 + 100 * 4, .len = 10 * 4 }, l.piece(2 + 2 * 1));
    // (value, head 0): the third block's rows and scales
    try std.testing.expectEqual(Piece{ .kv = true, .off = 2 * 100 * 256, .len = 10 * 256 }, l.piece(1 + 2 * 2));
    try std.testing.expectEqual(Piece{ .kv = true, .off = 4 * 100 * 256 + 2 * 100 * 4, .len = 10 * 4 }, l.piece(2 + 2 * 2));
}

test "a state resumes only a request with the same images before its position" {
    const d = [_]u64{ 1, 2, 3 };
    const kept = [_]Seen{.{ .at = 5, .id = d[0] }};
    // a later image is the pass's
    try std.testing.expect(sameImages(&kept, &.{ .{ .at = 5, .id = d[0] }, .{ .at = 40, .id = d[2] } }, 20));
    try std.testing.expect(!sameImages(&kept, &.{.{ .at = 5, .id = d[1] }}, 20)); // another image under the same ids
    try std.testing.expect(!sameImages(&kept, &.{.{ .at = 6, .id = d[0] }}, 20));
    try std.testing.expect(!sameImages(&kept, &.{}, 20)); // a text prompt never resumes an image's state
    // nor an image prompt a text state past its image
    try std.testing.expect(!sameImages(&.{}, &.{.{ .at = 5, .id = d[0] }}, 20));
    try std.testing.expect(sameImages(&.{}, &.{.{ .at = 5, .id = d[0] }}, 5)); // a text prefix before it
}

test "a restore refuses another cache format, another length, a position past the cache and other images" {
    var e: Engine = undefined;
    e.kit = null;
    e.g = testGeometry(8, 4, 4);
    if (comptime @hasField(Engine, "kv8")) e.kv8 = false;
    var cache: Cache = undefined;
    cache.cap = 64;
    cache.paged = null; // contiguous: restoreCache reads it, and undefined bytes can read as a page table
    const d: u64 = 9;
    var images = [_]Seen{.{ .at = 2, .id = d }};
    const l: Layout = .{ .g = e.g, .format = .bf16, .cap = 64, .at = 10 };
    const host = try std.testing.allocator.alignedAlloc(u8, .@"16", @intCast(l.total()));
    defer std.testing.allocator.free(host);
    var copy: Copy = .{ .host = .{ .d = undefined, .bytes = host }, .len = l.total(), .at = 10, .delta = 0, .format = .bf16, .images = &images };
    const same = [_]Seen{.{ .at = 2, .id = d }};
    try std.testing.expect(restorable(&e, &cache, &copy, &same));
    try std.testing.expect(!restorable(&e, &cache, &copy, &.{.{ .at = 2, .id = 8 }}));
    copy.format = .fp8; // a state kept from fp8 rows never lands in bf16 ones, nor the other way
    try std.testing.expect(!restorable(&e, &cache, &copy, &same));
    copy.format = .bf16;
    copy.len -= 1;
    try std.testing.expect(!restorable(&e, &cache, &copy, &same));
    copy.len += 1;
    cache.cap = 10;
    try std.testing.expect(!restorable(&e, &cache, &copy, &same));
}

/// A driver whose later copies and stuck waits fail with CUDA_ERROR_NOT_READY (check logs nothing), counting frees.
const RestoreFake = struct {
    var copies: usize = 0;
    var fail_at: usize = 1;
    var stuck = true;
    var freed: usize = 0;

    fn htod(_: cuda.abi.DevicePtr, _: ?*const anyopaque, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        copies += 1;
        return if (copies - 1 >= fail_at) cuda.abi.error_not_ready else cuda.abi.success;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return if (stuck) cuda.abi.error_not_ready else cuda.abi.success;
    }
    fn hostAlloc(p: *?*anyopaque, len: usize, _: c_uint) callconv(.c) cuda.abi.Result {
        p.* = (std.testing.allocator.alignedAlloc(u8, .@"16", len) catch return 2).ptr;
        return cuda.abi.success;
    }
    fn freeHost(_: ?*anyopaque) callconv(.c) cuda.abi.Result {
        freed += 1;
        return cuda.abi.success;
    }
};

test "a failed restore's host copy stays through a drop until a wait, the restore's or the drop's, drains the stream" {
    const log_level = std.testing.log_level;
    std.testing.log_level = .err;
    defer std.testing.log_level = log_level;
    var api: cuda.abi.Api = undefined;
    api.cuMemcpyHtoDAsync_v2 = RestoreFake.htod;
    api.cuStreamSynchronize = RestoreFake.sync;
    api.cuMemHostAlloc = RestoreFake.hostAlloc;
    api.cuMemFreeHost = RestoreFake.freeHost;
    const d: cuda.Driver = .{ .lib = undefined, .api = api };
    var e: Engine = undefined;
    e.kit = null;
    e.gpa = std.testing.allocator;
    e.d = &d;
    e.g = testGeometry(8, 4, 4);
    e.stream = .{ .d = &d, .handle = null };
    if (comptime @hasField(Engine, "kv8")) e.kv8 = false;
    var cache: Cache = undefined;
    cache.cap = 64;
    cache.paged = null; // contiguous: restoreCache reads it, and undefined bytes can read as a page table
    cache.kv = .{ .d = &d, .ptr = 0x10000, .len = 0 };
    cache.state = .{ .d = &d, .ptr = 0x20000, .len = 0 };
    const l: Layout = .{ .g = e.g, .format = .bf16, .cap = 64, .at = 10 };
    // the stream stuck through the restore and the drop, stuck only through the restore, or never
    for ([_][2]bool{ .{ true, true }, .{ true, false }, .{ false, false } }) |stuck| {
        RestoreFake.copies = 0;
        RestoreFake.freed = 0;
        RestoreFake.stuck = stuck[0];
        const copy = try std.testing.allocator.create(Copy);
        copy.* = .{ .host = try cuda.HostBuffer.alloc(&d, @intCast(l.total())), .len = l.total(), .at = 10, .delta = 0, .format = .bf16, .images = try std.testing.allocator.alloc(Seen, 0) };
        const pinned = copy.host.bytes; // the fake's free keeps them: the test gives them back
        defer std.testing.allocator.free(pinned);
        try std.testing.expectError(error.NotReady, restoreCache(&e, &cache, copy, &.{}));
        try std.testing.expectEqual(@as(usize, 2), RestoreFake.copies); // the state queued, the first head's rows refused
        RestoreFake.stuck = stuck[1];
        const before = cuda.copy_source.retained();
        dropCopy(&e, copy);
        // a stream that will not drain may still read the copy: quarantine holds it, counted; once drained, it is freed
        try std.testing.expectEqual(@as(usize, @intFromBool(!stuck[1])), RestoreFake.freed);
        try std.testing.expectEqual(before + @intFromBool(stuck[1]), cuda.copy_source.retained());
        RestoreFake.stuck = false;
        try e.stream.synchronize();
        try std.testing.expectEqual(@as(usize, 1), RestoreFake.freed);
        try std.testing.expectEqual(before, cuda.copy_source.retained());
    }
}

/// A driver that counts device and pinned allocations, frees, and copies by direction; every call succeeds.
const DeviceFake = struct {
    var allocs: usize = 0;
    var pinned: usize = 0;
    var frees: usize = 0;
    var dtod: usize = 0;
    var host_copies: usize = 0;

    fn alloc(p: *cuda.abi.DevicePtr, _: usize) callconv(.c) cuda.abi.Result {
        allocs += 1;
        p.* = 0x40000;
        return cuda.abi.success;
    }
    fn free(_: cuda.abi.DevicePtr) callconv(.c) cuda.abi.Result {
        frees += 1;
        return cuda.abi.success;
    }
    fn hostAlloc(_: *?*anyopaque, _: usize, _: c_uint) callconv(.c) cuda.abi.Result {
        pinned += 1;
        return cuda.abi.error_not_ready;
    }
    fn copy(_: cuda.abi.DevicePtr, _: cuda.abi.DevicePtr, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        dtod += 1;
        return cuda.abi.success;
    }
    fn htod(_: cuda.abi.DevicePtr, _: ?*const anyopaque, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        host_copies += 1;
        return cuda.abi.success;
    }
    fn dtoh(_: ?*anyopaque, _: cuda.abi.DevicePtr, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        host_copies += 1;
        return cuda.abi.success;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return cuda.abi.success;
    }
};

test "a device save pins nothing: it copies on the device both ways, and its drop frees the device buffer" {
    var api: cuda.abi.Api = undefined;
    api.cuMemAlloc_v2 = DeviceFake.alloc;
    api.cuMemFree_v2 = DeviceFake.free;
    api.cuMemHostAlloc = DeviceFake.hostAlloc;
    api.cuMemcpyDtoDAsync_v2 = DeviceFake.copy;
    api.cuMemcpyHtoDAsync_v2 = DeviceFake.htod;
    api.cuMemcpyDtoHAsync_v2 = DeviceFake.dtoh;
    api.cuStreamSynchronize = DeviceFake.sync;
    const d: cuda.Driver = .{ .lib = undefined, .api = api };
    var e: Engine = undefined;
    e.kit = null;
    e.gpa = std.testing.allocator;
    e.d = &d;
    e.g = testGeometry(8, 4, 4);
    e.stream = .{ .d = &d, .handle = null };
    if (comptime @hasField(Engine, "kv8")) e.kv8 = false;
    var cache: Cache = undefined;
    cache.cap = 64;
    cache.len = 10;
    cache.last = null;
    cache.pend = 0;
    cache.delta = 3;
    cache.paged = null;
    cache.kv = .{ .d = &d, .ptr = 0x10000, .len = 0 };
    cache.state = .{ .d = &d, .ptr = 0x20000, .len = 0 };
    const l: Layout = .{ .g = e.g, .format = .bf16, .cap = 64, .at = 10 };
    const copy = try saveCache(&e, &cache, 10, &.{}, true);
    try std.testing.expectEqual(@as(usize, 1), DeviceFake.allocs);
    try std.testing.expectEqual(@as(usize, l.count()), DeviceFake.dtod); // a copy a piece, contiguous
    try std.testing.expectEqual(@as(usize, l.total()), copy.room());
    try restoreCache(&e, &cache, copy, &.{});
    try std.testing.expectEqual(@as(usize, 2 * l.count()), DeviceFake.dtod);
    try std.testing.expectEqual(@as(i32, 3), cache.delta);
    dropCopy(&e, copy);
    try std.testing.expectEqual(@as(usize, 1), DeviceFake.frees);
    try std.testing.expectEqual(@as(usize, 0), DeviceFake.pinned + DeviceFake.host_copies);
}

test "adopt sink: record owns a token copy, clear and re-record free it" {
    const a = std.testing.allocator;
    var sink: AdoptSink = .{};
    defer sink.clear(a);
    var copy_a: u8 = 0;
    var copy_b: u8 = 0;
    try sink.record(a, &copy_a, 12, &.{ 1, 2, 3 });
    try std.testing.expect(sink.live);
    try std.testing.expectEqual(@as(?*anyopaque, @ptrCast(&copy_a)), sink.copy);
    try std.testing.expectEqual(@as(u32, 12), sink.at);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3 }, sink.tokens);
    // re-record takes over: the old token slice is freed, the record names the new copy
    try sink.record(a, &copy_b, 20, &.{ 4, 5 });
    try std.testing.expectEqual(@as(?*anyopaque, @ptrCast(&copy_b)), sink.copy);
    try std.testing.expectEqualSlices(u32, &.{ 4, 5 }, sink.tokens);
    // clear retires it whole (the drop hook's path)
    sink.clear(a);
    try std.testing.expect(!sink.live);
    try std.testing.expectEqual(@as(?*anyopaque, null), sink.copy);
    try std.testing.expectEqual(@as(usize, 0), sink.tokens.len);
}
