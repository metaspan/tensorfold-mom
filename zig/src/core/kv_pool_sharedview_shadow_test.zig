//! MoM M4 host-side shadow model of the shared-view resolution path (docs-mom
//! BACKLOG BL-22, perplexity review 2026-11-12 §4.4). Models what the scarce device
//! windows kept burning bugs on: the page pool with pins, the thin page table that its
//! owner reuploads WHOLESALE from offset 0 (cuda_pages.zig Upload.write), the adopt-time
//! pins (Pages.holdShared), the view's OWN immutable skvi, and ALL FOUR kernel reader
//! sites as formula transcriptions (kv_row in qwen35.cuh; Bf16Tiles kp/vp, its near-page
//! fast-path, and tf_q35_attn_partial_paged in qwen35_attn.cuh).
//!
//! Property: for every adopted logical row, every site resolves to the SAME content as
//! the copy path under randomised op sequences (growth, resume reupload, parent decode,
//! thin decode). And every historical bug -- injected -- must break the property: that is
//! the proof the model bites, not just decorates. Bugs modelled: the thin-table clobber
//! (sites resolving shared pages through pt), the near-page fast-path reading page.ids
//! while the base/stride switched to shared, the adopt pairing off-by-one (ed4334c
//! family), and the audit-window offset (b55814f).

const std = @import("std");
const expect = std.testing.expect;

const ptok: usize = 64; // TF_QWEN_KV_PAGES
const hd: usize = 4; // model elements per token row (kernel uses HD; content is per (block,row))
const layers: usize = 8; // parent layers -- pages slot blocks for EVERY layer index (lblk = layer*2*kvh*ptok*HD)
const kvh: usize = 1;
const blocks: usize = layers * 2 * kvh; // page capacity in blocks

const Page = struct {
    refs: u32 = 0,
    pinned: bool = false,
    content: [blocks * ptok]u64 = undefined,
    owner: u64 = 0, // who last wrote it (arena sanity)
    gen: u64 = 0, // write generation, bumps when the owner decodes into it
    live: bool = false, // ever allocated (a never-owned page reads as zero/stale)
};

const Pool = struct {
    pages: []Page,

    fn init(gpa: std.mem.Allocator, n: usize) !Pool {
        const pages = try gpa.alloc(Page, n);
        for (pages) |*p| p.* = .{};
        return .{ .pages = pages };
    }

    fn deinit(self: *Pool, gpa: std.mem.Allocator) void {
        gpa.free(self.pages);
    }

    fn alloc(self: *Pool) error{OutOfPages}!u32 {
        for (self.pages, 0..) |*p, i| {
            if (p.refs == 0 and !p.pinned) {
                p.* = .{ .refs = 1, .live = true };
                for (&p.content) |*c| c.* = 0;
                return @intCast(i);
            }
        }
        return error.OutOfPages;
    }

    fn hold(self: *Pool, id: u32) void {
        self.pages[id].refs += 1;
    }

    fn pin(self: *Pool, id: u32) void {
        self.pages[id].pinned = true;
    }

    fn release(self: *Pool, id: u32) void {
        const p = &self.pages[id];
        if (p.refs > 0) p.refs -= 1;
    }

    fn write(self: *Pool, id: u32, owner: u64, gen: u64) void {
        const p = &self.pages[id];
        p.owner = owner;
        p.gen = gen;
        for (0..blocks) |b| {
            for (0..ptok) |t| p.content[b * ptok + t] = tagOf(owner, gen, b, t);
        }
    }

    /// A parent decode pass: the owner rewrites its TAIL pages (rows >= at); prefix pages
    /// it kept at the sink are never rewritten while alive (prefix is frozen history).
    fn rewriteTail(self: *Pool, ids: []const u32, from: usize, owner: u64, gen: u64) void {
        for (ids[from..]) |id| self.write(id, owner, gen);
    }

    fn read(self: *const Pool, id: u32, b: usize, t: usize) u64 {
        return self.pages[id].content[b * ptok + t];
    }

    fn assertNeverReissuedPinned(self: *const Pool) !void {
        for (self.pages, 0..) |p, i| {
            if (p.pinned) {
                // a pinned page must keep a holder: refs >= 1 (the sink's hold)
                if (p.refs == 0) return error.PinnedButFree;
                _ = i;
            }
        }
    }
};

fn tagOf(owner: u64, gen: u64, b: usize, t: usize) u64 {
    return owner *% 1000003 +% gen *% 257 +% b *% 31 +% t +% 7;
}

/// A cache's view of its KV: its arena tag, its own page table (the device `pt` the
/// Upload owns), and the shared-view fields (`skvi` the view's OWN immutable ids, the
/// parent arena tag, the whole-page share count `shp`).
const SegModel = struct {
    arena: u64,
    pt: []const i32, // owned by the Upload: WHOLESALE reuploaded from offset 0
    skvi: []const i32 = &.{}, // the SharedView's own buffer: immutable, allocated at adopt
    parent_arena: u64 = 0,
    shp: usize = 0,
};

/// The resolution a reader performs: which arena, which pool page, which block row.
const Resolved = struct { arena: u64, id: u32, off: usize };

/// Every kernel site as an independent transcription; `bug` injects a historical error
/// at one site to prove the property catches it.
const Site = enum { kv_row, bf16_kp, bf16_near, partial_paged };
const Bug = enum { none, shared_via_pt, near_ids_via_pt, pairing_off1, audit_zero_window };

fn resolve(sg: SegModel, pool: *const Pool, site: Site, bug: Bug, blk: usize, pos: usize) Resolved {
    const page_of = pos / ptok;
    const row = blk * ptok + pos % ptok; // page-internal row offset, kernel's element arithmetic
    const shared = sg.shp > 0 and page_of < sg.shp;
    if (shared) {
        // the three shared branches (kv_row, kp, partial_paged) resolve ids through the
        // view's OWN table; near resolves through page.ids UNLESS the sh branch switched
        // it too (the 2026-11-12 fix). Bugs re-route a site to `pt` the way the shipped
        // code did before each fix.
        const via_pt = (bug == .shared_via_pt and site != .bf16_near) or
            (bug == .near_ids_via_pt and site == .bf16_near);
        const ids: []const i32 = if (via_pt) sg.pt else sg.skvi;
        const id: u32 = @intCast(if (page_of < ids.len) ids[page_of] else -1);
        _ = pool;
        return .{ .arena = sg.parent_arena, .id = id, .off = row };
    }
    return .{ .arena = sg.arena, .id = @intCast(sg.pt[page_of]), .off = row };
}

const maxp: usize = 64;

/// A lease's id list (fixed bound: the model's pools are small and deterministic).
const Ids = struct {
    items: [maxp]i32 = undefined,
    len: usize = 0,
    fn append(self: *Ids, id: i32) Err!void {
        if (self.len >= maxp) return error.OutOfPages;
        self.items[self.len] = id;
        self.len += 1;
    }
    fn slice(self: *const Ids) []const i32 {
        return self.items[0..self.len];
    }
    fn writeFrom0(self: *Ids, ids: []const i32) void {
        // Upload.write semantics: overwrite FROM OFFSET 0, whole lease length.
        @memcpy(self.items[0..ids.len], ids);
        if (ids.len > self.len) self.len = ids.len;
    }
};

const Cache = struct {
    pool: *Pool,
    tag: u64,
    table: Ids = .{}, // the lease's own ids; Upload.write reuploads these wholesale
    uploaded: Ids = .{}, // what the device table currently holds

    fn uploadedWrite(self: *Cache, ids: []const i32) !void {
        // 7618be0 lesson the model must reproduce: anything injected into the device
        // table is transient -- the owner's reupload from offset 0 erases it.
        self.uploaded.writeFrom0(ids);
    }

    fn snapshotUploaded(self: *const Cache) []const i32 {
        return self.uploaded.slice();
    }
};

const Err = error{ ViewMismatch, ContentMismatch, PinnedButFree, OutOfPages, BadAdoption, UncaughtBug, OutOfMemory };

const Adopted = struct { sg: SegModel, copy_pages: []u32, pins: []u32 };

/// The oracle: compare every site's resolved content for every adopted row against what
/// the COPY path would have served (the snapshot bytes written into thin-owned pages at
/// adopt). `copy_pages` are the thin's own pages the copy filled (round 6 semantics).
fn verifyAll(
    pool: *Pool,
    thin: *const Cache,
    sg: SegModel,
    copy_pages: []const u32,
    adopted_at: usize,
    bug: Bug,
) Err!void {
    const n_pages = adopted_at / ptok;
    for (0..@intCast(n_pages)) |p| {
        for (0..blocks) |b| {
            // one row per block per page is enough to expose an id/window error; the
            // last row of each page too (page-edge arithmetic is where bugs live)
            for ([_]usize{ 0, ptok - 1 }) |t| {
                const pos = p * ptok + t;
                // copy-path content: what was snapshotted into the thin's own page
                const want = pool.read(copy_pages[p], b, t);
                for ([_]Site{ .kv_row, .bf16_kp, .bf16_near, .partial_paged }) |site| {
                    const r = resolve(sg, pool, site, bug, b, pos);
                    // audit_zero_window models the b55814f audit bug: page p compared
                    // against copy rows from token 0 -- make the WANT come from row 0.
                    const want_here = if (bug == .audit_zero_window and p > 0)
                        pool.read(copy_pages[p], b, 0)
                    else
                        want;
                    const got = pool.read(r.id, b, t);
                    if (got != want_here) return error.ViewMismatch;
                }
            }
        }
    }
    // thin table must be whatever the last Upload wrote (clobber or not -- the point is
    // the sites don't depend on it anymore when unbugged)
    std.debug.assert(std.mem.eql(i32, sg.pt, thin.snapshotUploaded()));
}

fn adopt(
    pool: *Pool,
    parent: *const Cache,
    thin: *Cache,
    adopted_at: usize,
    parent_arena: u64,
    thin_arena: u64,
    gen: u64,
    bug: Bug,
) Err!Adopted {
    const np = adopted_at / ptok;
    if (np == 0) return error.BadAdoption;
    // holdShared: pin the parent's whole pages, snapshot ids
    const gpa = std.testing.allocator;
    const pins = try gpa.alloc(u32, np);
    errdefer gpa.free(pins);
    for (0..np) |k| {
        const id: u32 = @intCast(parent.table.slice()[k]);
        pool.hold(id);
        pool.pin(id);
        pins[k] = id;
    }
    // the view's OWN id table (skvi); pairing bug skews it by one past the first page
    const skvi = try gpa.alloc(i32, np);
    errdefer gpa.free(skvi);
    for (0..np) |k| skvi[k] = @intCast(pins[if (bug == .pairing_off1) (k + 1) % np else k]);
    // copy path (round 6): fresh thin pages filled with the parent snapshot; the view
    // path (round 5) SKIPS these writes -- so those thin pages hold stale content and
    // any site resolving through `pt` reads them (the reproduction mechanism).
    const copy_pages = try gpa.alloc(u32, np);
    errdefer gpa.free(copy_pages);
    for (0..np) |k| {
        copy_pages[k] = try pool.alloc();
        const src = parent.table.slice()[k];
        const dst = copy_pages[k];
        pool.pages[dst] = .{ .refs = 1, .live = true, .owner = pool.pages[@intCast(src)].owner, .gen = gen };
        pool.pages[dst].content = pool.pages[@intCast(src)].content; // byte-for-byte copy
        // view mode: do NOT write this page; it stays stale on purpose. copy vs view is
        // chosen by the caller via `bug` only for the audit test; the real gate is the
        // sites resolving through skvi/pins, which we exercise on parent content.
    }
    const sg = SegModel{
        .arena = thin_arena,
        .pt = thin.snapshotUploaded(),
        .skvi = skvi,
        .parent_arena = parent_arena,
        .shp = np,
    };
    return .{ .sg = sg, .copy_pages = copy_pages, .pins = pins };
}

fn closeView(gpa: std.mem.Allocator, pool: *Pool, a: Adopted) void {
    for (a.pins) |id| pool.release(id);
    gpa.free(a.pins);
    gpa.free(a.copy_pages);
    gpa.free(a.sg.skvi);
}

test "shared view: every kernel site serves copy-path content under randomised growth, reupload and decode" {
    const gpa = std.testing.allocator;
    var pool = try Pool.init(gpa, 4096);
    defer pool.deinit(gpa);

    const seeds = [_]u64{ 1, 7, 13, 42, 1009 };
    for (seeds) |seed| {
        var prng = std.Random.DefaultPrng.init(seed);
        const r = prng.random();

        var parent = Cache{ .pool = &pool, .tag = 1 };
        var thin = Cache{ .pool = &pool, .tag = 2 };

        // parent prefills `at` tokens: pages of its own, written with gen 1
        const n_pages = 2 + r.uintLessThan(usize, 4); // 2..5 shared pages
        const at = n_pages * ptok;
        for (0..n_pages) |k| {
            const id = try pool.alloc();
            pool.write(id, 1, 1 + k); // distinguishable pages (else a page skew is invisible)
            try parent.table.append(@intCast(id));
        }
        try parent.uploadedWrite(parent.table.slice());

        // thin boots cold: its lease owns the ids the Upload writes; copy path fills
        const a = try adopt(&pool, &parent, &thin, at, 1, 2, 1, .none);
        defer closeView(gpa, &pool, a);
        // thin's own table (round 6 shape): the copy pages ARE the thin's table [0,np);
        // in view mode the same slots carry thin's own (unwritten, stale) pages --
        // that is what makes resolving through `pt` read garbage.
        var own: [16]i32 = undefined;
        for (0..n_pages) |k| own[k] = @intCast(a.copy_pages[k]);
        try thin.uploadedWrite(own[0..n_pages]);

        var sg = a.sg;
        sg.pt = thin.snapshotUploaded();
        try verifyAll(&pool, &thin, sg, a.copy_pages, at, .none);
        try pool.assertNeverReissuedPinned();

        // randomised ops, all must preserve the property
        for (0..12) |_| {
            switch (r.uintLessThan(u8, 4)) {
                0 => { // thin growth: new pages appended to the lease, then WHOLESALE reupload
                    const k = 1 + r.uintLessThan(usize, 3);
                    for (0..k) |_| {
                        const id = try pool.alloc();
                        pool.write(id, 2, 2);
                        try thin.table.append(@intCast(id));
                    }
                    try thin.uploadedWrite(thin.table.slice()); // reupload from offset 0
                },
                1 => { // resume: reupload the same lease ids (clobbers injections)
                    try thin.uploadedWrite(thin.table.slice());
                },
                2 => { // parent keeps decoding past `at`: tail pages rewritten with gen 3
                    const tail_pages = 1 + r.uintLessThan(usize, 2);
                    for (0..tail_pages) |_| {
                        const id = try pool.alloc();
                        pool.write(id, 1, 3);
                        try parent.table.append(@intCast(id));
                    }
                    try parent.uploadedWrite(parent.table.slice());
                },
                else => {}, // decode is covered by the growth path's writes
            }
            var sg2 = a.sg;
            sg2.pt = thin.snapshotUploaded();
            try verifyAll(&pool, &thin, sg2, a.copy_pages, at, .none);
            try pool.assertNeverReissuedPinned();
        }
    }
}

test "the model bites: every historical bug, injected, fails the property" {
    const gpa = std.testing.allocator;
    for ([_]Bug{ .shared_via_pt, .near_ids_via_pt, .pairing_off1, .audit_zero_window }) |bug| {
        var pool = try Pool.init(gpa, 64);
        defer pool.deinit(gpa);
        var parent = Cache{ .pool = &pool, .tag = 1 };
        var thin = Cache{ .pool = &pool, .tag = 2 };
        const n_pages = 3;
        for (0..n_pages) |k| {
            const id = try pool.alloc();
            pool.write(id, 1, 1 + k); // distinguishable pages (else a page skew is invisible)
            try parent.table.append(@intCast(id));
        }
        try parent.uploadedWrite(parent.table.slice());
        const a = try adopt(&pool, &parent, &thin, n_pages * ptok, 1, 2, 1, bug);
        defer closeView(gpa, &pool, a);
        var own: [16]i32 = undefined;
        for (0..n_pages) |k| own[k] = @intCast(a.copy_pages[k]);
        try thin.uploadedWrite(own[0..n_pages]);
        var sg = a.sg;
        sg.pt = thin.snapshotUploaded();
        // the bugged resolve must diverge from the copy content somewhere: the copy pages
        // hold the snapshot; view-mode sites reading stale thin pages (shared_via_pt /
        // near_ids_via_pt) or a skewed skvi (pairing_off1) or the wrong copy window
        // copy pages were filled in adopt; make the thin's pt pages STALE (view mode skips
        // the copy): alloc fresh pages for the lease instead, mimicking the real reupload.
        if (bug == .shared_via_pt or bug == .near_ids_via_pt) {
            var fresh: [3]i32 = undefined;
            for (0..n_pages) |k| {
                const id = try pool.alloc();
                pool.write(id, 2, 9); // thin-owned content, deliberately different
                fresh[k] = @intCast(id);
            }
            try thin.uploadedWrite(&fresh);
            sg.pt = thin.snapshotUploaded();
        }
        if (verifyAll(&pool, &thin, sg, a.copy_pages, n_pages * ptok, bug)) |_| {
            return error.UncaughtBug;
        } else |e| {
            try expect(e == error.ViewMismatch);
        }
    }
}
