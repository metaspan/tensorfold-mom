//! Mixture-of-Models request router (Track B, B2 pure policy core).
//!
//! Port of `experiments/mom_router.py` — the shared policy spec — so Track A and the
//! native fork keep ONE decision table. Pure: no I/O, no allocation on the decide path
//! (reason strings are static, lanes are borrowed), so the <5 ms budget from the brief is
//! dominated by the caller's own parsing, not the decision.
//!
//! Decision order (docs/00-design-brief.md "Routing Policy (v1)"):
//!   1. explicit registered id (or alias)        -> that lane   reason="explicit"
//!   2. sticky conversation hit (when enabled)   -> that lane   reason="sticky"
//!   3. trivial rules                            -> trivial lane reason="pattern:<p>"/"short"
//!   4. otherwise                                -> default lane reason="default"
//! An unknown non-empty explicit id is a hard error, replacing the silent fallback in
//! openai.zig replyModel — a caller asking for a model this server does not serve is
//! told so, never quietly answered by a different model.
const std = @import("std");

pub const max_lanes = 16;

pub const Role = enum { trivial, default_lane, other };

pub const Lane = struct {
    id: []const u8,
    role: Role,
    aliases: []const []const u8 = &.{},
    enabled: bool = true,

    fn matches(self: Lane, model: []const u8) bool {
        // Case-SENSITIVE exact id/alias match, per the Python spec: the model field is
        // a client-supplied identifier, and a silent case-fold could resolve two ids
        // that only differ in case to the wrong lane.
        if (std.mem.eql(u8, self.id, model)) return true;
        for (self.aliases) |a| if (std.mem.eql(u8, a, model)) return true;
        return false;
    }
};

pub const Config = struct {
    lanes: []const Lane,
    trivial_max_chars: usize = 240,
    sticky_enabled: bool = true,
    sticky_capacity: usize = 1024,
};

pub const RouteError = error{UnknownModel};

/// First trivial rule hit, or null. Mirrors trivial_reason() byte for byte:
/// tool requests are never trivial; greeting patterns win; code fences/hints
/// disqualify; short single-line text is trivial UNLESS it is multi-sentence or
/// contains a work verb (Gate A finding 2026-11-10: without the verb bail,
/// "Explain the CAP theorem…" leaked 40/60 real prompts to the small lane).
pub fn trivialReason(text: []const u8, cfg: Config) ?[]const u8 {
    const trimmed = std.mem.trim(u8, text, " \t\r\n");
    if (trimmed.len == 0) return null;
    for (pattern_list) |pat| {
        if (matchesPattern(trimmed, pat)) return pat; // pattern_list entries are "pattern:<p>" labels
    }
    if (std.mem.indexOf(u8, text, "```") != null) return null;
    for (code_hints) |hint| if (containsIgnoreCase(text, hint)) return null;
    if (trimmed.len < cfg.trivial_max_chars and std.mem.indexOfScalar(u8, trimmed, '\n') == null) {
        if (sentenceEnds(trimmed) > 1) return null;
        if (hasWorkVerb(trimmed)) return null;
        return "short";
    }
    return null;
}

/// Labels double as matcher needles: "pattern:<p>" — the matcher compares after the colon.
const pattern_list = [_][]const u8{ "pattern:hello", "pattern:hi", "pattern:hey", "pattern:status", "pattern:ack", "pattern:acknowledge", "pattern:thanks", "pattern:thank you", "pattern:ok", "pattern:okay", "pattern:yes", "pattern:no", "pattern:ping" };
const code_hints = [_][]const u8{ "def ", "class ", "function ", "import ", "#include", "fn main", "select ", "lambda " };
const work_verbs = [_][]const u8{ "explain", "derive", "design", "write", "refactor", "implement", "benchmark", "compare", "analyze", "analyse", "prove", "calculate", "summarize", "summarise", "review", "architect", "optimize", "optimise", "debug" };

fn eqlIgnoreCase(a: []const u8, b: []const u8) bool {
    return std.ascii.eqlIgnoreCase(a, b);
}

fn containsIgnoreCase(hay: []const u8, needle: []const u8) bool {
    var i: usize = 0;
    while (i + needle.len <= hay.len) : (i += 1) {
        if (std.ascii.eqlIgnoreCase(hay[i .. i + needle.len], needle)) return true;
    }
    return false;
}

/// "hello" / "hello." / "hello," / "hello!" / bare "hello!" (punctuation stripped).
/// `label` is "pattern:<needle>"; comparison uses the needle after the colon.
fn matchesPattern(text: []const u8, label: []const u8) bool {
    const colon = std.mem.indexOfScalar(u8, label, ':').?;
    const pat = label[colon + 1 ..];
    if (eqlIgnoreCase(text, pat)) return true;
    if (text.len > pat.len and std.ascii.eqlIgnoreCase(text[0..pat.len], pat)) {
        switch (text[pat.len]) {
            ' ', ',', '!', '.' => return true,
            else => {},
        }
    }
    const stripped = std.mem.trimEnd(u8, text, "!.?");
    return eqlIgnoreCase(stripped, pat);
}

fn isWordChar(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_';
}

fn hasWorkVerb(text: []const u8) bool {
    for (work_verbs) |verb| {
        var i: usize = 0;
        while (i + verb.len <= text.len) : (i += 1) {
            const before_ok = i == 0 or !isWordChar(text[i - 1]);
            const after_ok = i + verb.len == text.len or !isWordChar(text[i + verb.len]);
            if (before_ok and after_ok and std.ascii.eqlIgnoreCase(text[i .. i + verb.len], verb)) return true;
        }
    }
    return false;
}

/// Count of [.!?] followed by whitespace or end (mirrors r"[.!?](\s|$)").
fn sentenceEnds(text: []const u8) usize {
    var n: usize = 0;
    for (text, 0..) |c, i| {
        if (c == '.' or c == '!' or c == '?') {
            if (i + 1 == text.len or text[i + 1] == ' ' or text[i + 1] == '\t' or text[i + 1] == '\n') n += 1;
        }
    }
    return n;
}

pub const Sticky = struct {
    const Entry = struct { key: u64, lane: usize, lru_prev: ?usize, lru_next: ?usize, used: bool };

    slots: []Entry,
    head: ?usize = null, // most recently used
    tail: ?usize = null, // least recently used
    count: usize = 0,

    pub fn init(alloc: std.mem.Allocator, capacity: usize) !Sticky {
        const slots = try alloc.alloc(Entry, capacity);
        for (slots) |*e| e.* = .{ .key = 0, .lane = 0, .lru_prev = null, .lru_next = null, .used = false };
        return .{ .slots = slots, .head = null, .tail = null, .count = 0 };
    }

    pub fn deinit(self: *Sticky, alloc: std.mem.Allocator) void {
        alloc.free(self.slots);
        self.* = .{ .slots = &.{}, .count = 0 };
    }

    /// O(capacity) scan for the key; n is a conversation count (thousands), not a
    /// per-token path, and the whole decide() still lands in microseconds.
    pub fn get(self: *Sticky, key: u64) ?usize {
        for (self.slots, 0..) |e, i| {
            if (e.used and e.key == key) {
                self.touch(i);
                return e.lane;
            }
        }
        return null;
    }

    pub fn put(self: *Sticky, key: u64, lane: usize) void {
        for (self.slots, 0..) |e, i| {
            if (e.used and e.key == key) {
                self.slots[i].lane = lane;
                self.touch(i);
                return;
            }
        }
        var victim: ?usize = null;
        if (self.count < self.slots.len) { // room left: take any free slot, evict nothing
            for (self.slots, 0..) |e, i| {
                if (!e.used) {
                    victim = i;
                    break;
                }
            }
        }
        if (victim == null) victim = self.tail; // full: evict least recently used
        const vi = victim orelse return; // capacity 0 => sticky disabled
        const v = &self.slots[vi];
        if (v.used) {
            self.count -= 1;
            self.unlink(vi);
        }
        v.* = .{ .key = key, .lane = lane, .lru_prev = null, .lru_next = self.head, .used = true };
        if (self.head) |h| self.slots[h].lru_prev = vi;
        self.head = vi;
        if (self.tail == null) self.tail = vi;
        self.count += 1;
    }

    fn touch(self: *Sticky, i: usize) void {
        if (self.head == i) return;
        self.unlink(i);
        const e = &self.slots[i];
        e.lru_prev = null;
        e.lru_next = self.head;
        if (self.head) |h| self.slots[h].lru_prev = i;
        self.head = i;
        if (self.tail == null) self.tail = i;
    }

    fn unlink(self: *Sticky, i: usize) void {
        const e = self.slots[i];
        if (e.lru_prev) |p| self.slots[p].lru_next = e.lru_next else if (self.head == i) self.head = e.lru_next;
        if (e.lru_next) |n| self.slots[n].lru_prev = e.lru_prev else if (self.tail == i) self.tail = e.lru_prev;
        self.slots[i].lru_prev = null;
        self.slots[i].lru_next = null;
    }
};

pub const Decision = struct {
    lane: usize, // index into cfg.lanes
    reason: []const u8, // static string
};

pub const Router = struct {
    cfg: Config,
    sticky: Sticky,

    /// `conv_key` is the caller-computed conversation identity (client header or a hash
    /// of the first system+user turn — see conv_key() in the Python spec; it must be
    /// STABLE across turns or stickiness silently dies, unit-tested there).
    pub fn decide(self: *Router, model: ?[]const u8, text: []const u8, conv_key: u64) RouteError!Decision {
        if (model) |m| {
            if (m.len > 0) {
                for (self.cfg.lanes, 0..) |lane, i| {
                    if (lane.enabled and lane.matches(m)) return .{ .lane = i, .reason = "explicit" };
                }
                return RouteError.UnknownModel;
            }
        }
        if (self.cfg.sticky_enabled) {
            if (self.sticky.get(conv_key)) |lane| {
                if (self.cfg.lanes[lane].enabled) return .{ .lane = lane, .reason = "sticky" };
            }
        }
        var chosen: usize = undefined;
        const reason: []const u8 = if (trivialReason(text, self.cfg)) |r| blk: {
            chosen = self.trivialLane() orelse self.defaultLane() orelse 0;
            break :blk r;
        } else blk: {
            chosen = self.defaultLane() orelse 0;
            break :blk "default";
        };
        if (self.cfg.sticky_enabled) self.sticky.put(conv_key, chosen);
        return .{ .lane = chosen, .reason = reason };
    }

    fn trivialLane(self: Router) ?usize {
        for (self.cfg.lanes, 0..) |lane, i| {
            if (lane.enabled and lane.role == .trivial) return i;
        }
        return null;
    }

    fn defaultLane(self: Router) ?usize {
        for (self.cfg.lanes, 0..) |lane, i| {
            if (lane.enabled and lane.role == .default_lane) return i;
        }
        return null;
    }
};

// ─── host tests (mirror test_mom_router.py decision table) ──────────────────

fn twoLanes() []const Lane {
    return &.{
        .{ .id = "small", .role = .trivial, .aliases = &.{"tiny"} },
        .{ .id = "big", .role = .default_lane },
    };
}

test "router: greetings hit trivial patterns" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var r = Router{ .cfg = .{ .lanes = twoLanes() }, .sticky = try Sticky.init(arena.allocator(), 8) };
    const cases = [_][]const u8{ "hello", "Hello!", "hi there", "hey, how's it going", "status?", "thanks!", "ok", "no" };
    for (cases, 0..) |text, i| {
        const d = try r.decide(null, text, 1000 + i);
        try std.testing.expectEqual(@as(usize, 0), d.lane);
        try std.testing.expect(d.reason.len > 0 and !std.mem.eql(u8, d.reason, "default"));
    }
}

test "router: short everyday line is trivial" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var r = Router{ .cfg = .{ .lanes = twoLanes() }, .sticky = try Sticky.init(arena.allocator(), 8) };
    const d = try r.decide(null, "what's the capital of France", 1);
    try std.testing.expectEqualStrings("short", d.reason);
    try std.testing.expectEqual(@as(usize, 0), d.lane);
}

test "router: work verbs, multi-sentence, and code bail to default" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var r = Router{ .cfg = .{ .lanes = twoLanes() }, .sticky = try Sticky.init(arena.allocator(), 8) };
    const hard = [_][]const u8{
        "Explain the CAP theorem in one paragraph",
        "What is 2+2? And why?",
        "def solve(): pass",
        "please DEBUG this",
    };
    for (hard, 0..) |text, i| {
        const d = try r.decide(null, text, 42 + i);
        try std.testing.expectEqual(@as(usize, 1), d.lane);
        try std.testing.expectEqualStrings("default", d.reason);
    }
}

test "router: explicit id wins; alias maps; unknown id is refused" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var r = Router{ .cfg = .{ .lanes = twoLanes() }, .sticky = try Sticky.init(arena.allocator(), 8) };
    const d = try r.decide("small", "derive the Euler identity step by step. Then verify it.", 1);
    try std.testing.expectEqual(@as(usize, 0), d.lane);
    try std.testing.expectEqualStrings("explicit", d.reason);
    const a = try r.decide("tiny", "hello", 2);
    try std.testing.expectEqual(@as(usize, 0), a.lane);
    try std.testing.expectEqualStrings("explicit", a.reason);
    try std.testing.expectError(RouteError.UnknownModel, r.decide("Tiny", "hello", 2)); // case-sensitive, per spec
    try std.testing.expectError(RouteError.UnknownModel, r.decide("gpt-9", "hello", 3));
}

test "router: sticky keeps a conversation put across turns" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var r = Router{ .cfg = .{ .lanes = twoLanes() }, .sticky = try Sticky.init(arena.allocator(), 8) };
    const first = try r.decide(null, "derive the Euler identity step by step. Then verify it.", 7);
    try std.testing.expectEqualStrings("default", first.reason);
    const later = try r.decide(null, "hello", 7);
    try std.testing.expectEqual(@as(usize, 1), later.lane);
    try std.testing.expectEqualStrings("sticky", later.reason);
}

test "router: sticky disabled lets every message re-route" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var r = Router{ .cfg = .{ .lanes = twoLanes(), .sticky_enabled = false }, .sticky = try Sticky.init(arena.allocator(), 8) };
    _ = try r.decide(null, "derive the Euler identity step by step. Then verify it.", 7);
    const later = try r.decide(null, "hello", 7);
    try std.testing.expectEqual(@as(usize, 0), later.lane);
}

test "router: sticky LRU evicts the least recently used conversation" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var r = Router{ .cfg = .{ .lanes = twoLanes() }, .sticky = try Sticky.init(arena.allocator(), 2) };
    _ = try r.decide(null, "derive the Euler identity. Verify it.", 1); // fills slot for k1 -> big
    _ = try r.decide(null, "derive the Riemann hypothesis. Verify it.", 2); // fills slot for k2 -> big
    const touch = try r.decide(null, "hello", 1); // k1 is MRU now
    try std.testing.expectEqualStrings("sticky", touch.reason);
    _ = try r.decide(null, "design a compiler. Test it.", 3); // full -> evicts LRU = k2
    const kept = try r.decide(null, "hello", 1); // k1 survived (was touched) — check before k2's put evicts it
    try std.testing.expectEqualStrings("sticky", kept.reason);
    try std.testing.expectEqual(@as(usize, 1), kept.lane);
    const evicted = try r.decide(null, "hello", 2); // k2 gone -> rules again -> trivial
    try std.testing.expectEqual(@as(usize, 0), evicted.lane);
    try std.testing.expectEqualStrings("pattern:hello", evicted.reason);
}
