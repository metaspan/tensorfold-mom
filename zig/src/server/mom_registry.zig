//! Mixture-of-Models per-lane registry (Track B, B2): the opened engines a request
//! may land on, plus the routing decision made once per request in openai.plan().
//!
//! The router core (`mom_router.zig`) is a pure 1:1 port of the Track A policy spec;
//! this module adapts it to the server: JSON body extraction (`lastUserText`,
//! `convKey`), a mutex so HTTP threads share the sticky table safely, and the
//! decision log. Requests reach the chosen engine via `chat.Input.mom_lane` at the
//! chat submit site; the child engine revalidates against its own `info()` there,
//! so a lane whose window is too small refuses with a clean InvalidRequest rather
//! than corrupting state.
//!
//! v1 lane constraints (documented in docs/07-b2-integration.md):
//! - lanes must share a tokenizer: mom_main refuses to start when the checkpoints'
//!   tokenizer.json digests differ (generation decodes with the server's `srv.text`);
//! - requests carrying tools/functions never take the trivial lane (the caller
//!   passes an empty text so no trivial rule can fire);
//! - conversation identity is content-anchored (first system + first user turn,
//!   FNV-1a); client conversation headers land with the HTTP-layer pass.
const std = @import("std");
const json = @import("json");
const api = @import("engine_api");
const mom_router = @import("mom_router.zig");

pub const Lane = struct {
    id: []const u8,
    engine: api.Engine,
    info: api.Info,
};

pub const Decision = struct { lane: usize, reason: []const u8 };

pub const Registry = struct {
    lanes: []const Lane,
    router: mom_router.Router,
    log_file: ?std.Io.File = null,
    io: std.Io,
    mutex: std.Io.Mutex = .init,
    /// Observability (brief component 7): decisions per lane since start; /v1/models
    /// reports them per advertised id. Capped by the router's lane limit.
    decision_counts: [16]u32 = @splat(0),
    decision_total: u64 = 0,

    pub fn laneRoleName(self: *const Registry, li: usize) []const u8 {
        const role: mom_router.Role = if (li < self.router.cfg.lanes.len) self.router.cfg.lanes[li].role else .other;
        return switch (role) { .trivial => "trivial", .default_lane => "default", .other => "other" };
    }

    pub fn decisionCount(self: *Registry, lane: usize) u32 {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        return if (lane < self.decision_counts.len) self.decision_counts[lane] else 0;
    }

    /// The single routing call per request: resolve the explicit field (unknown id
    /// is a hard refusal), else sticky, else trivial rules, else default lane.
    /// `model` is the raw request `model` string (null/empty = not asked).
    pub fn route(self: *Registry, a: std.mem.Allocator, model: ?[]const u8, body: json.Value) RouteErr!Decision {
        const has_tools = body.get("tools") != null or body.get("functions") != null;
        const text: []const u8 = if (has_tools) "" else try lastUserText(a, body);
        const ck = convKey(body);
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        const d = try self.router.decide(model, text, ck);
        if (d.lane < self.decision_counts.len) self.decision_counts[d.lane] += 1;
        self.decision_total += 1;
        if (self.log_file) |f| self.logDecision(f, model, d);
        return .{ .lane = d.lane, .reason = d.reason };
    }

    fn nowMs(self: *Registry) i64 {
        return @divTrunc(@as(i64, @intCast(@import("clock.zig").nowNs(self.io))), 1_000_000);
    }

    fn logDecision(self: *Registry, f: std.Io.File, model: ?[]const u8, d: mom_router.Decision) void {
        var buf: [1024]u8 = undefined;
        const line = std.fmt.bufPrint(&buf, "{{\"t\":{d},\"asked\":\"{s}\",\"routed\":\"{s}\",\"reason\":\"{s}\"}}\n", .{
            self.nowMs(),
            model orelse "",
            self.lanes[d.lane].id,
            d.reason,
        }) catch return;
        f.writeStreamingAll(self.io, line) catch {};
    }
};

pub const RouteErr = error{UnknownModel} || std.mem.Allocator.Error;

/// The last user message's text: a string content, or the text parts of a
/// multimodal content joined with single spaces (mirrors `_last_user_text`).
/// Empty when the body has no user message.
pub fn lastUserText(a: std.mem.Allocator, body: json.Value) ![]const u8 {
    const msgs = body.get("messages") orelse return "";
    if (msgs != .array) return "";
    var i = msgs.array.len;
    while (i > 0) {
        i -= 1;
        const m = msgs.array[i];
        if (m != .object) continue;
        const role = m.get("role");
        if (role == null or role.?.string.len != 4 or !std.ascii.eqlIgnoreCase(role.?.string, "user")) {} else {
            const c = m.get("content") orelse continue;
            if (c == .string) return c.string;
            if (c == .array) {
                var out = std.ArrayList(u8).empty;
                for (c.array) |part| {
                    if (part != .object) continue;
                    if (part.get("text")) |t| if (t == .string) {
                        if (out.items.len > 0) try out.append(a, ' ');
                        try out.appendSlice(a, t.string);
                    };
                }
                return out.items;
            }
        }
    }
    return "";
}

/// Stable conversation identity: FNV-1a over the FIRST system and FIRST user
/// message contents (2000 chars each, as the Python spec). Stable as turns grow —
/// hashing the whole history would re-key every incremental message and silently
/// kill stickiness (K3 probe guards this end to end).
pub fn convKey(body: json.Value) u64 {
    var h: u64 = 0xcbf29ce484222325;
    var seen_system = false;
    var seen_user = false;
    const msgs = body.get("messages") orelse return 0;
    if (msgs != .array) return 0;
    for (msgs.array) |m| {
        if (m != .object) continue;
        const role = (m.get("role") orelse continue);
        if (role != .string) continue;
        var is_sys = false;
        var is_user = false;
        if (std.ascii.eqlIgnoreCase(role.string, "system")) is_sys = true else if (std.ascii.eqlIgnoreCase(role.string, "user")) is_user = true;
        if (is_sys) {
            if (seen_system) continue;
            seen_system = true;
            fnvUpdate(&h, "system");
        } else if (is_user) {
            if (seen_user) continue;
            seen_user = true;
            fnvUpdate(&h, "user");
        } else continue;
        if (m.get("content")) |c| {
            if (c == .string) {
                fnvUpdate(&h, c.string[0..@min(c.string.len, 2000)]);
            } else if (c == .array) {
                for (c.array) |part| {
                    if (part != .object) continue;
                    if (part.get("text")) |t| if (t == .string) fnvUpdate(&h, t.string[0..@min(t.string.len, 2000)]);
                }
            }
        }
    }
    return h;
}

fn fnvUpdate(h: *u64, bytes: []const u8) void {
    for (bytes) |b| {
        h.* ^= b;
        h.* *%= 0x100000001b3;
    }
}

// ─── host tests ───────────────────────────────────────────────────────────────


test "registry: lastUserText takes the last user string, multimodal parts joined" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const body1 = (try json.parse(a, "{\"messages\":[{\"role\":\"user\",\"content\":\"first\"},{\"role\":\"assistant\",\"content\":\"a\"},{\"role\":\"user\",\"content\":\"second\"}]}")).ok;
    try std.testing.expectEqualStrings("second", try lastUserText(a, body1));
    const body2 = (try json.parse(a, "{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"a\"},{\"type\":\"image\"},{\"type\":\"text\",\"text\":\"b\"}]}]}")).ok;
    try std.testing.expectEqualStrings("a b", try lastUserText(a, body2));
    const body3 = (try json.parse(a, "{\"messages\":[{\"role\":\"system\",\"content\":\"s\"}]}")).ok;
    try std.testing.expectEqualStrings("", try lastUserText(a, body3));
}

test "registry: route counts decisions per lane" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const lane_defs = [_]mom_router.Lane{
        .{ .id = "small", .role = .trivial },
        .{ .id = "big", .role = .default_lane },
    };
    var reg: Registry = .{
        .lanes = &.{},
        .router = .{
            .cfg = .{ .lanes = &lane_defs, .sticky_enabled = false },
            .sticky = try mom_router.Sticky.init(a, 8),
        },
        .io = std.testing.io,
    };
    const trivial = (try json.parse(a, "{\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}")).ok;
    const heavy = (try json.parse(a, "{\"messages\":[{\"role\":\"user\",\"content\":\"Please implement a full test suite and then review the distributed cache design carefully.\"}]}")).ok;
    _ = try reg.route(a, null, trivial);
    _ = try reg.route(a, null, trivial);
    _ = try reg.route(a, null, heavy);
    try std.testing.expectEqual(@as(u32, 2), reg.decisionCount(0));
    try std.testing.expectEqual(@as(u32, 1), reg.decisionCount(1));
    try std.testing.expectEqual(@as(u64, 3), reg.decision_total);
    try std.testing.expectEqualStrings("trivial", reg.laneRoleName(0));
    try std.testing.expectEqualStrings("default", reg.laneRoleName(1));
}

test "registry: convKey is stable as the conversation grows" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const turn1 = (try json.parse(a, "{\"messages\":[{\"role\":\"system\",\"content\":\"be terse\"},{\"role\":\"user\",\"content\":\"why is the sky blue\"}]}")).ok;
    const k1 = convKey(turn1);
    const turn5 = (try json.parse(a, "{\"messages\":[{\"role\":\"system\",\"content\":\"be terse\"},{\"role\":\"user\",\"content\":\"why is the sky blue\"},{\"role\":\"assistant\",\"content\":\"scattering\"},{\"role\":\"user\",\"content\":\"and at sunset\"}]}")).ok;
    try std.testing.expectEqual(k1, convKey(turn5));
    const other = (try json.parse(a, "{\"messages\":[{\"role\":\"system\",\"content\":\"be terse\"},{\"role\":\"user\",\"content\":\"different question\"}]}")).ok;
    try std.testing.expect(k1 != convKey(other));
}
