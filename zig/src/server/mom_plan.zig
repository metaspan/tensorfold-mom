//! Mixture-of-Models load planning: the pre-open budget check that decides which models may share one
//! process, and the quota each gets. Pure arithmetic (this file imports nothing but std) so it is host
//! tested without a GPU; `main`'s mom path feeds measured numbers in and refuses startup on `refused`.
//!
//! The rule: the sum of every model's need, plus ONE process reserve (this is the single-process win —
//! separate processes each pay their own), must fit the free device memory the first open measured.
//! A model's need is its weights, its parallel streams' KV at the chosen window, and a fixed slack for
//! activations, graphs and the drafter's second network.

const std = @import("std");

pub const gib: f64 = 1 << 30;

/// One model as the plan sees it, before anything is opened. Sizes in bytes.
pub const ModelPlan = struct {
    id: []const u8,
    weights: u64, // family_memory.weights' estimate for this checkpoint and rank split
    streams: u32, // --parallel
    stream_bytes: u64, // served.stream_bytes: the KV and rows one admitted stream holds
    drafts: bool, // a drafter network rides along (its weights counted inside `weights` by the caller)
    pinned: bool, // never parked (the trivial lane); parkable models may later be woken under pressure
};

/// What the machine offers at plan time (measured by the first Context.memInfo + the memory budget).
pub const Capacity = struct {
    free: u64, // cuMemGetInfo's free at process start
    reserve: u64, // the process reserve the CUDA budget keeps (once, not per model)
    slack_per_model: u64 = 64 << 20, // activations, graph pools, allocator rounding
};

pub const Decision = union(enum) {
    pass: []const Quota,
    refused: []const u8, // the full startup refusal line, naming the dominating term
};

pub const Quota = struct {
    id: []const u8,
    need: u64, // what the plan charged it
    pinned: bool,
};

/// A model's need: weights + streams x stream bytes + slack. Saturating adds so a nonsense plan
/// reads as too big rather than wrapping into a pass.
pub fn need(m: ModelPlan, slack_per_model: u64) u64 {
    const kv = saturatingMul(m.streams, m.stream_bytes);
    return saturatingAdd(saturatingAdd(m.weights, kv), slack_per_model);
}

pub fn saturatingMul(a: u32, b: u64) u64 {
    return std.math.mul(u64, a, b) catch std.math.maxInt(u64);
}

pub fn saturatingAdd(a: u64, b: u64) u64 {
    return std.math.add(u64, a, b) catch std.math.maxInt(u64);
}

/// The room one model may take: its plan need (its admission hard cap, used by the loader's pools).
pub fn quota(m: ModelPlan, slack_per_model: u64) u64 {
    return need(m, slack_per_model);
}

/// Decide whether `models` co-reside under `cap`. On pass, one Quota per model in order; on refuse,
/// a message naming the model and term that dominates the overage, so the operator knows what to cut.
pub fn plan(a: std.mem.Allocator, models: []const ModelPlan, cap: Capacity) !Decision {
    if (models.len == 0) return .{ .refused = try a.dupe(u8, "mom: no models to plan") };
    if (models.len > 8) return .{ .refused = try a.dupe(u8, "mom: at most 8 models share one process") };
    var total: u64 = cap.reserve; // once for the process: the reserve, paid one time
    var biggest: usize = 0;
    var biggest_need: u64 = 0;
    const quotas = try a.alloc(Quota, models.len);
    errdefer a.free(quotas);
    for (models, 0..) |m, i| {
        const n = need(m, cap.slack_per_model);
        quotas[i] = .{ .id = m.id, .need = n, .pinned = m.pinned };
        total = saturatingAdd(total, n);
        if (n > biggest_need) {
            biggest_need = n;
            biggest = i;
        }
    }
    if (total > cap.free) {
        a.free(quotas); // refused decisions own no quotas
        const m = models[biggest];
        const kv = saturatingMul(m.streams, m.stream_bytes);
        const dominating = if (m.weights >= kv) "weights" else "streams x KV";
        return .{ .refused = try std.fmt.allocPrint(
            a,
            "mom: {s} ({d:.1} GiB {s}; {d} streams x {d:.1} GiB; plus {d} other model(s) and a {d:.1} GiB process reserve) needs {d:.1} GiB of the {d:.1} GiB free; lower --parallel or --context, serve fewer models, or free device memory",
            .{ m.id, @as(f64, @floatFromInt(m.weights)) / gib, dominating, m.streams, @as(f64, @floatFromInt(m.stream_bytes)) / gib, models.len - 1, @as(f64, @floatFromInt(cap.reserve)) / gib, @as(f64, @floatFromInt(total)) / gib, @as(f64, @floatFromInt(cap.free)) / gib },
        ) };
    }
    return .{ .pass = quotas };
}

/// One startup line: `mom plan: A weights=1.6 streams=8x0.2+slack=.. quota=.. pinned; B ...; free=.. reserve=..(once)`.
/// The result is owned by `a` (caller-freed as a unit; main's arena is the intended owner).
pub fn planLine(a: std.mem.Allocator, models: []const ModelPlan, cap: Capacity) ![]const u8 {
    var out: std.Io.Writer.Allocating = .init(a); // what it prints is what it allocates
    errdefer out.deinit();
    const w = &out.writer;
    try w.writeAll("mom plan:");
    for (models) |m| {
        try w.print(
            " {s} weights={d:.1}GiB {d}streams x {d:.1}GiB quota={d:.1}GiB{s}",
            .{ m.id, @as(f64, @floatFromInt(m.weights)) / gib, m.streams, @as(f64, @floatFromInt(m.stream_bytes)) / gib,
               @as(f64, @floatFromInt(quota(m, cap.slack_per_model))) / gib, if (m.pinned) " pinned" else "" },
        );
    }
    try w.print(
        "; free={d:.1}GiB reserve={d:.1}GiB (once, not per model)",
        .{ @as(f64, @floatFromInt(cap.free)) / gib, @as(f64, @floatFromInt(cap.reserve)) / gib },
    );
    return out.written();
}

// -- host tests -------------------------------------------------------------------------------------

test "plan passes a pair that fits and quotes each model its need" {
    const models = [_]ModelPlan{
        .{ .id = "qwen3.5-2b", .weights = 2 * gib, .streams = 8, .stream_bytes = 170 << 20, .drafts = false, .pinned = true },
        .{ .id = "qwen3.6-35b", .weights = 21 * gib, .streams = 5, .stream_bytes = 920 << 20, .drafts = true, .pinned = false },
    };
    const cap = Capacity{ .free = 100 * gib, .reserve = 12 * gib };
    const d = try plan(std.testing.allocator, &models, cap);
    defer switch (d) { .pass => |q| std.testing.allocator.free(q), .refused => |w| std.testing.allocator.free(w) };
    const q = switch (d) { .pass => |x| x, .refused => return error.UnexpectedRefusal };
    try std.testing.expectEqual(@as(usize, 2), q.len);
    try std.testing.expectEqualStrings("qwen3.5-2b", q[0].id);
    try std.testing.expect(q[0].pinned);
    try std.testing.expectEqual(need(models[0], cap.slack_per_model), q[0].need);
    // the small model's quota covers weights + 8 x 170 MiB + slack
    try std.testing.expect(q[0].need > 2 * gib + 8 * (170 << 20));
}

test "plan refuses an over-budget pair naming the dominating model and term" {
    const models = [_]ModelPlan{
        .{ .id = "tiny", .weights = 1 * gib, .streams = 1, .stream_bytes = 10 << 20, .drafts = false, .pinned = true },
        .{ .id = "whale", .weights = 10 * gib, .streams = 8, .stream_bytes = 16 * gib, .drafts = true, .pinned = false },
    };
    const cap = Capacity{ .free = 100 * gib, .reserve = 12 * gib };
    const d = try plan(std.testing.allocator, &models, cap);
    defer switch (d) { .pass => |q| std.testing.allocator.free(q), .refused => |w| std.testing.allocator.free(w) };
    const why = switch (d) { .refused => |w| w, .pass => return error.ExpectedRefusal };
    try std.testing.expect(std.mem.indexOf(u8, why, "whale") != null);
    try std.testing.expect(std.mem.indexOf(u8, why, "streams x KV") != null); // 8 x 16 GiB of KV dominates its 10 GiB of weights
    try std.testing.expect(std.mem.indexOf(u8, why, "process reserve") != null);
}

test "plan charges the process reserve once, not per model: the exact-fit boundary" {
    const slack: u64 = 1 << 20;
    // each model's need is exactly half the room left after the ONE reserve: the pair fits to the byte
    const per = (100 * gib - 12 * gib) / 2 - 1 * gib - slack;
    const models = [_]ModelPlan{
        .{ .id = "a", .weights = 1 * gib, .streams = 1, .stream_bytes = per, .drafts = false, .pinned = false },
        .{ .id = "b", .weights = 1 * gib, .streams = 1, .stream_bytes = per, .drafts = false, .pinned = false },
    };
    const d = try plan(std.testing.allocator, &models, .{ .free = 100 * gib, .reserve = 12 * gib, .slack_per_model = slack });
    defer switch (d) { .pass => |q| std.testing.allocator.free(q), .refused => |w| std.testing.allocator.free(w) };
    _ = switch (d) { .pass => |q| q, .refused => return error.ExpectedPassAtExactFit };
    // three of them refuse: the third adds need with no reserve saving left to matter
    const three = [_]ModelPlan{ models[0], models[1], models[0] };
    const d3 = try plan(std.testing.allocator, &three, .{ .free = 100 * gib, .reserve = 12 * gib, .slack_per_model = slack });
    defer switch (d3) { .pass => |q| std.testing.allocator.free(q), .refused => |w| std.testing.allocator.free(w) };
    _ = switch (d3) { .refused => |w| w, .pass => return error.ExpectedRefusalForThree };
}

test "plan saturates instead of wrapping: absurd sizes refuse" {
    const models = [_]ModelPlan{
        .{ .id = "absurd", .weights = std.math.maxInt(u64) - 1, .streams = std.math.maxInt(u32), .stream_bytes = std.math.maxInt(u32), .drafts = false, .pinned = false },
    };
    const d = try plan(std.testing.allocator, &models, .{ .free = 100 * gib, .reserve = 12 * gib });
    defer switch (d) { .pass => |q| std.testing.allocator.free(q), .refused => |w| std.testing.allocator.free(w) };
    _ = switch (d) { .refused => |w| w, .pass => return error.ExpectedRefusal };
}

test "planLine names every model, the single reserve, and the free room" {
    const models = [_]ModelPlan{
        .{ .id = "small", .weights = gib, .streams = 4, .stream_bytes = 100 << 20, .drafts = false, .pinned = true },
        .{ .id = "big", .weights = 20 * gib, .streams = 5, .stream_bytes = gib, .drafts = true, .pinned = false },
    };
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit(); // the line is the arena's, freed as a unit like the startup path frees its own
    const line = try planLine(arena.allocator(), &models, .{ .free = 100 * gib, .reserve = 12 * gib });
    for ([_][]const u8{ "small", "big", "pinned", "once, not per model", "free=100.0GiB" }) |want|
        try std.testing.expect(std.mem.indexOf(u8, line, want) != null);
}
