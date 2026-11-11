//! `tensorfold-native mom-serve`: one process, one endpoint, several checkpoints (Mixture-of-Models, B1).
//!
//! v1 (B1): every model is opened in-process, in order, by the unmodified backend open path; the
//! server serves on the DEFAULT model and advertises every registered id (extra ids arrive as
//! aliases until the B2 composite engine routes them). Shared context teardown is counted by
//! cuda.context_owner, so any model's deinit is safe while a sibling runs. Refused here, before any
//! work: --tp, rank>0, --slide, --recall, --learn-conversations (per-model bookkeeping lands with
//! B3-B5). The plan line reports each checkpoint's weight estimate, streams and quota, and names the
//! process reserve as paid ONCE — the single-process win the two-process deployment cannot claim.
const std = @import("std");
const cli = @import("cli.zig");
const log = @import("log.zig");
const hub = @import("hub.zig");
const engines = @import("engines.zig");
const serve = @import("serve.zig");
const hf_text = @import("hf_text.zig");
const startup = @import("startup.zig");
const api = @import("engine_api");
const mom_plan = @import("mom_plan.zig");
const mom_router = @import("mom_router.zig");
const mom_registry = @import("mom_registry.zig");
const mom_sleep = @import("mom_sleep.zig");
const tool_id_key = @import("tool_id_key.zig");

const max_models = 8;

/// A resolved checkpoint: its folder, family, and weight bytes (file scope so the
/// sleep/wake Reopen closure can carry the open recipe; mom_main.run fills it).
const Held = struct {
    spec: Spec,
    dir: []const u8,
    model_type: []const u8,
    weights: u64,
};

/// Stable state behind sleep/wake (docs/08): everything needed to reopen any lane.
/// `opened` points at run()'s slot array (stable: serve.run blocks in the same frame).
const SleepCtx = struct {
    held: []const Held,
    lanes: []mom_registry.Lane,
    opened: []?api.Opened,
    args: cli.Args,
    a: std.mem.Allocator,
    gpa: std.mem.Allocator,
};

const OpenError = error{ OpenFailed, OutOfMemory };

fn reopenLane(p: *anyopaque, a: std.mem.Allocator, io: std.Io, lane: usize) anyerror!void {
    const t: *SleepCtx = @ptrCast(@alignCast(p));
    var problem: []const u8 = "";
    const h = t.held[lane];
    const entry0: i64 = @intCast(@import("clock.zig").nowNs(io));
    const got = (engines.open(a, t.gpa, io, h.dir, h.model_type, t.args, &problem) catch |e| {
        log.line("mom: wake {s} open failed: {s}", .{ h.spec.name, @errorName(e) });
        return e;
    }) orelse {
        log.line("mom: wake {s} refused: {s}", .{ h.spec.name, if (problem.len > 0) problem else "the backend refused it" });
        return error.OpenFailed;
    };
    const slot = &t.opened[lane]; // the timer closed it; this is the belt to a stale slot
    if (slot.*) |o| {
        o.close(o.ctx);
        slot.* = null;
    }
    slot.* = got;
    t.lanes[lane].engine = got.engine;
    t.lanes[lane].info = got.engine.info();
    const ms = @divTrunc(@as(i64, @intCast(@import("clock.zig").nowNs(io))) - entry0, std.time.ns_per_ms);
    log.line("mom: woke {s} in {d} ms", .{ h.spec.name, ms });
}

/// One shared timer thread: parks idle lanes (non-trivial only: their idle_ns is 0).
const Sleeper = struct {
    reg: *mom_registry.Registry,
    ctx: *SleepCtx,
    io: std.Io,

    fn run(self: *Sleeper) void {
        while (true) {
            std.Io.sleep(self.io, .fromMilliseconds(1000), .awake) catch {};
            for (self.reg.lifecycles, 0..) |*lc, i| {
                if (lc.idle_ns == 0) continue;
                const now: i64 = @intCast(@import("clock.zig").nowNs(self.io));
                if (!lc.shouldSleep(now)) continue;
                if (!lc.claimUnload()) continue; // a lease (or a state move) saved it
                const oslot = &self.ctx.opened[i];
                if (oslot.*) |o| {
                    o.close(o.ctx);
                    oslot.* = null;
                }
                self.ctx.lanes[i].engine = null; // readers meet it resident (route wakes); submit's refuse is the belt
                lc.finishUnload();
                log.line("mom: {s} asleep (idle)", .{self.ctx.held[i].spec.name});
            }
        }
    }
};

/// One `--model DIR --name NAME [--lane NAME] [--pin]` group, in command order (the first is the default).
const Spec = struct {
    model: []const u8,
    name: []const u8,
    lane: []const u8 = "default",
    pin: bool = false,
};

/// The bytes the reply path actually depends on: the BPE inverse mapping is vocab.json
/// (or merges.txt, then tokenizer.json as last resort), and lanes must share it exactly,
/// because generation decodes every lane's tokens with the server-wide tokenizer.
/// added_tokens/special-token sets may differ (documented caveat: a lane emitting a
/// special token that is plain text in the default's table prints its marker text);
/// engines stop on their own EOS, so runaway is bounded regardless.
fn tokenizerDigest(io: std.Io, dir: []const u8) ?[32]u8 {
    const cwd = std.Io.Dir.cwd();
    const preferred = [_][]const u8{ "vocab.json", "merges.txt", "tokenizer.json" };
    for (preferred) |name| {
        var joined: [4096]u8 = undefined;
        const path = std.fmt.bufPrint(&joined, "{s}/{s}", .{ dir, name }) catch return null;
        var f = cwd.openFile(io, path, .{}) catch continue;
        defer f.close(io);
        var hasher = std.crypto.hash.sha2.Sha256.init(.{});
        var buf: [65536]u8 = undefined;
        var seen: u64 = 0;
        while (true) {
            const n = f.readPositional(io, &.{&buf}, seen) catch break;
            if (n == 0) break;
            hasher.update(buf[0..n]);
            seen += n;
        }
        if (seen == 0) continue;
        return hasher.finalResult();
    }
    return null;
}

fn laneId(lanes: []const mom_router.Lane, role: mom_router.Role) []const u8 {
    for (lanes) |l| if (l.role == role) return l.id;
    return "-";
}

fn openDecisionLog(io: std.Io, path: []const u8) !std.Io.File {
    const cwd = std.Io.Dir.cwd();
    return cwd.createFile(io, path, .{}); // v1: one decisions file per run; rotation is a B4 nicety
}

/// The parsed model specs and the argv with every mom-only flag removed (cli.parse sees the rest).
pub const Parsed = struct {
    /// --lane-priority: the trivial lane's engine opens its CUDA stream at .highest
    /// priority so its kernels preempt the heavy lane's under contention (docs/03 B4 note).
    lane_priority: bool = false,
    specs: []Spec,
    rest: []const []const u8,
    mom_log: ?[]const u8 = null,
    no_sticky: bool = false,
    lane_idle_s: i64 = 0,
};

/// Extracts the repeated --model/--name/--lane/--pin groups. --model starts a group; --name is
/// required in it. The rest of argv passes through untouched for cli.parse, so every ordinary
/// serve flag keeps working exactly as it does today.
pub fn parseSpecs(a: std.mem.Allocator, argv: []const []const u8, problem: *[]const u8) !?Parsed {
    var specs: std.ArrayList(Spec) = .empty;
    var rest: std.ArrayList([]const u8) = .empty;
    var pending: ?Spec = null;
    const flush = struct {
        fn one(list: *std.ArrayList(Spec), alloc: std.mem.Allocator, p: ?Spec, why: *[]const u8) bool {
            const s = p orelse return true;
            if (s.name.len == 0) {
                why.* = "mom-serve: --model needs a --name in its group (the served id)";
                return false;
            }
            list.append(alloc, .{ .model = s.model, .name = s.name, .lane = s.lane, .pin = s.pin }) catch return false;
            return true;
        }
    }.one;
    var mom_log: ?[]const u8 = null;
    var no_sticky = false;
    var lane_priority = false;
    var lane_idle: i64 = 0;
    var i: usize = 0;
    while (i < argv.len) : (i += 1) {
        const arg = argv[i];
        if (std.mem.eql(u8, arg, "--lane-idle")) {
            if (i + 1 >= argv.len) {
                problem.* = "mom-serve: --lane-idle wants a seconds count";
                return null;
            }
            const secs = std.fmt.parseInt(i64, argv[i + 1], 10) catch {
                problem.* = "mom-serve: --lane-idle wants a non-negative integer";
                return null;
            };
            if (secs < 0) {
                problem.* = "mom-serve: --lane-idle wants a non-negative integer";
                return null;
            }
            lane_idle = secs;
            i += 1;
            continue;
        }
        if (std.mem.eql(u8, arg, "--lane-priority")) {
            lane_priority = true;
            continue;
        }
        if (std.mem.eql(u8, arg, "--no-sticky")) {
            no_sticky = true;
            continue;
        }
        if (std.mem.eql(u8, arg, "--mom-log")) {
            if (i + 1 >= argv.len) {
                problem.* = "mom-serve: --mom-log wants a file path";
                return null;
            }
            mom_log = argv[i + 1];
            i += 1;
            continue;
        }
        if (std.mem.eql(u8, arg, "--model")) {
            if (i + 1 >= argv.len) {
                problem.* = "mom-serve: --model wants a checkpoint folder or repo id";
                return null;
            }
            if (pending != null and !flush(&specs, a, pending, problem)) return null;
            pending = .{ .model = argv[i + 1], .name = "" };
            i += 1;
            continue;
        }
        if (std.mem.eql(u8, arg, "--name") and pending != null and pending.?.name.len == 0) {
            if (i + 1 >= argv.len) {
                problem.* = "mom-serve: --name wants the served id";
                return null;
            }
            pending.?.name = argv[i + 1];
            i += 1;
            continue;
        }
        if (std.mem.eql(u8, arg, "--lane")) {
            if (pending == null) {
                problem.* = "mom-serve: --lane follows its --model";
                return null;
            }
            if (i + 1 >= argv.len) {
                problem.* = "mom-serve: --lane wants trivial or default";
                return null;
            }
            pending.?.lane = argv[i + 1];
            i += 1;
            continue;
        }
        if (std.mem.eql(u8, arg, "--pin")) {
            if (pending == null) {
                problem.* = "mom-serve: --pin follows its --model";
                return null;
            }
            pending.?.pin = true;
            continue;
        }
        try rest.append(a, arg);
    }
    if (pending != null and !flush(&specs, a, pending, problem)) return null;
    if (specs.items.len == 0) {
        problem.* = "mom-serve: at least one --model DIR --name NAME group is required";
        return null;
    }
    if (specs.items.len > max_models) {
        problem.* = "mom-serve: at most 8 models share one process";
        return null;
    }
    // ids must be distinct: the router's decision table and /v1/models both key on them
    for (specs.items, 0..) |x, xi| for (specs.items[xi + 1..]) |y| {
        if (std.mem.eql(u8, x.name, y.name)) {
            problem.* = try std.fmt.allocPrint(a, "mom-serve: two models share the name {s}", .{x.name});
            return null;
        }
    };
    return .{ .specs = specs.items, .rest = rest.items, .mom_log = mom_log, .no_sticky = no_sticky, .lane_priority = lane_priority, .lane_idle_s = lane_idle };
}

/// A refusal for flags the mom path has no per-model story for yet (B3-B5): the message says which stage brings them.
pub fn momRefusal(args: cli.Args) ?[]const u8 {
    if (args.tp > 1) return "mom-serve: --tp stays per-model with the shared scheduler (B3); serve that model alone for now";
    if (args.rank > 0) return "mom-serve: a TP rank serves no HTTP; ranks follow the scheduler work (B3)";
    if (args.slide) return "mom-serve: --slide learns into one model's folder; per-model learners land with B5";
    if (args.recall) return "mom-serve: --recall reads one record into one engine's rounds; per-model recall lands with B3";
    if (args.learn_conversations) return "mom-serve: --learn-conversations keys conversations per model; deferred to B3";
    return null;
}

/// Weight bytes of a checkpoint folder: the tensors a load will page in (sizes on disk are the plan's estimate).
/// The lane whose engine should open its CUDA stream at .highest priority: the
/// explicit trivial spec, else (roles falling back to weights) the lightest model.
/// Null for one lane: nothing to preempt. Mirrors the registry's role assignment.
fn priorityIndex(specs: []const Spec, weights: []const u64) ?usize {
    if (specs.len < 2) return null;
    for (specs, 0..) |sp, i| if (std.mem.eql(u8, sp.lane, "trivial")) return i;
    var heavy: usize = 0;
    var light: usize = 0;
    for (weights, 0..) |w, i| {
        if (w > weights[heavy]) heavy = i;
        if (w < weights[light]) light = i;
    }
    return if (heavy != light) light else null;
}

// This std exposes getenv but not setenv; the priority env window needs libc directly.
extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;
fn clearTrivialPriority() void {
    _ = unsetenv("TF_MOM_TRIVIAL_PRIORITY");
}

fn weightsBytes(a: std.mem.Allocator, io: std.Io, dir: []const u8) !u64 {
    const exts = [_][]const u8{ ".safetensors", ".gguf", ".msgpack", ".bin" };
    _ = a;
    var total: u64 = 0;
    var d = std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true }) catch return 0;
    defer d.close(io);
    var it = d.iterate();
    while (try it.next(io)) |e| {
        if (e.kind != .file) continue;
        for (exts) |ext| if (std.mem.endsWith(u8, e.name, ext)) {
            const st = try d.statFile(io, e.name, .{});
            total += st.size;
        };
    }
    return total;
}

fn modelType(a: std.mem.Allocator, io: std.Io, dir: []const u8, problem: *[]const u8) ![]const u8 {
    const bytes = (try @import("checkpoint_config").read(a, io, dir, problem)) orelse return "unknown";
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{}) catch return "unknown";
    if (doc != .object) return "unknown";
    const t = doc.object.get("model_type") orelse return "unknown";
    return if (t == .string) t.string else "unknown";
}

/// HfText.load with every failure named in `problem`, exactly as main's serve path does.
fn loadText(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, template: ?[]const u8, pa: std.mem.Allocator, problem: *[]const u8) !*hf_text.HfText {
    return hf_text.HfText.load(gpa, io, dir, template, pa, problem) catch |e| {
        if (problem.len == 0) problem.* = @errorName(e);
        return e;
    };
}

/// Opens every engine and runs the server on the first. `main` (main.zig) dispatches `mom-serve` here.
pub fn run(init: std.process.Init, argv: []const []const u8) !u8 {
    const entry = std.Io.Clock.awake.now(init.io).toNanoseconds();
    const gpa = init.gpa;
    const io = init.io;
    const a = init.arena.allocator();
    var problem: []const u8 = "";
    const parsed = (try parseSpecs(a, argv, &problem)) orelse {
        std.debug.print("tensorfold: {s}\n", .{problem});
        return 2;
    };
    var u: cli.Usage = .{};
    var args = blk: {
        // cli.parse wants the positional MODEL DIR; supply the default lane's dir so
        // every ordinary serve flag keeps working unchanged.
        const with_pos = try a.alloc([]const u8, parsed.rest.len + 1);
        with_pos[0] = parsed.specs[0].model;
        @memcpy(with_pos[1..], parsed.rest);
        break :blk cli.parse(a, with_pos, &u) catch |e| switch (e) {
        error.Usage => {
            std.debug.print("usage: tensorfold-native mom-serve [serve flags] --model DIR --name NAME [--lane L] [--pin] ...\ntensorfold mom-serve: error: {s}\n", .{u.message});
            return 2;
        },
        else => |x| return x,
        };
    };
    if (args.name.len == 0) args.name = parsed.specs[0].name; // /v1/models shows --name, not the dir basename
    {
        // Every extra lane id rides the ordinary --alias path, so /v1/models advertises
        // all of them; until B2 routes them, requests naming one are served by the default.
        if (parsed.specs.len > 1) {
            const merged = try a.alloc([]const u8, args.alias.len + parsed.specs.len - 1);
            @memcpy(merged[0..args.alias.len], args.alias);
            for (parsed.specs[1..], 0..) |sp, i| merged[args.alias.len + i] = sp.name;
            args.alias = merged;
        }
    }
    if (momRefusal(args)) |why| {
        std.debug.print("tensorfold: {s}\n", .{why});
        return 2;
    }
    if (try engines.environment(a, args)) |why| {
        std.debug.print("tensorfold: {s}\n", .{why});
        return 1;
    }
    const receipt = @import("core").startup_receipt;
    receipt.enable(io, entry, try @import("sys").environment.read(bool, "TF_STARTUP_RECEIPT", false));
    defer receipt.disable();
    // 1. resolve, type and hold every checkpoint before any open: a bad folder refuses with nothing loaded
    var held: [max_models]Held = undefined;
    for (parsed.specs, 0..) |spec, i| {
        const dir = (try hub.resolve(a, io, init.environ_map, spec.model, &problem)) orelse {
            std.debug.print("tensorfold: {s}\n", .{problem});
            return 1;
        };
        const mt = try modelType(a, io, dir, &problem);
        if (problem.len > 0) {
            std.debug.print("tensorfold: {s}\n", .{problem});
            return 1;
        }
        var unheld: []const u8 = "";
        const h = api.originals.serving(a, dir, false, &unheld) catch |e| switch (e) {
            error.OutOfMemory => |x| return x,
            else => {
                std.debug.print("tensorfold: {s}\n", .{unheld});
                return 1;
            },
        };
        _ = h; // held for the process lifetime, the same contract main relies on: a rollback refuses while served
        held[i] = .{ .spec = spec, .dir = dir, .model_type = mt, .weights = try weightsBytes(a, io, dir) };
    }
    // 2. the plan line: what the co-residency costs, with the reserve counted once for the process.
    // The streams estimate parses --parallel ("auto" counts as one); native admission refines each
    // engine's real room at its open, which is also where an over-budget pair refuses.
    const streams: u32 = if (std.mem.eql(u8, args.parallel, "auto")) 1 else std.fmt.parseInt(u32, args.parallel, 10) catch 1;
    // B2 lane guard: the generation loop decodes with the server-wide tokenizer, so
    // lanes must share one. Identical tokenizer.json digests (or vocab+merges when a
    // checkpoint ships none) prove it; a difference refuses at startup, never mid-reply.
    if (parsed.specs.len > 1) {
        var digests: [max_models]?[32]u8 = @splat(null);
        for (held, 0..) |h, i| {
            if (i >= parsed.specs.len) break;
            digests[i] = tokenizerDigest(io, h.dir);
        }
        for (digests[0..parsed.specs.len], 0..) |x, i| {
            for (digests[0..parsed.specs.len], 0..) |y, j| {
                if (j <= i) continue;
                if (x != null and y != null and !std.mem.eql(u8, &x.?, &y.?)) {
                    std.debug.print("tensorfold: mom-serve: lanes {s} and {s} ship different tokenizers; MoM lanes must share one (the reply path decodes with the default's)\n", .{ parsed.specs[i].name, parsed.specs[j].name });
                    return 2;
                }
            }
        }
    }
    var plans: [max_models]mom_plan.ModelPlan = undefined;
    for (held[0..parsed.specs.len], 0..) |h, i| plans[i] = .{
        .id = h.spec.name,
        .weights = h.weights,
        .streams = streams,
        .stream_bytes = 512 << 20, // the plan line's estimate until /health models[] reports per-engine pools (B2)
        .drafts = !args.no_drafts,
        .pinned = h.spec.pin,
    };
    log.line("{s}", .{try mom_plan.planLine(a, plans[0..parsed.specs.len], .{
        .free = 0, // capacity measurement joins the models[] work (B2); native refuses over-budget opens today
        .reserve = 12 * (1 << 30), // the GB10 budget's reserve, paid once, not per model
    })});
    // 3. open every engine in order (the default first: the server binds to it)
    var opened: [max_models]?api.Opened = undefined;
    for (&opened) |*slot| slot.* = null;
    var opened_n: usize = 0;
    defer {
        var i = opened_n;
        while (i > 0) {
            i -= 1;
            if (opened[i]) |o| o.close(o.ctx);
        }
    }
    var text_arena: std.heap.ArenaAllocator = .init(gpa);
    defer text_arena.deinit();
    var text_problem: []const u8 = "";
    var default_text: ?*hf_text.HfText = null;
    // --lane-priority: open exactly one engine (the trivial one) while the env names
    // the stream priority; cuda_engine reads it at its single stream-creation site.
    const prio_idx = if (parsed.lane_priority) blk: {
        const wts = try a.alloc(u64, parsed.specs.len);
        for (plans[0..parsed.specs.len], 0..) |mp, wi| wts[wi] = mp.weights;
        break :blk priorityIndex(parsed.specs, wts);
    } else null;
    for (held[0..parsed.specs.len], 0..) |h, i| {
        const prio_here = prio_idx != null and i == prio_idx.?;
        if (prio_here) {
            _ = setenv("TF_MOM_TRIVIAL_PRIORITY", "1", 1);
            defer clearTrivialPriority();
        }
        const got: ?api.Opened = if (i == 0) blk: {
            const up = startup.both(io, loadText, .{ gpa, io, h.dir, args.chat_template, text_arena.allocator(), &text_problem }, engines.open, .{ a, gpa, io, h.dir, h.model_type, args, &problem }) catch |e| {
                if (text_problem.len > 0) {
                    std.debug.print("tensorfold: {s}\n", .{text_problem});
                    return 1;
                }
                if (problem.len > 0) {
                    std.debug.print("tensorfold: {s}\n", .{problem});
                    return 1;
                }
                return e;
            } orelse {
                std.debug.print("tensorfold: {s}\n", .{problem});
                return 1;
            };
            default_text = up.text;
            break :blk up.engine;
        } else (try engines.open(a, gpa, io, h.dir, h.model_type, args, &problem)) orelse {
            std.debug.print("tensorfold: mom-serve: {s} failed to open: {s}\n", .{ h.spec.name, if (problem.len > 0) problem else "the backend refused it" });
            return 1;
        };
        opened[i] = got;
        opened_n += 1;
        log.line("mom: {s} ({s}) loaded{s}", .{ h.spec.name, h.model_type, if (h.spec.pin) " (pinned)" else "" });
    }
    // B5: qwen3_5 CUDA reprefills every conversation turn unless a family cache switch is set;
    // the host store keeps message-start states only under TF_QWEN_REQUEST_STARTS=1 (device
    // verified: 99.5% prefix reuse warm, TTFT 175->28 ms on the heavy lane). Say it once, here.
    for (held[0..parsed.specs.len]) |h| {
        if (!std.mem.startsWith(u8, h.model_type, "qwen3_5")) continue;
        const core_profile = @import("core").profile;
        if (core_profile.given(.TF_QWEN_REQUEST_STARTS) == null and core_profile.given(.TF_QWEN_KV_SHARE) == null)
            log.line("mom: warning: {s} ({s}) has no conversation prefix cache on: set TF_QWEN_REQUEST_STARTS=1 (verified 99.5% reuse) — docs/09", .{ h.spec.name, h.model_type });
    }
    const engine = opened[0].?;
    // B2: the routing registry — every opened engine is a lane; roles come from --lane
    // with a deterministic fallback (largest weights = default, smallest = trivial).
    var sleep_ctx: ?*SleepCtx = null; // lives past the registry block: the timer thread needs it
    const reg = blk: {
        const router_lanes = try a.alloc(mom_router.Lane, parsed.specs.len);
        for (parsed.specs, 0..) |sp, i| router_lanes[i] = .{ .id = sp.name, .role = .other };
        var have_trivial = false;
        var have_default = false;
        for (parsed.specs, router_lanes) |sp, *rl| {
            if (std.mem.eql(u8, sp.lane, "trivial")) {
                rl.role = .trivial;
                have_trivial = true;
            } else if (std.mem.eql(u8, sp.lane, "default")) {
                rl.role = .default_lane;
                have_default = true;
            }
        }
        if (!have_trivial or !have_default) {
            var heavy: usize = 0;
            var light: usize = 0;
            for (plans[0..parsed.specs.len], 0..) |mp, i| {
                if (mp.weights > plans[heavy].weights) heavy = i;
                if (mp.weights < plans[light].weights) light = i;
            }
            if (!have_default and heavy != light) router_lanes[heavy].role = .default_lane else if (!have_default) router_lanes[0].role = .default_lane;
            if (!have_trivial and heavy != light) router_lanes[light].role = .trivial;
        }
        const lanes = try a.alloc(mom_registry.Lane, parsed.specs.len);
        for (opened[0..parsed.specs.len], 0..) |o, i| lanes[i] = .{ .id = parsed.specs[i].name, .engine = o.?.engine, .info = o.?.engine.info() };
        // Sleep/wake (docs/08): non-trivial lanes may park after idle; the trivial lane is
        // permanently hot by policy (idle_ns 0 -> every lifecycle hook is an inert no-op).
        var lifecycles: []mom_sleep.LaneLifecycle = &.{};
        if (parsed.lane_idle_s > 0 and parsed.specs.len > 1) {
            lifecycles = try a.alloc(mom_sleep.LaneLifecycle, parsed.specs.len);
            for (router_lanes, lifecycles) |rl, *lc| lc.* = .{
                .io = io,
                .idle_ns = if (rl.role == .trivial) 0 else parsed.lane_idle_s * std.time.ns_per_s,
            };
            const sc = try a.create(SleepCtx);
            sc.* = .{
                .held = try a.dupe(Held, held[0..parsed.specs.len]),
                .lanes = lanes,
                .opened = opened[0..parsed.specs.len],
                .args = args,
                .a = a,
                .gpa = gpa,
            };
            sleep_ctx = sc;
        }
        var sticky_log: ?std.Io.File = null;
        if (parsed.mom_log) |path| sticky_log = openDecisionLog(io, path) catch null;
        const reg_ptr = try a.create(mom_registry.Registry);
        reg_ptr.* = .{
            .lanes = lanes,
            .router = .{
                .cfg = .{ .lanes = router_lanes, .sticky_enabled = !parsed.no_sticky, .sticky_capacity = 4096 },
                .sticky = try mom_router.Sticky.init(a, 4096),
            },
            .log_file = sticky_log,
            .io = io,
            .lifecycles = lifecycles,
            .reopen = if (sleep_ctx) |sc| mom_registry.Reopen{ .ctx = @ptrCast(sc), .call = reopenLane } else null,
        };
        log.line("mom: routing over {d} lane{s} (default={s}, trivial={s})", .{ lanes.len, if (lanes.len == 1) "" else "s", laneId(router_lanes, .default_lane), laneId(router_lanes, .trivial) });
        break :blk reg_ptr;
    };
    const text = default_text.?;
    // 4. serve on the default model, every id advertised: the extra ids join as aliases until B2 routes them
    if (sleep_ctx) |sc| { // arm the park timer once the registry can see every lane
        const sl = try a.create(Sleeper);
        sl.* = .{ .reg = reg, .ctx = sc, .io = io };
        const th = try std.Thread.spawn(.{}, Sleeper.run, .{sl});
        th.detach();
        log.line("mom: sleep/wake armed: non-trivial lanes park after {d}s idle", .{parsed.lane_idle_s});
    }
    const served = hub.servedName(args.name, args.model, held[0].dir);
    var tool_ids = tool_id_key.load(a, init.environ_map.get(tool_id_key.name), &problem) catch |e| switch (e) {
        error.OutOfMemory => |x| return x,
        error.Refused => {
            std.debug.print("tensorfold: {s}\n", .{problem});
            return 1;
        },
    };
    // the stop EngineStop closes the default engine once; the others go through the defer above
    const DefaultStop = struct {
        var closed = false;
        fn close(p: *anyopaque) void {
            if (closed) return;
            closed = true;
            const o: *?api.Opened = @ptrCast(@alignCast(p));
            if (o.*) |open| {
                open.close(open.ctx);
                o.* = null;
            }
        }
    };
    return serve.run(gpa, io, args, .{
        .engine = engine.engine,
        .stop = .{ .ctx = &opened[0], .close = DefaultStop.close },
        .text = text.text(),
        .served = served,
        .model_dir = held[0].dir,
        .mom = reg,
        .model_type = held[0].model_type,
        .sampling = try @import("family_sampling.zig").resolve(a, held[0].model_type, null, .{
            .temperature = args.temperature, .top_p = args.top_p, .top_k = args.top_k, .min_p = args.min_p,
        }),
        .environ = init.environ_map,
        .started = entry,
        .identity = engines.identity(a, args),
        .tool_id_key = if (tool_ids) |*key| key else null,
    });
}

// -- host tests: the argv split, which must never eat an ordinary flag ------------------------------------------

test "parseSpecs takes the mom groups and hands everything else to cli.parse" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    const argv = [_][]const u8{
        "--host", "127.0.0.1", "--port", "8123", "--parallel", "8",
        "--model", "/models/small", "--name", "qwen3.5-2b", "--lane", "trivial", "--pin",
        "--model", "/models/big", "--name", "qwen3.6-35b",
        "--context", "32768", "--drafter", "/models/draft",
    };
    const p = (try parseSpecs(a, &argv, &problem)).?;
    try std.testing.expectEqual(@as(usize, 2), p.specs.len);
    try std.testing.expectEqualStrings("qwen3.5-2b", p.specs[0].name);
    try std.testing.expectEqualStrings("trivial", p.specs[0].lane);
    try std.testing.expect(p.specs[0].pin);
    try std.testing.expectEqualStrings("default", p.specs[1].lane);
    try std.testing.expect(!p.specs[1].pin);
    for ([_][]const u8{ "--host", "127.0.0.1", "--port", "8123", "--parallel", "8", "--context", "32768", "--drafter", "/models/draft" }) |want| {
        var found = false;
        for (p.rest) |r| {
            if (std.mem.eql(u8, r, want)) found = true;
        }
        try std.testing.expect(found);
    }
    for (p.rest) |r| {
        try std.testing.expect(!std.mem.eql(u8, r, "--lane"));
        try std.testing.expect(!std.mem.eql(u8, r, "--pin"));
    }
}

test "priorityIndex picks the explicit trivial, else the lightest, else none" {
    const S = Spec;
    const specs = [_]S{
        .{ .model = "big", .name = "big", .lane = "default" },
        .{ .model = "small", .name = "small", .lane = "trivial" },
    };
    try std.testing.expectEqual(@as(?usize, 1), priorityIndex(&specs, &.{ 900, 100 }));
    const unlabeled = [_]S{
        .{ .model = "a", .name = "a" },
        .{ .model = "b", .name = "b" },
    };
    try std.testing.expectEqual(@as(?usize, 1), priorityIndex(&unlabeled, &.{ 900, 100 }));
    try std.testing.expectEqual(@as(?usize, 0), priorityIndex(&unlabeled, &.{ 100, 900 }));
    try std.testing.expectEqual(@as(?usize, null), priorityIndex(unlabeled[0..1], &.{ 100 })); // one lane: nothing to preempt
    try std.testing.expectEqual(@as(?usize, null), priorityIndex(&unlabeled, &.{ 500, 500 })); // tie: no clear trivial
}

test "parseSpecs refuses: no models, a nameless model, and duplicate names" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    try std.testing.expect(try parseSpecs(a, &.{}, &problem) == null);
    try std.testing.expect(std.mem.indexOf(u8, problem, "at least one") != null);
    problem = "";
    try std.testing.expect(try parseSpecs(a, &.{ "--model", "/x" }, &problem) == null);
    try std.testing.expect(std.mem.indexOf(u8, problem, "--name") != null);
    problem = "";
    try std.testing.expect(try parseSpecs(a, &.{ "--model", "/x", "--name", "same", "--model", "/y", "--name", "same" }, &problem) == null);
    try std.testing.expect(std.mem.indexOf(u8, problem, "share the name") != null);
}

test "weightsBytes adds the tensor files and ignores the rest" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [1024]u8 = undefined;
    @memset(&buf, 'x');
    const want: []const u8 = &buf;
    try tmp.dir.writeFile(io, .{ .sub_path = "model-00001-of-00002.safetensors", .data = want });
    try tmp.dir.writeFile(io, .{ .sub_path = "model-00002-of-00002.safetensors", .data = want });
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{}" });
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    try std.testing.expectEqual(@as(u64, 2048), try weightsBytes(a, io, dir));
}
