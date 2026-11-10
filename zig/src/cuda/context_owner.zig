//! Process-wide custody of a device's primary context while several engines share it (Mixture-of-Models).
//!
//! CUDA's primary context refcounts at the driver, so N retains are safe. What is NOT refcounted is
//! `Context.deinit`'s teardown: the device-wide synchronize and `copy_source.dropSpares` belong to the
//! CONTEXT, not the engine — with two engines loaded, the first to close would drop a sibling's spare
//! pinned buffers (and pay a whole-device wait for nothing). `acquire`/`release` decide whether a
//! teardown is the last one: only then do those run. Sticky-fault health already refcounts itself
//! (context_health.zig) and the driver's own Release refcounts itself; both stay per-Context.

const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;

const Entry = struct { driver: usize = 0, context: abi.Context = null, refs: usize = 0 };
var entries: [16]Entry = @splat(.{});
var mutex: std.atomic.Mutex = .unlocked;

fn lock() void {
    while (!mutex.tryLock()) std.Thread.yield() catch {};
}

/// Call after a successful `cuDevicePrimaryCtxRetain` on this context, before it escapes.
/// A full table is never expected (a process serves a handful of engines); if it happens the
/// context stays untracked and every deinit keeps today's full-teardown behaviour — safe, just
/// not shared.
pub fn acquire(d: *const Driver, context: abi.Context) void {
    lock();
    defer mutex.unlock();
    for (&entries) |*e| if (e.refs > 0 and e.driver == @intFromPtr(d) and e.context == context) {
        e.refs += 1;
        return;
    };
    for (&entries) |*e| if (e.refs == 0) {
        e.* = .{ .driver = @intFromPtr(d), .context = context, .refs = 1 };
        return;
    };
}

/// Call from `Context.deinit` after the driver's own Release; true when this was the last live owner,
/// i.e. the caller must still synchronize the device and drop the context's copy-source spares.
/// An untracked release returns true (full teardown, as before multi-context custody existed).
pub fn release(d: *const Driver, context: abi.Context) bool {
    lock();
    defer mutex.unlock();
    for (&entries) |*e| if (e.refs > 0 and e.driver == @intFromPtr(d) and e.context == context) {
        e.refs -= 1;
        const last = e.refs == 0;
        if (last) e.* = .{};
        return last;
    };
    return true;
}

/// Live owners of `context` (0 when untracked): for /health and tests.
pub fn owners(d: *const Driver, context: abi.Context) usize {
    lock();
    defer mutex.unlock();
    for (entries) |e| if (e.refs > 0 and e.driver == @intFromPtr(d) and e.context == context) return e.refs;
    return 0;
}

test "multi-context custody: the second owner's release is not the last; the first's teardown is shared" {
    var d: Driver = undefined;
    const shared_ctx: abi.Context = @ptrFromInt(0x51a0);
    const other_ctx: abi.Context = @ptrFromInt(0x51a1);

    acquire(&d, shared_ctx); // engine A opens
    acquire(&d, shared_ctx); // engine B opens the same primary context
    try std.testing.expectEqual(@as(usize, 2), owners(&d, shared_ctx));
    try std.testing.expect(!release(&d, shared_ctx)); // A closes: spares and the device stay for B
    try std.testing.expectEqual(@as(usize, 1), owners(&d, shared_ctx));
    try std.testing.expect(release(&d, shared_ctx)); // B closes last: full teardown runs
    try std.testing.expectEqual(@as(usize, 0), owners(&d, shared_ctx));

    // a different context is its own custody; an untracked release still demands a full teardown
    acquire(&d, other_ctx);
    try std.testing.expect(release(&d, shared_ctx)); // untracked: today's behaviour
    try std.testing.expect(release(&d, other_ctx));
}

test "multi-context custody: two drivers on one context are separate custodies" {
    var d1: Driver = undefined;
    var d2: Driver = undefined;
    const ctx: abi.Context = @ptrFromInt(0x52b0);
    acquire(&d1, ctx);
    acquire(&d2, ctx); // different driver pointer: the entry keys on both
    try std.testing.expectEqual(@as(usize, 1), owners(&d1, ctx));
    try std.testing.expectEqual(@as(usize, 1), owners(&d2, ctx));
    try std.testing.expect(release(&d1, ctx));
    try std.testing.expect(release(&d2, ctx));
}
