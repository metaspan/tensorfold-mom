//! Per-lane engine lifecycle for MoM sleep/wake (docs/08-b4-sleep-wake.md).
//!
//! The state machine generalized from server/idle_unload.zig's Manager: same
//! lifecycle semantics (leases prevent closing an engine mid-request; a
//! request arriving during `unloading` cancels the park), scoped to one lane
//! slot instead of the server's single engine. Pure logic: no CUDA, no clock
//! of its own — the caller passes `now_ns` it read from server/clock.zig, so
//! every transition is host-testable.

const std = @import("std");

pub const State = enum { resident, unloading, unloaded, loading };

pub const LaneLifecycle = struct {
    mutex: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    io: std.Io,
    state: State = .resident,
    /// Engine-using requests in flight; a park waits for zero.
    active: u32 = 0,
    /// Last lease() on the awake clock; refreshed by route() and lease().
    last_used_ns: i64 = 0,
    /// Idle window past `last_used_ns` before the lane may sleep; 0 = never.
    idle_ns: i64 = 0,
    /// Observability counters (brief component 7: switch latency + counts).
    sleep_total: u32 = 0,
    wake_total: u32 = 0,
    last_wake_ns: i64 = 0,

    pub fn lock(self: *LaneLifecycle) void {
        self.mutex.lockUncancelable(self.io);
    }
    pub fn unlock(self: *LaneLifecycle) void {
        self.mutex.unlock(self.io);
    }

    /// A request starts engine work: raises `active`, refreshes the idle
    /// deadline, and cancels an in-flight park (idle_unload.begin semantics).
    pub fn lease(self: *LaneLifecycle, now_ns: i64) void {
        self.lock();
        defer self.unlock();
        if (self.state == .unloading) self.state = .resident;
        self.active += 1;
        self.last_used_ns = now_ns;
    }

    /// The request ended, any outcome: drops `active` and re-arms the window.
    pub fn release(self: *LaneLifecycle, now_ns: i64) void {
        self.lock();
        defer self.unlock();
        if (self.active > 0) self.active -= 1;
        self.last_used_ns = now_ns;
        self.cond.broadcast(self.io);
    }

    /// The timer's predicate: eligible iff resident, idle past the window,
    /// and nothing in flight. `idle_ns == 0` (never sleep) never qualifies.
    pub fn shouldSleep(self: *LaneLifecycle, now_ns: i64) bool {
        self.lock();
        defer self.unlock();
        return self.idle_ns > 0 and self.state == .resident and
            self.active == 0 and now_ns - self.last_used_ns >= self.idle_ns;
    }

    /// Claim the park: resident -> unloading when safe. False = somebody is
    /// using it (or state moved), keep serving.
    pub fn claimUnload(self: *LaneLifecycle) bool {
        self.lock();
        defer self.unlock();
        if (self.state != .resident or self.active > 0) return false;
        self.state = .unloading;
        return true;
    }

    /// The weights+KV were closed: unloading (or cancelled-to-resident) -> unloaded.
    /// Cancelling back to resident mid-close is lease()'s; this only lands a
    /// still-unloading claim.
    pub fn finishUnload(self: *LaneLifecycle) void {
        self.lock();
        defer self.unlock();
        if (self.state != .unloading) return; // a lease cancelled it
        self.state = .unloaded;
        self.sleep_total += 1;
    }

    /// Claim a wake: unloaded -> loading; the caller then opens the engine and
    /// calls finishWake (or wakeFailed to return to unloaded).
    pub fn claimWake(self: *LaneLifecycle) bool {
        self.lock();
        defer self.unlock();
        if (self.state != .unloaded) return self.state == .resident or self.state == .loading;
        self.state = .loading;
        return true;
    }

    /// The reopened engine is in place: loading -> unloaded... -> resident,
    /// stamping the wake for latency accounting.
    pub fn finishWake(self: *LaneLifecycle, now_ns: i64) void {
        self.lock();
        defer self.unlock();
        self.state = .resident;
        self.wake_total += 1;
        self.last_wake_ns = now_ns;
        self.last_used_ns = now_ns;
        self.cond.broadcast(self.io);
    }

    /// A failed open returns the lane to unloaded (retryable; server stays up).
    pub fn wakeFailed(self: *LaneLifecycle) void {
        self.lock();
        defer self.unlock();
        if (self.state == .loading) self.state = .unloaded;
        self.cond.broadcast(self.io);
    }
};

// ─── host tests ───────────────────────────────────────────────────────────────

test "sleep: idle arithmetic respects the never-sleep window" {
    var lc: LaneLifecycle = .{ .io = std.testing.io, .idle_ns = 0 };
    try std.testing.expect(!lc.shouldSleep(1_000_000_000_000));
    lc.idle_ns = 20 * std.time.ns_per_s;
    lc.last_used_ns = 0;
    try std.testing.expect(!lc.shouldSleep(19 * std.time.ns_per_s));
    try std.testing.expect(lc.shouldSleep(20 * std.time.ns_per_s));
}

test "sleep: lease blocks the park and cancels a claimed one" {
    var lc: LaneLifecycle = .{ .io = std.testing.io, .idle_ns = 10 * std.time.ns_per_s };
    lc.last_used_ns = 0;
    try std.testing.expect(lc.claimUnload());
    try std.testing.expectEqual(State.unloading, lc.state);
    // A request arriving mid-park cancels it (idle_unload.begin semantics).
    lc.lease(50 * std.time.ns_per_s);
    try std.testing.expectEqual(State.resident, lc.state);
    lc.finishUnload(); // stale claim must not unload a resident lane
    try std.testing.expectEqual(State.resident, lc.state);
    lc.release(60 * std.time.ns_per_s);
    try std.testing.expectEqual(@as(u32, 0), lc.active);
}

test "sleep: sleep then wake round-trips counters" {
    var lc: LaneLifecycle = .{ .io = std.testing.io, .idle_ns = std.time.ns_per_s };
    lc.last_used_ns = 0;
    try std.testing.expect(lc.claimUnload());
    lc.finishUnload();
    try std.testing.expectEqual(State.unloaded, lc.state);
    try std.testing.expectEqual(@as(u32, 1), lc.sleep_total);
    // Wake: only unloaded claims; a second waker sees loading/resident as ok.
    try std.testing.expect(lc.claimWake());
    try std.testing.expect(!lc.claimUnload()); // cannot park a loading lane
    lc.finishWake(90 * std.time.ns_per_s);
    try std.testing.expectEqual(State.resident, lc.state);
    try std.testing.expectEqual(@as(u32, 1), lc.wake_total);
    try std.testing.expectEqual(@as(i64, 90 * std.time.ns_per_s), lc.last_wake_ns);
}

test "sleep: failed wake stays retryable" {
    var lc: LaneLifecycle = .{ .io = std.testing.io, .idle_ns = std.time.ns_per_s };
    try std.testing.expect(lc.claimUnload());
    lc.finishUnload();
    try std.testing.expect(lc.claimWake());
    lc.wakeFailed();
    try std.testing.expectEqual(State.unloaded, lc.state);
    try std.testing.expect(lc.claimWake()); // retry path open
    lc.finishWake(std.time.ns_per_s);
    try std.testing.expectEqual(@as(u32, 1), lc.wake_total);
}
