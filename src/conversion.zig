const std = @import("std");

pub fn bf16ToFP32One(value: u16) f32 {
    const wide_value: u32 = value;
    return @bitCast(wide_value << 16);
}

pub fn bf16ToFP32(allocator: std.mem.Allocator, buf: []align(2) u8) ![]f32 {
    // Reinterpret as u16
    var u16_buf: []u16 = undefined;
    u16_buf.len = buf.len / 2;
    u16_buf.ptr = @ptrCast(buf.ptr);

    const f32_buf = try allocator.alloc(f32, buf.len / 2);
    for (u16_buf, 0..) |u16_value, i| {
        f32_buf[i] = bf16ToFP32One(u16_value);
    }

    return f32_buf;
}
