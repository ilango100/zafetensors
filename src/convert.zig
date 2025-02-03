const std = @import("std");
const sts = @import("SafeTensors.zig");

const convert_usage =
    \\ Usage: convert <file> <dtype>
    \\  <file>: safetensors file path
    \\  <dtype>: Dtype to convert to
    \\  Output file will be written as <file>_<dtype>.safetensors
;

pub fn convert(allocator: std.mem.Allocator, args: *std.process.ArgIterator) !u8 {
    const path = args.next() orelse {
        std.debug.print("{s}\n", .{convert_usage});
        return 1;
    };
    const dtype = args.next() orelse {
        std.debug.print("{s}\n", .{convert_usage});
        return 1;
    };

    std.debug.print("convert {s} {s}\n", .{ path, dtype });

    var st = try sts.open(allocator, path);
    defer st.deinit();

    var it = st.header.iterator();
    var i: u32 = 0;
    while (it.next()) |entry| {
        std.debug.print("{s}\n", .{entry.key_ptr.*});
        const buf = try st.loadTensor(entry.key_ptr.*);
        defer allocator.free(buf);
        switch (entry.value_ptr.dtype) {
            .BF16 => {
                const f16_buf = try bf16ToFP32(allocator, buf);
                defer allocator.free(f16_buf);
                for (f16_buf[0..16]) |value| {
                    std.debug.print("{e:.4} ", .{value});
                }
                std.debug.print("\n", .{});
            },
            else => unreachable,
        }
        i += 1;
        if (i >= 2)
            break;
    }
    return 0;
}

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
