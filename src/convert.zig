const std = @import("std");
const SafeTensors = @import("SafeTensors.zig");

const convert_usage =
    \\ Usage: convert <in_file> <out_file>
    \\ Converts BF16 tensors to FP32
    \\  <in_file>:  safetensors input path
    \\  <out_file>: safetensors output path
;

pub fn convert(allocator: std.mem.Allocator, args: *std.process.ArgIterator) !u8 {
    const in_path = args.next() orelse {
        std.debug.print("{s}\n", .{convert_usage});
        return 1;
    };
    const out_path = args.next() orelse {
        std.debug.print("{s}\n", .{convert_usage});
        return 1;
    };

    var in_st = try SafeTensors.open(allocator, in_path);
    defer in_st.deinit();

    var out_st = try SafeTensors.create(allocator, out_path);
    defer out_st.deinit();

    // Update output file header
    var it = in_st.header.iterator();
    while (it.next()) |entry| {
        try out_st.header.putNoClobber(entry.key_ptr.*, .{
            .dtype = if (entry.value_ptr.dtype == .BF16) .F32 else entry.value_ptr.dtype,
            .shape = entry.value_ptr.shape,
        });
    }
    defer out_st.header.clearAndFree(); // Avoid double free of keys from both in and out

    // Write output header
    try out_st.writeHeader();

    // Write output tensors
    it = in_st.header.iterator();
    while (it.next()) |entry| {
        std.debug.print("{s}\n", .{entry.key_ptr.*});
        const in_buf = try in_st.loadTensor(entry.key_ptr.*);
        defer allocator.free(in_buf);
        const out_buf = if (entry.value_ptr.dtype == .BF16) try bf16ToFP32(allocator, in_buf) else in_buf;
        try out_st.writeTensor(entry.key_ptr.*, out_buf);
        if (entry.value_ptr.dtype == .BF16)
            allocator.free(out_buf);
    }
    return 0;
}

pub fn bf16ToFP32One(value: u16) f32 {
    const wide_value: u32 = value;
    return @bitCast(wide_value << 16);
}

pub fn bf16ToFP32(allocator: std.mem.Allocator, buf: []align(2) u8) ![]u8 {
    // Reinterpret as u16
    var u16_buf: []u16 = undefined;
    u16_buf.len = buf.len / 2;
    u16_buf.ptr = @ptrCast(buf.ptr);

    const f32_buf = try allocator.alloc(f32, buf.len / 2);
    for (u16_buf, 0..) |u16_value, i| {
        f32_buf[i] = bf16ToFP32One(u16_value);
    }

    var out_buf: []u8 = undefined;
    out_buf.len = f32_buf.len * 4;
    out_buf.ptr = @ptrCast(f32_buf.ptr);

    return out_buf;
}
