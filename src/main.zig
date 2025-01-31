const std = @import("std");
const lib = @import("lib.zig");

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const file_path = "model.safetensors";
    const file = try std.fs.cwd().openFile(file_path, .{ .mode = .read_only });
    defer file.close();
    const header = try lib.read_header(allocator, file.reader().any());
    const byte_buffer_pos = try file.getPos();
    var it = header.iterator();
    while (it.next()) |entry| {
        // std.debug.print("{s}, {s}, {any}\n", .{ entry.key_ptr.*, entry.value_ptr.dtype, entry.value_ptr.shape });
        const begin = entry.value_ptr.data_offsets[0];
        const end = entry.value_ptr.data_offsets[1];
        const length = (end - begin) / 2;

        const buf = try allocator.alloc(u8, length * 2);
        try file.seekTo(byte_buffer_pos + begin);
        _ = try file.read(buf);

        // Reinterprect as u16
        var u16_buf: []u16 = undefined;
        u16_buf.len = length;
        u16_buf.ptr = @alignCast(@ptrCast(buf.ptr));

        // Reinterpret as f16
        var f16_buf: []f16 = undefined;
        f16_buf.len = length;
        f16_buf.ptr = @alignCast(@ptrCast(buf.ptr));

        for (0..length) |i| {
            const value: u16 = u16_buf[i];
            // std.debug.print("{any}\n", .{value});
            const sign = (1 << 15) & value;
            const exponent = (((value & 0b0_11111111_0000000) >> 7) - 112) << 10;
            const fraction = (value & 0b0_00000000_1111111) << 3;
            const bits = sign | exponent | fraction;
            f16_buf[i] = @bitCast(bits);
            // std.debug.print("{}\n", .{bits});
            // break;
        }

        // const batch_size: comptime_int = 256 / 16;
        // var i: u16 = 0;
        // const VT = @Vector(batch_size, u16);
        // const sign_mask: VT = @splat(1 << 15);
        // const exp_mask: VT = @splat(0b0_11111111_0000000);
        // const frac_mask: VT = @splat(0b0_00000000_1111111);
        // while (i < length) {
        //     const vec: VT = u16_buf[i..][0..batch_size].*;
        //     std.debug.print("{any}\n", .{vec});
        //     const sign = vec & sign_mask;
        //     const exp = (((vec & exp_mask) >> @splat(7)) - @as(VT, @splat(112))) << @splat(10);
        //     const frac = (vec & frac_mask) << @splat(3);
        //     const bits = sign | exp | frac;
        //     std.debug.print("{}\n", .{bits});
        //     f16_buf.ptr[i..][0..batch_size].* = @bitCast(bits);
        //     std.debug.print("{any}\n", .{f16_buf[i .. i + batch_size]});
        //     i += batch_size;
        //     break;
        // }

        break;
    }
}
