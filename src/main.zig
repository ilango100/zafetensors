const std = @import("std");
const SafeTensors = @import("SafeTensors.zig");
const convert = @import("convert.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const file_path = "model.safetensors";
    const file = try std.fs.cwd().openFile(file_path, .{ .mode = .read_only });
    var st = try SafeTensors.openFile(allocator, file);
    defer st.deinit();

    var it = st.header.iterator();
    var i: u32 = 0;
    while (it.next()) |entry| {
        std.debug.print("{s}\n", .{entry.key_ptr.*});
        const buf = try st.loadTensor(entry.key_ptr.*);
        defer allocator.free(buf);
        switch (entry.value_ptr.dtype) {
            .BF16 => {
                const f16_buf = try convert.bf16ToFP32(allocator, buf);
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
}
