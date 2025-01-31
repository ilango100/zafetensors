const std = @import("std");
const SafeTensors = @import("SafeTensors.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const file_path = "model.safetensors";
    const file = try std.fs.cwd().openFile(file_path, .{ .mode = .read_only });
    var st = try SafeTensors.load(allocator, file);
    defer st.deinit();

    var it = st.header.?.iterator();
    while (it.next()) |entry| {
        std.debug.print("{s}, {}, {any}\n", .{ entry.key_ptr.*, entry.value_ptr.dtype, entry.value_ptr.shape });
    }
}
