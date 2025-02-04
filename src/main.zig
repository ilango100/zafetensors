const std = @import("std");
const SafeTensors = @import("SafeTensors.zig");
const convert = @import("convert.zig");

const usage =
    \\ Available subcommands:
    \\    show
    \\    convert
;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();
    _ = args.next().?;
    var exit_code: u8 = 0;
    const command = args.next() orelse "";

    // Call respective command
    if (std.mem.eql(u8, command, "show")) {
        exit_code = show(allocator, &args);
    } else if (std.mem.eql(u8, command, "convert")) {
        exit_code = try convert.convert(allocator, &args);
    } else {
        std.debug.print("{s}\n", .{usage});
        exit_code = 1;
    }
    std.process.exit(exit_code);
}

const show_usage =
    \\ Usage: show <file>
    \\  <file>: safetensors file path
;

pub fn show(allocator: std.mem.Allocator, args: *std.process.ArgIterator) u8 {
    const path = args.next() orelse {
        std.debug.print("{s}\n", .{show_usage});
        return 1;
    };
    var st = SafeTensors.open(allocator, path) catch |e| {
        std.debug.print("Unable to open safetensors file: {}", .{e});
        return 1;
    };
    defer st.deinit();
    var it = st.header.iterator();
    while (it.next()) |entry| {
        std.debug.print("{s} {} {any}\n", .{ entry.key_ptr.*, entry.value_ptr.dtype, entry.value_ptr.shape });
    }
    return 0;
}
