const std = @import("std");
const SafeTensors = @import("SafeTensors.zig");
const conversion = @import("conversion.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();
    _ = args.next().?;
    var exit_code = 0;
    if (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "show")) {
            exit_code = show(allocator, &args);
        } else if (std.mem.eql(u8, arg, "convert")) {
            exit_code = try convert(allocator, &args);
        }
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

const convert_usage =
    \\ Usage: convert <file> <dtype>
    \\  <file>: safetensors file path
    \\  <dtype>: Dtype to convert to
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

    var st = try SafeTensors.open(allocator, path);
    defer st.deinit();

    var it = st.header.iterator();
    var i: u32 = 0;
    while (it.next()) |entry| {
        std.debug.print("{s}\n", .{entry.key_ptr.*});
        const buf = try st.loadTensor(entry.key_ptr.*);
        defer allocator.free(buf);
        switch (entry.value_ptr.dtype) {
            .BF16 => {
                const f16_buf = try conversion.bf16ToFP32(allocator, buf);
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
