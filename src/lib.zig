const std = @import("std");
const json = std.json;

const TensorInfo = struct {
    dtype: []const u8,
    shape: []u64,
    data_offsets: [2]u64,
};

pub fn read_header(allocator: std.mem.Allocator, reader: std.io.AnyReader) !std.StringArrayHashMap(TensorInfo) {
    // Read whole header into memory
    const header_len = try reader.readInt(u64, .little);
    const header_buf = try allocator.alloc(u8, header_len); // No dealloc because the strings will be directly pointed here
    _ = try reader.readAtLeast(header_buf, header_len);

    // Parse the metadata into ordered hashmap
    var scanner = json.Scanner.initCompleteInput(allocator, header_buf);
    defer scanner.deinit();
    var header = std.StringArrayHashMap(TensorInfo).init(allocator);
    var token = try scanner.next();
    if (token != .object_begin) // Must be an object
        return error.UnexpectedToken;

    // Parse key-value pairs of weight name - tensor info
    while (true) {
        token = try scanner.next();
        switch (token) {
            .string => |key| {
                // Skip special metadata
                if (std.mem.eql(u8, key, "__metadata__")) {
                    try scanner.skipValue();
                    continue;
                }
                // Load object corresponding to the key
                const tensor_info = try json.innerParse(TensorInfo, allocator, &scanner, .{ .max_value_len = json.default_max_value_len, .allocate = .alloc_if_needed });
                try header.putNoClobber(key, tensor_info); // Insert with check that the key doesn't already exist
            },
            .object_end => break,
            else => unreachable,
        }
    }
    return header;
}
