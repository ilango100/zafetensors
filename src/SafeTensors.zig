const std = @import("std");
const json = std.json;

const TensorInfo = struct {
    dtype: []u8,
    shape: []u64,
    data_offsets: [2]u64,
};

allocator: std.mem.Allocator,
header_buf: ?[]u8 = null,
byte_buffer_pos: u64 = undefined,
header: ?std.StringArrayHashMap(TensorInfo) = null,
file: ?std.fs.File = null,

const Self = @This();

pub fn load(allocator: std.mem.Allocator, file: std.fs.File) !Self {
    var self = Self{
        .allocator = allocator,
        .file = file,
    };
    try self.load_header_buf();
    try self.parse_header();
    return self;
}

pub fn deinit(self: *Self) void {
    if (self.header) |*header| {
        var it = header.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.value_ptr.shape);
            self.allocator.free(entry.value_ptr.dtype);
        }
        header.deinit();
        self.header = null;
    }
    if (self.header_buf) |header_buf| {
        self.allocator.free(header_buf);
        self.header_buf = null;
    }
    if (self.file) |file| {
        file.close();
        self.file = null;
    }
}

fn load_header_buf(self: *Self) !void {
    if (self.header_buf != null) {
        return;
    }
    // Load whole header into memory
    const reader = self.file.?.reader();
    const header_len = try reader.readInt(u64, .little);
    const header_buf = try self.allocator.alloc(u8, header_len);
    const read_len = try reader.read(header_buf);
    if (header_len != read_len)
        return error.UnableToLoadFullHeader;
    self.header_buf = header_buf;
    self.byte_buffer_pos = header_len + 8;
}

fn parse_header(self: *Self) !void {
    if (self.header != null) {
        return;
    }
    // Parse the metadata into ordered hashmap
    var scanner = json.Scanner.initCompleteInput(self.allocator, self.header_buf.?);
    defer scanner.deinit();
    var header = std.StringArrayHashMap(TensorInfo).init(self.allocator);
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
                const tensor_info = try json.innerParse(TensorInfo, self.allocator, &scanner, .{ .max_value_len = json.default_max_value_len, .allocate = .alloc_if_needed });
                try header.putNoClobber(key, tensor_info); // Insert with check that the key doesn't already exist
            },
            .object_end => break,
            else => return error.UnexpectedToken,
        }
    }
    self.header = header;
}

pub fn load_tensor(self: Self, name: []u8) ![]u8 {
    // Lookup tensor in the header
    const header = self.header orelse return error.HeaderNotInitialized;
    const tinfo = header.get(name) orelse return error.TensorNotFound;

    const begin = tinfo.data_offsets[0];
    const end = tinfo.data_offsets[1];
    const len = end - begin;

    const buf = try self.allocator.alloc(u8, len);
    try self.file.seekTo(begin);
    const read_len = try self.file.read(buf);
    if (len != read_len)
        return error.UnableToLoadFullTensor;
    return buf;
}
