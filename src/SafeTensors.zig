const std = @import("std");
const json = std.json;

const DType = enum {
    F64,
    F32,
    F16,
    BF16,
    I64,
    U64,
    I32,
    U32,
    I16,
    U16,
    I8,
    U8,
    BOOL,
    F8_E4M3,
    F8_E5M2,
};

const TensorInfo = struct {
    dtype: DType,
    shape: []u64,
    offset: u64,
    len: u64,
};

allocator: std.mem.Allocator,
file: std.fs.File,
header: std.StringArrayHashMap(TensorInfo),

const Self = @This();

pub fn openFile(allocator: std.mem.Allocator, file: std.fs.File) !Self {
    const header_buf = try loadHeaderBuf(allocator, file);
    defer allocator.free(header_buf);
    const header = try parseHeader(allocator, header_buf);
    return Self{
        .allocator = allocator,
        .file = file,
        .header = header,
    };
}

pub fn deinit(self: *Self) void {
    var it = self.header.iterator();
    while (it.next()) |entry| {
        self.allocator.free(entry.key_ptr.*);
        self.allocator.free(entry.value_ptr.shape);
    }
    self.header.deinit();
    self.file.close();
}

fn loadHeaderBuf(allocator: std.mem.Allocator, file: std.fs.File) ![]u8 {
    // Load whole header into memory
    const reader = file.reader();
    const header_len = try reader.readInt(u64, .little);
    const header_buf = try allocator.alloc(u8, header_len);
    const read_len = try reader.read(header_buf);
    if (read_len != header_len)
        return error.UnableToLoadFullHeader;
    return header_buf;
}

fn parseHeader(allocator: std.mem.Allocator, header_buf: []u8) !std.StringArrayHashMap(TensorInfo) {
    // Parse the header into ordered hashmap
    var scanner = json.Scanner.initCompleteInput(allocator, header_buf);
    defer scanner.deinit();
    const byte_buffer_pos = header_buf.len + 8;
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
                const tinfo = try json.innerParse(
                    struct {
                        dtype: []u8,
                        shape: []u64,
                        data_offsets: [2]u64,
                    },
                    allocator,
                    &scanner,
                    .{ .max_value_len = json.default_max_value_len, .allocate = .alloc_if_needed },
                );

                // Insert only new keys
                const tensor_key = try allocator.dupe(u8, key);
                try header.putNoClobber(tensor_key, TensorInfo{
                    .dtype = std.meta.stringToEnum(DType, tinfo.dtype).?,
                    .shape = tinfo.shape,
                    .offset = byte_buffer_pos + tinfo.data_offsets[0],
                    .len = tinfo.data_offsets[1] - tinfo.data_offsets[0],
                });
                allocator.free(tinfo.dtype);
            },
            .object_end => break,
            else => return error.UnexpectedToken,
        }
    }
    return header;
}

pub fn loadTensor(self: Self, name: []const u8) ![]align(8) u8 {
    // Lookup tensor in the header
    const tinfo = self.header.get(name) orelse return error.TensorNotFound;
    const buf = try self.allocator.alignedAlloc(u8, 8, tinfo.len);
    try self.file.seekTo(tinfo.offset);
    const read_len = try self.file.read(buf);
    if (read_len != tinfo.len)
        return error.UnableToLoadFullTensor;
    return buf;
}
