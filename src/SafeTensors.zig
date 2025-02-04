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
    offset: u64 = 0,
    len: u64 = 0,
};

allocator: std.mem.Allocator,
file: std.fs.File,
header: std.StringArrayHashMap(TensorInfo),
byte_buffer_offset: u64 = 0,

const Self = @This();

pub fn open(allocator: std.mem.Allocator, path: []const u8) !Self {
    const file = try std.fs.cwd().openFile(path, .{ .mode = .read_only });
    return try openFile(allocator, file);
}

pub fn openFile(allocator: std.mem.Allocator, file: std.fs.File) !Self {
    const header_buf = try loadHeaderBuf(allocator, file);
    defer allocator.free(header_buf);
    const header = try parseHeader(allocator, header_buf);
    return Self{
        .allocator = allocator,
        .file = file,
        .header = header,
        .byte_buffer_offset = header_buf.len + 8,
    };
}

pub fn create(allocator: std.mem.Allocator, path: []const u8) !Self {
    const file = try std.fs.cwd().createFile(path, .{});
    return try createFile(allocator, file);
}

pub fn createFile(allocator: std.mem.Allocator, file: std.fs.File) !Self {
    return Self{
        .allocator = allocator,
        .file = file,
        .header = std.StringArrayHashMap(TensorInfo).init(allocator),
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

fn parseHeader(allocator: std.mem.Allocator, header_buf: []const u8) !std.StringArrayHashMap(TensorInfo) {
    // Parse the header into ordered hashmap
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
                    .offset = tinfo.data_offsets[0],
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

test "parse_header" {
    const t_allocator = std.testing.allocator;
    try std.testing.expectError(error.UnexpectedEndOfInput, parseHeader(t_allocator, ""));
    try std.testing.expectError(error.UnexpectedToken, parseHeader(t_allocator, "[]"));
    var hmap = try parseHeader(t_allocator, "{}");
    try std.testing.expectEqual(0, hmap.count());
    hmap = try parseHeader(t_allocator, "{\"__metadata__\": [1, 2, 3]}");
    try std.testing.expectEqual(0, hmap.count());
    try std.testing.expectError(error.UnexpectedToken, parseHeader(t_allocator, "{\"key\": [1, 2, 3]}"));
    try std.testing.expectError(error.MissingField, parseHeader(t_allocator, "{\"key\": {\"data_offsets\":[1, 2]}}"));
}

pub fn loadTensor(self: Self, name: []const u8) ![]align(8) u8 {
    // Lookup tensor in the header
    const tinfo = self.header.get(name) orelse return error.TensorNotFound;
    const buf = try self.allocator.alignedAlloc(u8, 8, tinfo.len);
    try self.file.seekTo(self.byte_buffer_offset + tinfo.offset);
    const read_len = try self.file.read(buf);
    if (read_len != tinfo.len)
        return error.UnableToLoadFullTensor;
    return buf;
}

fn computeTensorBufferLength(tinfo: TensorInfo) u64 {
    var tbuf_len: u64 = 1;
    for (tinfo.shape) |dim_shape| {
        tbuf_len *= dim_shape;
    }
    tbuf_len *= switch (tinfo.dtype) {
        .BOOL, .F8_E4M3, .F8_E5M2, .U8, .I8 => 1,
        .F16, .BF16, .U16, .I16 => 2,
        .F32, .U32, .I32 => 4,
        .F64, .U64, .I64 => 8,
    };
    return tbuf_len;
}

pub fn writeHeader(self: Self) !void {
    // Write initial length as 0, to be re-written later
    try self.file.seekTo(0);
    try self.file.writer().writeInt(u64, self.byte_buffer_offset, .little);

    // Write the actual header
    var writer = json.writeStream(self.file, .{});
    try writer.beginObject();
    var it = self.header.iterator();
    var tensor_offset = 0;
    while (it.next()) |entry| {
        // Write the tensor name as key
        try writer.objectField(entry.key_ptr.*);

        // Compute tensor buffer length and update in header map
        const tensor_len = computeTensorBufferLength(entry.value_ptr.*);
        entry.value_ptr.offset = tensor_offset;
        entry.value_ptr.len = tensor_len;

        // Write the entry
        const dtype = std.enums.tagName(DType, entry.value_ptr.dtype).?;
        try writer.write(.{
            .dtype = dtype,
            .shape = entry.value_ptr.shape,
            .data_offsets = [2]u64{
                tensor_offset,
                tensor_offset + tensor_len,
            },
        });

        // Update offset for next tensor
        tensor_offset += tensor_len;
    }

    // Re-write the correct header length
    self.byte_buffer_offset = try self.file.getPos();
    const header_len = self.byte_buffer_offset - 8;
    try self.file.seekTo(0);
    try self.file.writer().writeInt(u64, header_len, .little);
    try self.file.seekFromEnd(0);
}

pub fn writeTensor(self: Self, name: []const u8, data: []u8) !void {
    const tInfo = self.header.get(name).?;
    if (self.byte_buffer_offset == 0 or (tInfo.offset == 0 and tInfo.len == 0)) {
        return error.HeaderNotWritten;
    }
    try self.file.seekTo(self.byte_buffer_offset + tInfo.offset);
    const written_len = try self.file.write(data);
    const computed_len = computeTensorBufferLength(tInfo);
    std.debug.assert(written_len == computed_len);
}
