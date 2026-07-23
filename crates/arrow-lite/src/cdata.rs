//! The Arrow C Data Interface: how batches leave and enter the engine.
//!
//! This is the crate's unsafe core and its *entire* interop surface — no
//! IPC, no Flight, no Parquet (settled no). Three ABI structs cross the
//! boundary: [`ArrowSchema`] (type description), [`ArrowArray`] (data),
//! and [`ArrowArrayStream`] (a pull-based stream of batches, matching
//! segment-at-a-time execution with no final concatenation copy).
//!
//! ## Ownership across the boundary
//!
//! The C Data Interface transfers ownership through *release callbacks*: a
//! producer fills a struct and installs a `release` function; the consumer
//! calls it exactly once when done, and a released struct is marked by
//! `release == NULL`. On export, every node's `private_data` holds an
//! `Arc` of the whole batch (plus that node's pointer tables), so buffers
//! stay alive until the last callback runs, wherever the structs get
//! moved. On import we take the opposite stance: **import copies** into
//! our own aligned buffers and releases the foreign structs immediately —
//! zero-copy is the export story; imports are for tests, oracles, and
//! ingest, where owning the bytes is the point.
//!
//! ## What a mistake costs
//!
//! A wrong buffer count, format string, or callback here is silent data
//! corruption or a use-after-free in a consumer process. That is why the
//! acceptance gate for this module is the round-trip oracle harness
//! (issue #15) — arrow-rs and PyArrow importing our exports and vice
//! versa — not these unit tests alone.

use crate::bitmap::Bitmap;
use crate::buffer::{Buffer, Element, NumericColumn};
use crate::column::{Column, ColumnType, NumericData};
use crate::key::{Dictionary, KeyColumn};
use crate::logical::{LogicalType, DECIMAL64_PRECISION};
use crate::schema::{Field, RecordBatch, Schema};
use std::ffi::{c_char, c_void, CStr, CString};
use std::fmt;
use std::sync::Arc;

/// `ArrowSchema.flags` bit: the field may hold nulls.
const ARROW_FLAG_NULLABLE: i64 = 2;

/// Errno returned by stream callbacks on invalid input.
const EINVAL: i32 = 22;

/// ABI struct describing a type. Layout fixed by the Arrow C Data
/// Interface specification.
#[repr(C)]
#[derive(Debug)]
pub struct ArrowSchema {
    /// Format string (e.g. `"g"` for f64, `"+s"` for struct).
    pub format: *const c_char,
    /// Field name, or null.
    pub name: *const c_char,
    /// Binary metadata, or null (this crate emits none).
    pub metadata: *const c_char,
    /// Bit flags (`ARROW_FLAG_*`).
    pub flags: i64,
    /// Number of children.
    pub n_children: i64,
    /// Child type descriptions.
    pub children: *mut *mut ArrowSchema,
    /// Dictionary value type, for dictionary-encoded fields.
    pub dictionary: *mut ArrowSchema,
    /// Consumer calls exactly once; null marks a released struct.
    pub release: Option<unsafe extern "C" fn(*mut ArrowSchema)>,
    /// Producer-owned state for `release`.
    pub private_data: *mut c_void,
}

/// ABI struct carrying data. Layout fixed by the specification.
#[repr(C)]
#[derive(Debug)]
pub struct ArrowArray {
    /// Logical row count.
    pub length: i64,
    /// Null count, or -1 for unknown.
    pub null_count: i64,
    /// Logical offset into the buffers, in rows.
    pub offset: i64,
    /// Number of entries in `buffers`.
    pub n_buffers: i64,
    /// Number of children.
    pub n_children: i64,
    /// Buffer pointers; individual entries may be null.
    pub buffers: *mut *const c_void,
    /// Child arrays.
    pub children: *mut *mut ArrowArray,
    /// Dictionary values, for dictionary-encoded fields.
    pub dictionary: *mut ArrowArray,
    /// Consumer calls exactly once; null marks a released struct.
    pub release: Option<unsafe extern "C" fn(*mut ArrowArray)>,
    /// Producer-owned state for `release`.
    pub private_data: *mut c_void,
}

/// ABI struct for a pull-based stream of batches sharing one schema.
#[repr(C)]
#[derive(Debug)]
pub struct ArrowArrayStream {
    /// Writes the stream's schema into `out`; returns 0 or an errno.
    pub get_schema: Option<unsafe extern "C" fn(*mut ArrowArrayStream, *mut ArrowSchema) -> i32>,
    /// Writes the next batch into `out` (release == null means end of
    /// stream); returns 0 or an errno.
    pub get_next: Option<unsafe extern "C" fn(*mut ArrowArrayStream, *mut ArrowArray) -> i32>,
    /// Describes the last error, valid until the next call.
    pub get_last_error: Option<unsafe extern "C" fn(*mut ArrowArrayStream) -> *const c_char>,
    /// Consumer calls exactly once; null marks a released struct.
    pub release: Option<unsafe extern "C" fn(*mut ArrowArrayStream)>,
    /// Producer-owned state.
    pub private_data: *mut c_void,
}

impl ArrowSchema {
    /// A zeroed struct, for out-parameters. `release == None` means "not
    /// a live export".
    pub fn empty() -> Self {
        ArrowSchema {
            format: std::ptr::null(),
            name: std::ptr::null(),
            metadata: std::ptr::null(),
            flags: 0,
            n_children: 0,
            children: std::ptr::null_mut(),
            dictionary: std::ptr::null_mut(),
            release: None,
            private_data: std::ptr::null_mut(),
        }
    }
}

impl ArrowArray {
    /// A zeroed struct, for out-parameters and end-of-stream markers.
    pub fn empty() -> Self {
        ArrowArray {
            length: 0,
            null_count: 0,
            offset: 0,
            n_buffers: 0,
            n_children: 0,
            buffers: std::ptr::null_mut(),
            children: std::ptr::null_mut(),
            dictionary: std::ptr::null_mut(),
            release: None,
            private_data: std::ptr::null_mut(),
        }
    }
}

impl ArrowArrayStream {
    /// A zeroed struct, for out-parameters.
    pub fn empty() -> Self {
        ArrowArrayStream {
            get_schema: None,
            get_next: None,
            get_last_error: None,
            release: None,
            private_data: std::ptr::null_mut(),
        }
    }
}

/// A failure while importing foreign C Data structs.
///
/// Import errors are `Result`s, not panics: the data crossed a process
/// boundary and its shape is not ours to promise.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportError(String);

impl ImportError {
    fn new(message: impl Into<String>) -> Self {
        ImportError(message.into())
    }
}

impl fmt::Display for ImportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "arrow import: {}", self.0)
    }
}

impl std::error::Error for ImportError {}

// ---------------------------------------------------------------------
// Export: schema
// ---------------------------------------------------------------------

/// Heap state a live exported [`ArrowSchema`] owns; freed by its release
/// callback.
struct SchemaPrivate {
    _format: CString,
    _name: CString,
    children_ptrs: Vec<*mut ArrowSchema>,
}

/// Builds one exported schema node; children and dictionary move onto the
/// heap and are freed by [`release_schema`].
fn new_schema_node(
    format: String,
    name: &str,
    flags: i64,
    children: Vec<ArrowSchema>,
    dictionary: Option<ArrowSchema>,
) -> ArrowSchema {
    let format = CString::new(format).expect("format has no NUL");
    let name = CString::new(name).expect("name has no NUL");
    let children_ptrs: Vec<*mut ArrowSchema> = children
        .into_iter()
        .map(|c| Box::into_raw(Box::new(c)))
        .collect();
    let dictionary = match dictionary {
        Some(d) => Box::into_raw(Box::new(d)),
        None => std::ptr::null_mut(),
    };
    let mut private = Box::new(SchemaPrivate {
        _format: format,
        _name: name,
        children_ptrs,
    });
    ArrowSchema {
        format: private._format.as_ptr(),
        name: private._name.as_ptr(),
        metadata: std::ptr::null(),
        flags,
        n_children: private.children_ptrs.len() as i64,
        // Vec heap storage is address-stable while the private box holds
        // the Vec, wherever the box or this struct move.
        children: if private.children_ptrs.is_empty() {
            std::ptr::null_mut()
        } else {
            private.children_ptrs.as_mut_ptr()
        },
        dictionary,
        release: Some(release_schema),
        private_data: Box::into_raw(private).cast(),
    }
}

/// The release callback installed on every exported schema node.
///
/// # Safety
/// Called by the consumer, once, on a node this module exported (the C
/// Data Interface contract).
unsafe extern "C" fn release_schema(schema: *mut ArrowSchema) {
    if schema.is_null() {
        return;
    }
    let node = unsafe { &mut *schema };
    if node.release.is_none() {
        return; // already released
    }
    // SAFETY: private_data was Box::into_raw of a SchemaPrivate in
    // new_schema_node; children/dictionary pointers were Box::into_raw of
    // nodes this module built, each released exactly once here.
    unsafe {
        let private = Box::from_raw(node.private_data as *mut SchemaPrivate);
        for &child in &private.children_ptrs {
            if let Some(release) = (*child).release {
                release(child);
            }
            drop(Box::from_raw(child));
        }
        if !node.dictionary.is_null() {
            if let Some(release) = (*node.dictionary).release {
                release(node.dictionary);
            }
            drop(Box::from_raw(node.dictionary));
        }
        drop(private);
    }
    node.release = None;
    node.private_data = std::ptr::null_mut();
}

/// The C-Data format string for one field.
fn field_format(field: &Field) -> String {
    match field.column_type() {
        ColumnType::F64 => "g".to_owned(),
        ColumnType::I64 => match field.logical() {
            Some(logical) => logical.c_data_format(),
            None => "l".to_owned(),
        },
        // Dictionary-encoded: the field's own format is the *index* type
        // (u32); the value type rides on `dictionary`.
        ColumnType::Key => "I".to_owned(),
    }
}

/// Exports a schema as a struct-typed root whose children are the fields.
pub fn export_schema(schema: &Schema) -> ArrowSchema {
    let children = schema
        .fields()
        .iter()
        .map(|field| {
            let flags = if field.nullable() {
                ARROW_FLAG_NULLABLE
            } else {
                0
            };
            let dictionary = match field.column_type() {
                ColumnType::Key => Some(new_schema_node("u".to_owned(), "", 0, vec![], None)),
                _ => None,
            };
            new_schema_node(field_format(field), field.name(), flags, vec![], dictionary)
        })
        .collect();
    new_schema_node("+s".to_owned(), "", 0, children, None)
}

// ---------------------------------------------------------------------
// Export: arrays
// ---------------------------------------------------------------------

/// Heap state a live exported [`ArrowArray`] owns; freed by its release
/// callback. The `Arc` is what keeps every buffer pointer valid until the
/// last node is released.
struct ArrayPrivate {
    _keep_alive: Arc<RecordBatch>,
    buffers: Vec<*const c_void>,
    children_ptrs: Vec<*mut ArrowArray>,
}

/// Builds one exported array node over buffers owned (transitively) by
/// `keep_alive`.
fn new_array_node(
    length: usize,
    null_count: usize,
    buffers: Vec<*const c_void>,
    children: Vec<ArrowArray>,
    dictionary: Option<ArrowArray>,
    keep_alive: Arc<RecordBatch>,
) -> ArrowArray {
    let children_ptrs: Vec<*mut ArrowArray> = children
        .into_iter()
        .map(|c| Box::into_raw(Box::new(c)))
        .collect();
    let dictionary = match dictionary {
        Some(d) => Box::into_raw(Box::new(d)),
        None => std::ptr::null_mut(),
    };
    let mut private = Box::new(ArrayPrivate {
        _keep_alive: keep_alive,
        buffers,
        children_ptrs,
    });
    ArrowArray {
        length: length as i64,
        null_count: null_count as i64,
        offset: 0,
        n_buffers: private.buffers.len() as i64,
        n_children: private.children_ptrs.len() as i64,
        buffers: if private.buffers.is_empty() {
            std::ptr::null_mut()
        } else {
            private.buffers.as_mut_ptr()
        },
        children: if private.children_ptrs.is_empty() {
            std::ptr::null_mut()
        } else {
            private.children_ptrs.as_mut_ptr()
        },
        dictionary,
        release: Some(release_array),
        private_data: Box::into_raw(private).cast(),
    }
}

/// The release callback installed on every exported array node.
///
/// # Safety
/// Called by the consumer, once, on a node this module exported.
unsafe extern "C" fn release_array(array: *mut ArrowArray) {
    if array.is_null() {
        return;
    }
    let node = unsafe { &mut *array };
    if node.release.is_none() {
        return; // already released
    }
    // SAFETY: mirrors release_schema — every raw pointer here came from
    // Box::into_raw in new_array_node and is freed exactly once.
    unsafe {
        let private = Box::from_raw(node.private_data as *mut ArrayPrivate);
        for &child in &private.children_ptrs {
            if let Some(release) = (*child).release {
                release(child);
            }
            drop(Box::from_raw(child));
        }
        if !node.dictionary.is_null() {
            if let Some(release) = (*node.dictionary).release {
                release(node.dictionary);
            }
            drop(Box::from_raw(node.dictionary));
        }
        drop(private);
    }
    node.release = None;
    node.private_data = std::ptr::null_mut();
}

/// The validity buffer pointer for an optional bitmap: null when absent.
fn validity_ptr(validity: Option<&Bitmap>) -> *const c_void {
    match validity {
        Some(bitmap) => bitmap.as_bytes().as_ptr().cast(),
        None => std::ptr::null(),
    }
}

/// Builds the exported node for one column.
fn column_node(column: &Column, keep_alive: Arc<RecordBatch>) -> ArrowArray {
    match column {
        Column::Numeric(NumericData::F64(col)) => numeric_node(col, keep_alive),
        Column::Numeric(NumericData::I64(col)) => numeric_node(col, keep_alive),
        Column::Key(col) => key_node(col, keep_alive),
    }
}

fn numeric_node<T: Element>(col: &NumericColumn<T>, keep_alive: Arc<RecordBatch>) -> ArrowArray {
    new_array_node(
        col.len(),
        col.null_count(),
        vec![validity_ptr(col.validity()), col.values().as_ptr().cast()],
        vec![],
        None,
        keep_alive,
    )
}

fn key_node(col: &KeyColumn, keep_alive: Arc<RecordBatch>) -> ArrowArray {
    let dict = col.dictionary();
    // The dictionary values ride as a Utf8 array: validity (none — values
    // are never null), i32 offsets, data bytes.
    let values = new_array_node(
        dict.len(),
        0,
        vec![
            std::ptr::null(),
            dict.offsets().as_ptr().cast(),
            dict.bytes().as_ptr().cast(),
        ],
        vec![],
        None,
        keep_alive.clone(),
    );
    new_array_node(
        col.len(),
        col.null_count(),
        vec![validity_ptr(col.validity()), col.codes().as_ptr().cast()],
        vec![],
        Some(values),
        keep_alive,
    )
}

/// Exports a batch's data as a struct-typed root. The batch moves into
/// shared ownership held by the exported nodes' release callbacks.
pub fn export_array(batch: RecordBatch) -> ArrowArray {
    let keep = Arc::new(batch);
    let children = keep
        .columns()
        .iter()
        .map(|column| column_node(column, keep.clone()))
        .collect();
    new_array_node(
        keep.num_rows(),
        0,
        vec![std::ptr::null()],
        children,
        None,
        keep,
    )
}

/// Exports a batch as the (schema, array) struct pair consumers like
/// PyArrow's `_import_from_c` expect.
pub fn export_batch(batch: RecordBatch) -> (ArrowSchema, ArrowArray) {
    (export_schema(batch.schema()), export_array(batch))
}

// ---------------------------------------------------------------------
// Export: stream
// ---------------------------------------------------------------------

/// Producer state behind an exported [`ArrowArrayStream`].
struct StreamPrivate {
    schema: Schema,
    batches: Box<dyn Iterator<Item = RecordBatch> + Send>,
    last_error: Option<CString>,
}

/// Exports a stream of batches sharing `schema`.
///
/// Each `get_next` pulls one batch from `batches` and hands it over with
/// full ownership transfer — no batch is ever concatenated or copied. A
/// batch whose schema disagrees with `schema` fails that `get_next` with
/// `EINVAL` (visible via `get_last_error`).
pub fn export_stream(
    schema: Schema,
    batches: impl Iterator<Item = RecordBatch> + Send + 'static,
) -> ArrowArrayStream {
    let private = Box::new(StreamPrivate {
        schema,
        batches: Box::new(batches),
        last_error: None,
    });
    ArrowArrayStream {
        get_schema: Some(stream_get_schema),
        get_next: Some(stream_get_next),
        get_last_error: Some(stream_get_last_error),
        release: Some(release_stream),
        private_data: Box::into_raw(private).cast(),
    }
}

/// # Safety
/// Called by the consumer on a stream this module exported.
unsafe extern "C" fn stream_get_schema(
    stream: *mut ArrowArrayStream,
    out: *mut ArrowSchema,
) -> i32 {
    // SAFETY: private_data is the StreamPrivate installed by
    // export_stream; out is a consumer-provided struct to overwrite.
    unsafe {
        let private = &mut *((*stream).private_data as *mut StreamPrivate);
        out.write(export_schema(&private.schema));
    }
    0
}

/// # Safety
/// Called by the consumer on a stream this module exported.
unsafe extern "C" fn stream_get_next(stream: *mut ArrowArrayStream, out: *mut ArrowArray) -> i32 {
    // SAFETY: as for stream_get_schema.
    unsafe {
        let private = &mut *((*stream).private_data as *mut StreamPrivate);
        match private.batches.next() {
            Some(batch) if batch.schema() == &private.schema => {
                out.write(export_array(batch));
                0
            }
            Some(_) => {
                private.last_error =
                    Some(CString::new("batch schema differs from stream schema").expect("no NUL"));
                out.write(ArrowArray::empty());
                EINVAL
            }
            // End of stream: release == null.
            None => {
                out.write(ArrowArray::empty());
                0
            }
        }
    }
}

/// # Safety
/// Called by the consumer on a stream this module exported.
unsafe extern "C" fn stream_get_last_error(stream: *mut ArrowArrayStream) -> *const c_char {
    // SAFETY: as for stream_get_schema.
    unsafe {
        let private = &mut *((*stream).private_data as *mut StreamPrivate);
        match &private.last_error {
            Some(message) => message.as_ptr(),
            None => std::ptr::null(),
        }
    }
}

/// # Safety
/// Called by the consumer, once.
unsafe extern "C" fn release_stream(stream: *mut ArrowArrayStream) {
    if stream.is_null() {
        return;
    }
    let node = unsafe { &mut *stream };
    if node.release.is_none() {
        return;
    }
    // SAFETY: private_data was Box::into_raw of StreamPrivate.
    unsafe { drop(Box::from_raw(node.private_data as *mut StreamPrivate)) };
    node.release = None;
    node.private_data = std::ptr::null_mut();
}

// ---------------------------------------------------------------------
// Import
// ---------------------------------------------------------------------

/// Reads bit `index` of an LSB-ordered validity buffer.
unsafe fn validity_bit(bits: *const u8, index: usize) -> bool {
    // SAFETY: caller ensures index is within the buffer's bit range.
    unsafe { *bits.add(index / 8) & (1 << (index % 8)) != 0 }
}

/// Parses a foreign schema into ours, borrowing only — the caller still
/// owns (and must release) the foreign struct.
///
/// # Safety
/// `schema` must be a valid, unreleased `ArrowSchema` per the C Data
/// Interface contract.
unsafe fn parse_schema(schema: &ArrowSchema) -> Result<Schema, ImportError> {
    if schema.release.is_none() {
        return Err(ImportError::new("schema struct already released"));
    }
    // SAFETY: format on a live schema is a valid NUL-terminated string.
    let format = unsafe { cstr(schema.format) }?;
    if format != "+s" {
        return Err(ImportError::new(format!(
            "expected struct root '+s', got '{format}'"
        )));
    }
    let n =
        usize::try_from(schema.n_children).map_err(|_| ImportError::new("negative child count"))?;
    let mut fields = Vec::with_capacity(n);
    for i in 0..n {
        // SAFETY: a live schema's children array has n_children valid
        // pointers.
        let child = unsafe { &**schema.children.add(i) };
        fields.push(unsafe { parse_field(child) }?);
    }
    Ok(Schema::new(fields))
}

/// # Safety
/// `child` must be a valid, unreleased field-level `ArrowSchema`.
unsafe fn parse_field(child: &ArrowSchema) -> Result<Field, ImportError> {
    // SAFETY: live schema; name may be null.
    let name = if child.name.is_null() {
        String::new()
    } else {
        unsafe { cstr(child.name) }?.to_owned()
    };
    let format = unsafe { cstr(child.format) }?;
    let nullable = child.flags & ARROW_FLAG_NULLABLE != 0;
    let (column_type, logical) = if child.dictionary.is_null() {
        match format {
            "g" => (ColumnType::F64, None),
            "l" => (ColumnType::I64, None),
            "tsn:" => (ColumnType::I64, Some(LogicalType::TimestampNs)),
            decimal if decimal.starts_with("d:") => {
                (ColumnType::I64, Some(parse_decimal(decimal)?))
            }
            other => {
                return Err(ImportError::new(format!(
                    "unsupported format '{other}' for field '{name}'"
                )))
            }
        }
    } else {
        if format != "I" {
            return Err(ImportError::new(format!(
                "unsupported dictionary index format '{format}' (u32 'I' only)"
            )));
        }
        // SAFETY: non-null dictionary on a live schema is a valid schema.
        let values_format = unsafe { cstr((*child.dictionary).format) }?;
        if values_format != "u" {
            return Err(ImportError::new(format!(
                "unsupported dictionary value format '{values_format}' (utf8 'u' only)"
            )));
        }
        (ColumnType::Key, None)
    };
    let mut field = Field::new(name, column_type, nullable);
    if let Some(logical) = logical {
        field = field.with_logical(logical);
    }
    Ok(field)
}

/// Parses `d:precision,scale,bitwidth`; only the exact shape this crate
/// exports (`d:18,s,64`) is accepted.
fn parse_decimal(format: &str) -> Result<LogicalType, ImportError> {
    let parts: Vec<&str> = format[2..].split(',').collect();
    let reject = || ImportError::new(format!("unsupported decimal format '{format}'"));
    let [precision, scale, bits] = parts.as_slice() else {
        return Err(reject());
    };
    if *bits != "64" || precision.parse::<u8>().ok() != Some(DECIMAL64_PRECISION) {
        return Err(reject());
    }
    let scale: u8 = scale.parse().map_err(|_| reject())?;
    LogicalType::from_parts(2, scale).ok_or_else(reject)
}

/// # Safety
/// `ptr` must be null or a valid NUL-terminated string.
unsafe fn cstr<'a>(ptr: *const c_char) -> Result<&'a str, ImportError> {
    if ptr.is_null() {
        return Err(ImportError::new("unexpected null string"));
    }
    // SAFETY: caller guarantees NUL termination.
    unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .map_err(|_| ImportError::new("string is not UTF-8"))
}

/// Imports a foreign (schema, array) pair into an owned [`RecordBatch`],
/// copying every buffer, and releases both structs.
///
/// # Safety
/// Both structs must be valid, unreleased C Data Interface exports whose
/// buffers match their declared layout — the core contract of the
/// interface, which cannot be checked from this side.
pub unsafe fn import_batch(
    mut schema: ArrowSchema,
    mut array: ArrowArray,
) -> Result<RecordBatch, ImportError> {
    // SAFETY: caller guarantees a live schema.
    let parsed = unsafe { parse_schema(&schema) };
    // SAFETY: releasing a live struct exactly once.
    unsafe {
        if let Some(release) = schema.release {
            release(&mut schema);
        }
    }
    let parsed = match parsed {
        Ok(parsed) => parsed,
        Err(error) => {
            // SAFETY: as above.
            unsafe {
                if let Some(release) = array.release {
                    release(&mut array);
                }
            }
            return Err(error);
        }
    };
    // SAFETY: caller guarantees a live array.
    unsafe { import_array(&parsed, array) }
}

/// Imports a foreign array against an already-parsed schema, copying, and
/// releases the struct.
///
/// # Safety
/// `array` must be a valid, unreleased C Data Interface export matching
/// `schema`.
pub unsafe fn import_array(
    schema: &Schema,
    mut array: ArrowArray,
) -> Result<RecordBatch, ImportError> {
    // SAFETY: caller guarantees a live array; inner only borrows.
    let result = unsafe { import_array_inner(schema, &array) };
    // SAFETY: releasing a live struct exactly once.
    unsafe {
        if let Some(release) = array.release {
            release(&mut array);
        }
    }
    result
}

/// # Safety
/// As for [`import_array`]; borrows only.
unsafe fn import_array_inner(
    schema: &Schema,
    array: &ArrowArray,
) -> Result<RecordBatch, ImportError> {
    if array.release.is_none() {
        return Err(ImportError::new("array struct already released"));
    }
    let n =
        usize::try_from(array.n_children).map_err(|_| ImportError::new("negative child count"))?;
    if n != schema.fields().len() {
        return Err(ImportError::new(format!(
            "schema has {} fields but array has {n} children",
            schema.fields().len()
        )));
    }
    let num_rows =
        usize::try_from(array.length).map_err(|_| ImportError::new("negative length"))?;
    let root_offset =
        usize::try_from(array.offset).map_err(|_| ImportError::new("negative offset"))?;
    if array.null_count > 0 {
        return Err(ImportError::new("struct-level nulls are unsupported"));
    }
    let mut columns = Vec::with_capacity(n);
    for (i, field) in schema.fields().iter().enumerate() {
        // SAFETY: a live array's children array has n_children valid
        // pointers.
        let child = unsafe { &**array.children.add(i) };
        let column = unsafe { import_column(field, child, root_offset, num_rows) }?;
        columns.push(column);
    }
    Ok(RecordBatch::new(schema.clone(), columns))
}

/// Everything shared by the per-type importers: bounds-checked geometry
/// and the validity bitmap.
struct ChildGeometry {
    /// Physical index of the first row to read.
    start: usize,
    /// Rows to read.
    rows: usize,
    /// Rebuilt bitmap, already offset-corrected; `None` for no nulls.
    validity: Option<Bitmap>,
}

/// # Safety
/// `child` must be a valid, unreleased array node with the buffer layout
/// its parent's schema declared.
unsafe fn child_geometry(
    field: &Field,
    child: &ArrowArray,
    root_offset: usize,
    num_rows: usize,
    validity_buffer: *const c_void,
) -> Result<ChildGeometry, ImportError> {
    if child.release.is_none() {
        return Err(ImportError::new("child array already released"));
    }
    let length =
        usize::try_from(child.length).map_err(|_| ImportError::new("negative child length"))?;
    let offset =
        usize::try_from(child.offset).map_err(|_| ImportError::new("negative child offset"))?;
    // The parent's offset applies to children: logical row i lives at
    // physical child index child.offset + root.offset + i.
    if root_offset + num_rows > length {
        return Err(ImportError::new(format!(
            "column '{}': child length {length} < parent offset {root_offset} + rows {num_rows}",
            field.name()
        )));
    }
    let start = offset + root_offset;
    let validity = if validity_buffer.is_null() {
        None
    } else {
        let bits = validity_buffer.cast::<u8>();
        // SAFETY: a non-null validity buffer covers the array's physical
        // rows; we read bits start..start + num_rows.
        let bitmap =
            Bitmap::from_bools((0..num_rows).map(|i| unsafe { validity_bit(bits, start + i) }));
        if bitmap.count_set() == num_rows {
            None // no actual nulls: drop the bitmap (NOT NULL-clean)
        } else {
            Some(bitmap)
        }
    };
    if validity.is_some() && !field.nullable() {
        return Err(ImportError::new(format!(
            "column '{}' declared NOT NULL but has nulls",
            field.name()
        )));
    }
    Ok(ChildGeometry {
        start,
        rows: num_rows,
        validity,
    })
}

/// Fetches buffer `index`, requiring `child.n_buffers > index`.
unsafe fn buffer_at(
    child: &ArrowArray,
    index: usize,
    what: &str,
) -> Result<*const c_void, ImportError> {
    let n = usize::try_from(child.n_buffers).unwrap_or(0);
    if index >= n {
        return Err(ImportError::new(format!(
            "expected {what} at buffer {index}, but array has {n} buffers"
        )));
    }
    // SAFETY: a live array's buffers array has n_buffers entries.
    Ok(unsafe { *child.buffers.add(index) })
}

/// # Safety
/// As for [`child_geometry`].
unsafe fn import_column(
    field: &Field,
    child: &ArrowArray,
    root_offset: usize,
    num_rows: usize,
) -> Result<Column, ImportError> {
    // Every supported layout puts validity at buffer 0.
    let validity_buffer = unsafe { buffer_at(child, 0, "validity") }?;
    match field.column_type() {
        ColumnType::F64 => {
            let geo =
                unsafe { child_geometry(field, child, root_offset, num_rows, validity_buffer) }?;
            let values = unsafe { copy_values::<f64>(child, &geo) }?;
            Ok(Column::Numeric(NumericData::F64(make_numeric(values, geo))))
        }
        ColumnType::I64 => {
            let geo =
                unsafe { child_geometry(field, child, root_offset, num_rows, validity_buffer) }?;
            let values = unsafe { copy_values::<i64>(child, &geo) }?;
            Ok(Column::Numeric(NumericData::I64(make_numeric(values, geo))))
        }
        ColumnType::Key => unsafe { import_key_column(field, child, root_offset, num_rows) },
    }
}

fn make_numeric<T: Element>(values: Buffer<T>, geo: ChildGeometry) -> NumericColumn<T> {
    match geo.validity {
        Some(bitmap) => NumericColumn::new_nullable(values, bitmap),
        None => NumericColumn::new_non_null(values),
    }
}

/// Copies `geo.rows` elements from the data buffer (buffer 1).
///
/// # Safety
/// As for [`child_geometry`].
unsafe fn copy_values<T: Element>(
    child: &ArrowArray,
    geo: &ChildGeometry,
) -> Result<Buffer<T>, ImportError> {
    let data = unsafe { buffer_at(child, 1, "data") }?;
    if data.is_null() && geo.rows > 0 {
        return Err(ImportError::new("null data buffer"));
    }
    let mut buffer = Buffer::with_capacity(geo.rows);
    if geo.rows > 0 {
        // SAFETY: the data buffer covers the physical rows the geometry
        // bounds-checked; extend_from_raw copies byte-wise (no alignment
        // assumption on the foreign buffer).
        unsafe { buffer.extend_from_raw(data.cast::<T>().add(geo.start), geo.rows) };
    }
    Ok(buffer)
}

/// # Safety
/// As for [`child_geometry`]; `child.dictionary` must be a valid Utf8
/// values array.
unsafe fn import_key_column(
    field: &Field,
    child: &ArrowArray,
    root_offset: usize,
    num_rows: usize,
) -> Result<Column, ImportError> {
    let validity_buffer = unsafe { buffer_at(child, 0, "validity") }?;
    let geo = unsafe { child_geometry(field, child, root_offset, num_rows, validity_buffer) }?;
    let codes = unsafe { copy_values::<u32>(child, &geo) }?;
    if child.dictionary.is_null() {
        return Err(ImportError::new(format!(
            "key column '{}' has no dictionary array",
            field.name()
        )));
    }
    // SAFETY: non-null dictionary on a live array is a valid array node.
    let dictionary = unsafe { import_dictionary_values(&*child.dictionary) }?;
    // Codes under valid slots must be in range; a code under a null slot
    // is unspecified by Arrow, so sanitize it to 0.
    let mut codes = codes;
    for row in 0..codes.len() {
        let code = codes.as_slice()[row] as usize;
        let valid = geo.validity.as_ref().is_none_or(|bm| bm.get(row));
        if valid {
            if code >= dictionary.len() {
                return Err(ImportError::new(format!(
                    "key column '{}': code {code} out of dictionary range {}",
                    field.name(),
                    dictionary.len()
                )));
            }
        } else if code >= dictionary.len() {
            if dictionary.is_empty() {
                return Err(ImportError::new(format!(
                    "key column '{}': null rows but empty dictionary",
                    field.name()
                )));
            }
            codes.as_mut_slice()[row] = 0;
        }
    }
    let column = match geo.validity {
        Some(bitmap) => KeyColumn::new_nullable(codes, bitmap, dictionary),
        None => KeyColumn::new_non_null(codes, dictionary),
    };
    Ok(Column::Key(column))
}

/// Rebuilds a [`Dictionary`] from a foreign Utf8 values array.
///
/// # Safety
/// `values` must be a valid, unreleased Utf8 array node.
unsafe fn import_dictionary_values(values: &ArrowArray) -> Result<Dictionary, ImportError> {
    if values.release.is_none() {
        return Err(ImportError::new("dictionary array already released"));
    }
    let count = usize::try_from(values.length)
        .map_err(|_| ImportError::new("negative dictionary length"))?;
    let offset = usize::try_from(values.offset)
        .map_err(|_| ImportError::new("negative dictionary offset"))?;
    if values.null_count > 0 {
        return Err(ImportError::new("null dictionary values are unsupported"));
    }
    let offsets = unsafe { buffer_at(values, 1, "utf8 offsets") }?;
    let data = unsafe { buffer_at(values, 2, "utf8 data") }?;
    if count > 0 && (offsets.is_null() || data.is_null()) {
        return Err(ImportError::new("null utf8 buffer"));
    }
    let mut dictionary = Dictionary::new();
    for i in 0..count {
        // SAFETY: the offsets buffer has count + 1 entries starting at
        // the values offset; entries may be unaligned in principle, so
        // read unaligned.
        let (start, end) = unsafe {
            let base = offsets.cast::<i32>();
            (
                base.add(offset + i).read_unaligned(),
                base.add(offset + i + 1).read_unaligned(),
            )
        };
        let (start, end) = (
            usize::try_from(start).map_err(|_| ImportError::new("negative utf8 offset"))?,
            usize::try_from(end).map_err(|_| ImportError::new("negative utf8 offset"))?,
        );
        if end < start {
            return Err(ImportError::new("utf8 offsets not monotone"));
        }
        // SAFETY: the data buffer covers start..end per the offsets the
        // producer declared.
        let bytes =
            unsafe { std::slice::from_raw_parts(data.cast::<u8>().add(start), end - start) };
        let value = std::str::from_utf8(bytes)
            .map_err(|_| ImportError::new("dictionary value is not UTF-8"))?;
        let code = dictionary.intern(value);
        if code as usize != i {
            return Err(ImportError::new(
                "duplicate dictionary values are unsupported",
            ));
        }
    }
    Ok(dictionary)
}

// ---------------------------------------------------------------------
// Import: stream
// ---------------------------------------------------------------------

/// An iterator over a foreign [`ArrowArrayStream`]'s batches; releases
/// the stream on drop.
pub struct StreamReader {
    stream: ArrowArrayStream,
    schema: Schema,
}

impl StreamReader {
    /// Takes ownership of a foreign stream and reads its schema.
    ///
    /// # Safety
    /// `stream` must be a valid, unreleased C Data Interface stream
    /// export.
    pub unsafe fn new(mut stream: ArrowArrayStream) -> Result<StreamReader, ImportError> {
        let release_on_error = |stream: &mut ArrowArrayStream| {
            // SAFETY: releasing the live stream exactly once on the error
            // path (success hands ownership to the reader).
            unsafe {
                if let Some(release) = stream.release {
                    release(stream);
                }
            }
        };
        let Some(get_schema) = stream.get_schema else {
            release_on_error(&mut stream);
            return Err(ImportError::new("stream has no get_schema"));
        };
        if stream.get_next.is_none() {
            release_on_error(&mut stream);
            return Err(ImportError::new("stream has no get_next"));
        }
        let mut ffi_schema = ArrowSchema::empty();
        // SAFETY: calling a live stream's callback per the contract.
        let code = unsafe { get_schema(&mut stream, &mut ffi_schema) };
        if code != 0 {
            let error = ImportError::new(format!(
                "get_schema failed ({code}): {}",
                // SAFETY: last-error lookup on the live stream.
                unsafe { last_error(&mut stream) }
            ));
            release_on_error(&mut stream);
            return Err(error);
        }
        // SAFETY: get_schema succeeded, so ffi_schema is live.
        let parsed = unsafe { parse_schema(&ffi_schema) };
        // SAFETY: releasing the schema struct we own, exactly once.
        unsafe {
            if let Some(release) = ffi_schema.release {
                release(&mut ffi_schema);
            }
        }
        match parsed {
            Ok(schema) => Ok(StreamReader { stream, schema }),
            Err(error) => {
                release_on_error(&mut stream);
                Err(error)
            }
        }
    }

    /// The stream's schema.
    pub fn schema(&self) -> &Schema {
        &self.schema
    }
}

/// # Safety
/// `stream` must be live.
unsafe fn last_error(stream: &mut ArrowArrayStream) -> String {
    let Some(get_last_error) = stream.get_last_error else {
        return "(no error detail)".to_owned();
    };
    // SAFETY: calling the live stream's callback.
    let ptr = unsafe { get_last_error(stream) };
    if ptr.is_null() {
        return "(no error detail)".to_owned();
    }
    // SAFETY: get_last_error returns a NUL-terminated string.
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

impl Iterator for StreamReader {
    type Item = Result<RecordBatch, ImportError>;

    fn next(&mut self) -> Option<Self::Item> {
        let get_next = self.stream.get_next.expect("checked at construction");
        let mut ffi_array = ArrowArray::empty();
        // SAFETY: the reader owns a live stream (checked at
        // construction, released only on drop).
        let code = unsafe { get_next(&mut self.stream, &mut ffi_array) };
        if code != 0 {
            return Some(Err(ImportError::new(format!(
                "get_next failed ({code}): {}",
                // SAFETY: last-error lookup on the live stream.
                unsafe { last_error(&mut self.stream) }
            ))));
        }
        // End of stream is marked by release == null.
        ffi_array.release?;
        // SAFETY: a live array we now own.
        Some(unsafe { import_array(&self.schema, ffi_array) })
    }
}

impl Drop for StreamReader {
    fn drop(&mut self) {
        // SAFETY: the reader owns the live stream; releasing exactly
        // once.
        unsafe {
            if let Some(release) = self.stream.release {
                release(&mut self.stream);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::Buffer;

    fn test_batch() -> RecordBatch {
        let schema = Schema::new(vec![
            Field::new("ts", ColumnType::I64, false).with_logical(LogicalType::TimestampNs),
            Field::new("sym", ColumnType::Key, false),
            Field::new("px", ColumnType::F64, true),
            Field::new("qty", ColumnType::I64, false),
        ]);
        RecordBatch::new(
            schema,
            vec![
                Column::Numeric(NumericData::I64(NumericColumn::new_non_null(
                    Buffer::from_slice(&[1_000, 2_000, 3_000]),
                ))),
                Column::Key(KeyColumn::from_values(["AAPL", "MSFT", "AAPL"])),
                Column::Numeric(NumericData::F64(NumericColumn::new_nullable(
                    Buffer::from_slice(&[101.5, 0.0, 99.25]),
                    Bitmap::from_bools([true, false, true]),
                ))),
                Column::Numeric(NumericData::I64(NumericColumn::new_non_null(
                    Buffer::from_slice(&[10, 20, 30]),
                ))),
            ],
        )
    }

    #[test]
    fn self_round_trip_preserves_batch() {
        let batch = test_batch();
        let expected = batch.clone();
        let (schema, array) = export_batch(batch);
        let imported = unsafe { import_batch(schema, array) }.expect("round trip");
        assert_eq!(imported, expected);
    }

    #[test]
    fn export_is_zero_copy() {
        let batch = test_batch();
        let Column::Numeric(NumericData::F64(px)) = &batch.columns()[2] else {
            unreachable!()
        };
        let px_ptr = px.values().as_ptr().cast::<c_void>();
        let (mut schema, mut array) = export_batch(batch);
        // The exported px data buffer IS the column's buffer.
        let px_child = unsafe { &**array.children.add(2) };
        let data_ptr = unsafe { *px_child.buffers.add(1) };
        assert_eq!(data_ptr, px_ptr);
        unsafe {
            release_schema(&mut schema);
            release_array(&mut array);
        }
    }

    #[test]
    fn release_is_idempotent() {
        let (mut schema, mut array) = export_batch(test_batch());
        unsafe {
            release_array(&mut array);
            release_array(&mut array); // second call must be a no-op
            release_schema(&mut schema);
            release_schema(&mut schema);
        }
        assert!(schema.release.is_none());
        assert!(array.release.is_none());
    }

    #[test]
    fn empty_batch_round_trips() {
        let schema = Schema::new(vec![
            Field::new("x", ColumnType::F64, false),
            Field::new("k", ColumnType::Key, false),
        ]);
        let batch = RecordBatch::new(
            schema,
            vec![
                Column::Numeric(NumericData::F64(NumericColumn::new_non_null(Buffer::new()))),
                Column::Key(KeyColumn::from_values(std::iter::empty())),
            ],
        );
        let expected = batch.clone();
        let (s, a) = export_batch(batch);
        assert_eq!(unsafe { import_batch(s, a) }.expect("round trip"), expected);
    }

    #[test]
    fn nullable_key_round_trips() {
        let mut dict = Dictionary::new();
        dict.intern("only");
        let schema = Schema::new(vec![Field::new("k", ColumnType::Key, true)]);
        let batch = RecordBatch::new(
            schema,
            vec![Column::Key(KeyColumn::new_nullable(
                Buffer::from_slice(&[0, 0, 0]),
                Bitmap::from_bools([true, false, true]),
                dict,
            ))],
        );
        let expected = batch.clone();
        let (s, a) = export_batch(batch);
        assert_eq!(unsafe { import_batch(s, a) }.expect("round trip"), expected);
    }

    #[test]
    fn logical_annotations_survive_round_trip() {
        let schema = Schema::new(vec![
            Field::new("ts", ColumnType::I64, false).with_logical(LogicalType::TimestampNs),
            Field::new("amt", ColumnType::I64, false)
                .with_logical(LogicalType::Decimal64 { scale: 2 }),
        ]);
        let batch = RecordBatch::new(
            schema,
            vec![
                Column::Numeric(NumericData::I64(NumericColumn::new_non_null(
                    Buffer::from_slice(&[1, 2]),
                ))),
                Column::Numeric(NumericData::I64(NumericColumn::new_non_null(
                    Buffer::from_slice(&[199, 250]),
                ))),
            ],
        );
        let expected = batch.clone();
        let (s, a) = export_batch(batch);
        let imported = unsafe { import_batch(s, a) }.expect("round trip");
        assert_eq!(imported, expected);
        assert_eq!(
            imported.schema().fields()[1].logical(),
            Some(LogicalType::Decimal64 { scale: 2 })
        );
    }

    #[test]
    fn stream_round_trips_batches_in_order() {
        let batches: Vec<RecordBatch> = (0..3)
            .map(|i| {
                let schema = Schema::new(vec![Field::new("x", ColumnType::F64, false)]);
                RecordBatch::new(
                    schema,
                    vec![Column::Numeric(NumericData::F64(
                        NumericColumn::new_non_null(Buffer::from_slice(&[
                            i as f64,
                            i as f64 + 0.5,
                        ])),
                    ))],
                )
            })
            .collect();
        let schema = batches[0].schema().clone();
        let expected = batches.clone();
        let stream = export_stream(schema.clone(), expected.clone().into_iter());
        let reader = unsafe { StreamReader::new(stream) }.expect("schema reads");
        assert_eq!(reader.schema(), &schema);
        let collected: Result<Vec<_>, _> = reader.collect();
        assert_eq!(collected.expect("batches read"), batches);
    }

    #[test]
    fn stream_rejects_mismatched_batch_schema() {
        let schema_a = Schema::new(vec![Field::new("a", ColumnType::F64, false)]);
        let schema_b = Schema::new(vec![Field::new("b", ColumnType::I64, false)]);
        let odd_batch = RecordBatch::new(
            schema_b,
            vec![Column::Numeric(NumericData::I64(
                NumericColumn::new_non_null(Buffer::from_slice(&[1])),
            ))],
        );
        let stream = export_stream(schema_a, std::iter::once(odd_batch));
        let mut reader = unsafe { StreamReader::new(stream) }.expect("schema reads");
        let error = reader.next().expect("one item").expect_err("mismatch");
        assert!(error.to_string().contains("schema differs"));
    }

    #[test]
    fn dropping_exports_unconsumed_frees_them() {
        // Export and release without ever importing: the release path
        // alone must free everything (checked for leaks under Miri).
        let (mut schema, mut array) = export_batch(test_batch());
        unsafe {
            release_schema(&mut schema);
            release_array(&mut array);
        }
        let mut stream = export_stream(
            Schema::new(vec![Field::new("x", ColumnType::F64, false)]),
            std::iter::empty(),
        );
        unsafe { release_stream(&mut stream) };
    }

    #[test]
    fn import_rejects_non_struct_root() {
        let schema = new_schema_node("g".to_owned(), "x", 0, vec![], None);
        let array = export_array(RecordBatch::new(Schema::new(vec![]), vec![]));
        let error = unsafe { import_batch(schema, array) }.expect_err("not a struct root");
        assert!(error.to_string().contains("expected struct root"));
    }

    #[test]
    fn import_rejects_released_structs() {
        let (mut schema, mut array) = export_batch(test_batch());
        unsafe {
            release_schema(&mut schema);
            release_array(&mut array);
        }
        let error = unsafe { import_batch(schema, array) }.expect_err("released");
        assert!(error.to_string().contains("released"));
    }
}
