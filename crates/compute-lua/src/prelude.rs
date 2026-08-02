//! The shipped prelude: readable compositions over the native
//! primitives, compiled into the binary (#77.3 = a, ruled 2026-07-29).
//!
//! Every function here is one line over natives, so composing costs a
//! handful of interpreter entries and nothing per element. That is the
//! whole point of the split (#77's hybrid): the performance-bearing
//! layer is native because it must be — Lua cannot link a Rust crate,
//! and element loops written in Lua are the measured ~14× tier — while
//! the *readable* layer stays Lua, where a user can print it, copy it,
//! and change it.
//!
//! Nothing in here is a coined SQL name. These are Lua-side names on
//! Lua-side compositions; the SQL surface gained exactly two names in
//! M5.0 (`var_pop`, `stddev_pop`), both standard.
//!
//! **Read this as documentation that runs.** A user who wants a
//! variation copies the line and edits it — which is why the source is
//! printable (`.prelude`) rather than hidden behind the functions.

/// The prelude's source, run in every [`crate::LuaState`] at creation.
///
/// Kept deliberately small: a composition earns its place here only if
/// it is a *named idiom* a desk would otherwise rewrite. Anything
/// needing a loop belongs in native code instead, and anything used
/// once belongs at the call site.
pub const PRELUDE: &str = r#"-- TallyDB prelude — compositions over the native primitives.
-- Print with .prelude; copy a line and edit it to vary it.

-- Simple (arithmetic) returns: (x[i] - x[i-1]) / x[i-1].
-- The first row is NULL — there is no prior row to return against.
function returns(x)
  return diff(x) / lag(x, 1)
end

-- Expanding aggregates: a trailing frame as wide as the column is a
-- frame that starts at row 1 and grows.
function expanding_sum(x)
  return rolling_sum(x, #x)
end

function expanding_mean(x)
  return rolling_mean(x, #x)
end

function expanding_var(x)
  return rolling_var(x, #x)
end

function expanding_std(x)
  return rolling_std(x, #x)
end

-- The rolling z-score: how many standard deviations from the window
-- mean each row sits. NULL nowhere — a flat window gives 0/0 = NaN,
-- which is a value here (the D2 ruling), not an error.
function zscore(x, w)
  return (x - rolling_mean(x, w)) / rolling_std(x, w)
end
"#;

/// Every name the prelude defines — what `install` must have bound,
/// and what the test below checks is actually callable.
pub const PRELUDE_NAMES: &[&str] = &[
    "returns",
    "expanding_sum",
    "expanding_mean",
    "expanding_var",
    "expanding_std",
    "zscore",
];
