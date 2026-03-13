//! Shader template engine for generating precision-variant WGSL shaders.
//!
//! Replaces the ~18 near-identical `_f32.wgsl` / `_f16.wgsl` / `_bf16.wgsl` files with
//! a single canonical template per operation. The template uses placeholders that are
//! resolved at runtime before shader compilation.
//!
//! # Placeholders
//!
//! | Placeholder | f32 | f16 | bf16 |
//! |-------------|-----|-----|------|
//! | `{{FLOAT_TYPE}}` | `f32` | `f16` | `u32` |
//! | `{{ZERO_LIT}}` | `0.0` | `0.0h` | `0u` |
//! | `{{ENABLE_F16}}` | `` | `enable f16;\n` | `` |

/// Precision variant for shader template resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderPrecision {
    /// 32-bit float (`f32`).
    F32,
    /// 16-bit float (`f16`), requires `enable f16;` extension.
    F16,
    /// Brain float 16, stored as `u32` with manual bitcast.
    BF16,
}

/// Resolve a shader template string by replacing precision placeholders.
///
/// Supports two modes:
/// - **Template mode**: replaces `{{FLOAT_TYPE}}`, `{{ZERO_LIT}}`, `{{ENABLE_F16}}`
/// - **F32-source mode**: takes a working f32 shader and transforms it for f16/bf16
///   by replacing `array<f32>` → `array<f16>`, etc. For f32 input, returns as-is.
pub fn resolve(template: &str, precision: ShaderPrecision) -> String {
    match precision {
        ShaderPrecision::F32 => {
            // Template mode: replace placeholders; f32 source mode: return as-is
            template
                .replace("{{ENABLE_F16}}", "")
                .replace("{{FLOAT_TYPE}}", "f32")
                .replace("{{ZERO_LIT}}", "0.0")
        }
        ShaderPrecision::F16 => {
            let mut s = template.to_string();
            // F32-source mode: convert f32 types to f16
            s = s.replace("array<f32>", "array<f16>");
            s = s.replace(": f32", ": f16");
            s = s.replace("-> f32", "-> f16");
            s = s.replace("var res: f32", "var res: f16");
            // Template mode
            s = s.replace("{{ENABLE_F16}}", "enable f16;\n");
            s = s.replace("{{FLOAT_TYPE}}", "f16");
            s = s.replace("{{ZERO_LIT}}", "0.0h");
            // Prepend enable if not already present
            if !s.contains("enable f16") {
                s = format!("enable f16;\n{}", s);
            }
            s
        }
        ShaderPrecision::BF16 => {
            let mut s = template.to_string();
            // F32-source mode: convert f32 storage to u32 (bf16 packed)
            s = s.replace("array<f32>", "array<u32>");
            // Template mode
            s = s.replace("{{ENABLE_F16}}", "");
            s = s.replace("{{FLOAT_TYPE}}", "u32");
            s = s.replace("{{ZERO_LIT}}", "0u");
            s
        }
    }
}

/// Select the appropriate [`ShaderPrecision`] for a given `cv_core::DataType`.
pub fn precision_for_type<T: cv_core::float::Float + 'static>() -> crate::Result<ShaderPrecision> {
    match cv_core::DataType::from_type::<T>() {
        Ok(cv_core::DataType::F32) | Ok(cv_core::DataType::F64) => Ok(ShaderPrecision::F32),
        #[cfg(feature = "half-precision")]
        Ok(cv_core::DataType::F16) => Ok(ShaderPrecision::F16),
        #[cfg(feature = "half-precision")]
        Ok(cv_core::DataType::BF16) => Ok(ShaderPrecision::BF16),
        _ => Ok(ShaderPrecision::F32),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_f32() {
        let template =
            "{{ENABLE_F16}}var<storage> data: array<{{FLOAT_TYPE}}>;\nlet z = {{ZERO_LIT}};";
        let result = resolve(template, ShaderPrecision::F32);
        assert_eq!(result, "var<storage> data: array<f32>;\nlet z = 0.0;");
    }

    #[test]
    fn test_resolve_f16() {
        let template =
            "{{ENABLE_F16}}var<storage> data: array<{{FLOAT_TYPE}}>;\nlet z = {{ZERO_LIT}};";
        let result = resolve(template, ShaderPrecision::F16);
        assert_eq!(
            result,
            "enable f16;\nvar<storage> data: array<f16>;\nlet z = 0.0h;"
        );
    }

    #[test]
    fn test_resolve_bf16() {
        let template =
            "{{ENABLE_F16}}var<storage> data: array<{{FLOAT_TYPE}}>;\nlet z = {{ZERO_LIT}};";
        let result = resolve(template, ShaderPrecision::BF16);
        assert_eq!(result, "var<storage> data: array<u32>;\nlet z = 0u;");
    }
}
