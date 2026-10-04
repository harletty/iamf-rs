//! Rendering: planar element channels → target sound system channels,
//! using the gain matrices ported from libiamf v1.1.0.

use crate::DecodeError;
use crate::matrices::{H2M_TABLE, HoaOrder, M2M_TABLE, MatrixLayout};
use crate::reconstruct::Reconstructed;

/// Renders reconstructed element audio to `output` layout channels
/// (planar). Same-layout channel input is a passthrough via the identity
/// matrix in the tables.
pub fn render(input: &Reconstructed, output: MatrixLayout) -> Result<Vec<Vec<f32>>, DecodeError> {
    let mut out = Vec::new();
    render_into(input, output, &mut out)?;
    Ok(out)
}

/// [`render`] into planes the caller keeps from one frame to the next.
pub(crate) fn render_into(
    input: &Reconstructed,
    output: MatrixLayout,
    out: &mut Vec<Vec<f32>>,
) -> Result<(), DecodeError> {
    match input {
        Reconstructed::Channels { matrix, planar } => {
            render_channels_into(*matrix, output, planar, out)
        }
        Reconstructed::Hoa { order, planar } => render_hoa_into(*order, output, planar, out),
    }
}

/// Makes `out` `channels` silent planes of `frames` samples, keeping the
/// planes it already holds.
fn silence(out: &mut Vec<Vec<f32>>, channels: usize, frames: usize) {
    out.resize_with(channels, Vec::new);
    for plane in out.iter_mut() {
        plane.clear();
        plane.resize(frames, 0.0);
    }
}

/// [`render`] for channel-based planes, into planes the caller keeps from
/// one frame to the next.
pub(crate) fn render_channels_into(
    input: MatrixLayout,
    output: MatrixLayout,
    planar: &[Vec<f32>],
    out: &mut Vec<Vec<f32>>,
) -> Result<(), DecodeError> {
    let entry = M2M_TABLE
        .iter()
        .find(|e| e.input == input && e.output == output)
        .ok_or(DecodeError::Unimplemented("no m2m matrix for layout pair"))?;
    if planar.len() != entry.m {
        return Err(DecodeError::InvalidDescriptors(format!(
            "m2m expects {} input channels, got {}",
            entry.m,
            planar.len()
        )));
    }
    let frames = planar.first().map_or(0, Vec::len);
    silence(out, entry.n, frames);
    for (m, plane) in planar.iter().enumerate() {
        for (n, out_plane) in out.iter_mut().enumerate() {
            let gain = entry.mat[m * entry.n + n];
            if gain == 0.0 {
                continue;
            }
            for (o, &s) in out_plane.iter_mut().zip(plane.iter()) {
                *o += gain * s;
            }
        }
    }
    Ok(())
}

/// [`render`] for ambisonics planes, into planes the caller keeps from one
/// frame to the next.
fn render_hoa_into(
    order: HoaOrder,
    output: MatrixLayout,
    planar: &[Vec<f32>],
    out: &mut Vec<Vec<f32>>,
) -> Result<(), DecodeError> {
    let entry = H2M_TABLE
        .iter()
        .find(|e| e.input == order && e.output == output)
        .ok_or(DecodeError::Unimplemented("no h2m matrix for layout pair"))?;
    if planar.len() != entry.m {
        return Err(DecodeError::InvalidDescriptors(format!(
            "h2m expects {} ambisonics channels, got {}",
            entry.m,
            planar.len()
        )));
    }
    let frames = planar.first().map_or(0, Vec::len);
    // The matrix skips the LFE channels: they stay silent at lfe1/lfe2
    // (libiamf's LFE synthesis filter is optional and off by default), and
    // the matrix rows fill the other planes in order.
    silence(out, entry.total_channels, frames);
    let speakers = out
        .iter_mut()
        .enumerate()
        .filter(|&(i, _)| Some(i) != entry.lfe1 && Some(i) != entry.lfe2)
        .map(|(_, plane)| plane);
    for (n, out_plane) in speakers.take(entry.n).enumerate() {
        for (m, plane) in planar.iter().enumerate() {
            let gain = entry.mat[n * entry.m + m];
            if gain == 0.0 {
                continue;
            }
            for (o, &s) in out_plane.iter_mut().zip(plane.iter()) {
                *o += gain * s;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stereo_to_stereo_is_identity() {
        let input = Reconstructed::Channels {
            matrix: MatrixLayout::Stereo,
            planar: vec![vec![0.5, -0.5], vec![0.25, -0.25]],
        };
        let out = render(&input, MatrixLayout::Bs2051A).unwrap();
        assert_eq!(out[0], vec![0.5, -0.5]);
        assert_eq!(out[1], vec![0.25, -0.25]);
    }

    #[test]
    fn mono_to_stereo_pans_center() {
        let input = Reconstructed::Channels {
            matrix: MatrixLayout::Mono,
            planar: vec![vec![1.0]],
        };
        let out = render(&input, MatrixLayout::Bs2051A).unwrap();
        assert!((out[0][0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
        assert!((out[1][0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn surround_51_to_stereo_downmix() {
        // C at full scale: expect 1/sqrt(2) into both L and R.
        let mut planar = vec![vec![0.0f32]; 6];
        planar[2] = vec![1.0];
        let input = Reconstructed::Channels {
            matrix: MatrixLayout::Iamf51,
            planar,
        };
        let out = render(&input, MatrixLayout::Bs2051A).unwrap();
        assert_eq!(out.len(), 2);
        assert!(
            (out[0][0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6,
            "got {}",
            out[0][0]
        );
        assert!((out[1][0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn foa_to_stereo_shape() {
        let input = Reconstructed::Hoa {
            order: HoaOrder::Foa,
            planar: vec![vec![1.0]; 4],
        };
        let out = render(&input, MatrixLayout::Bs2051A).unwrap();
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn foa_to_51_inserts_silent_lfe() {
        let input = Reconstructed::Hoa {
            order: HoaOrder::Foa,
            planar: vec![vec![1.0]; 4],
        };
        let out = render(&input, MatrixLayout::Bs2051B).unwrap();
        assert_eq!(out.len(), 6);
        assert_eq!(out[3], vec![0.0]); // LFE slot silent
    }
}
