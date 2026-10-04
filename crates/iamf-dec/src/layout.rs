//! Channel layouts and sound systems (IAMF §3.7.4 loudspeaker_layout,
//! §7.3.2 sound systems), ported from libiamf v1.1.0 IAMF_layout.c.

use crate::matrices::MatrixLayout;

/// Output target: sound systems 0..=13 (mix presentation layout numbering,
/// also iamf-tools `OutputLayout`) plus binaural (14, iamf-tools
/// `kIAMF_Binaural`).
///
/// With the `binaural` feature, elements whose `headphones_rendering_mode`
/// is 1 go through the native obr-style HRTF renderer (see
/// `crate::binaural`); mode-0 elements — and every element when the
/// feature is off — render through the stereo gain matrices, matching
/// libiamf built without its binauralizer libraries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SoundSystem {
    /// Sound system A: 2.0 stereo (BS.2051-A: L, R).
    A,
    /// Sound system B: 5.1 surround (BS.2051-B: L, R, C, LFE, Ls, Rs).
    B,
    /// Sound system C: 5.1.2 immersive (BS.2051-C).
    C,
    /// Sound system D: 5.1.4 immersive (BS.2051-D).
    D,
    /// Sound system E: 7.1.2 immersive (BS.2051-E).
    E,
    /// Sound system F: 7.1.4 immersive (BS.2051-F).
    F,
    /// Sound system G: 9.1.4 immersive (BS.2051-G).
    G,
    /// Sound system H: 22.2 immersive (BS.2051-H).
    H,
    /// Sound system I: 7.1 surround (BS.2051-I).
    I,
    /// Sound system J: 7.1.4 immersive (BS.2051-J).
    J,
    /// Extended 7.1.2 layout.
    Ext712,
    /// Extended 3.1.2 layout.
    Ext312,
    /// Single-channel mono.
    Mono,
    /// Extended 9.1.6 layout.
    Ext916,
    /// Binaural headphone rendering (layout 14).
    Binaural,
}

impl TryFrom<u8> for SoundSystem {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        SoundSystem::from_u8(value).ok_or(())
    }
}

impl SoundSystem {
    /// Maps a raw layout integer ID (0..=14) to a [`SoundSystem`].
    pub fn from_u8(value: u8) -> Option<Self> {
        Some(match value {
            0 => SoundSystem::A,
            1 => SoundSystem::B,
            2 => SoundSystem::C,
            3 => SoundSystem::D,
            4 => SoundSystem::E,
            5 => SoundSystem::F,
            6 => SoundSystem::G,
            7 => SoundSystem::H,
            8 => SoundSystem::I,
            9 => SoundSystem::J,
            10 => SoundSystem::Ext712,
            11 => SoundSystem::Ext312,
            12 => SoundSystem::Mono,
            13 => SoundSystem::Ext916,
            14 => SoundSystem::Binaural,
            _ => return None,
        })
    }

    /// Number of output audio channels for this sound system.
    pub fn channels(&self) -> usize {
        match self {
            SoundSystem::Mono => 1,
            SoundSystem::A | SoundSystem::Binaural => 2,
            SoundSystem::B | SoundSystem::Ext312 => 6,
            SoundSystem::C | SoundSystem::I => 8,
            SoundSystem::D | SoundSystem::Ext712 => 10,
            SoundSystem::E => 11,
            SoundSystem::F | SoundSystem::J => 12,
            SoundSystem::G => 14,
            SoundSystem::Ext916 => 16,
            SoundSystem::H => 24,
        }
    }

    /// The output-side key into the rendering matrix tables.
    pub fn matrix_layout(&self) -> MatrixLayout {
        match self {
            SoundSystem::A => MatrixLayout::Bs2051A,
            SoundSystem::B => MatrixLayout::Bs2051B,
            SoundSystem::C => MatrixLayout::Bs2051C,
            SoundSystem::D => MatrixLayout::Bs2051D,
            SoundSystem::E => MatrixLayout::Bs2051E,
            SoundSystem::F => MatrixLayout::Bs2051F,
            SoundSystem::G => MatrixLayout::Bs2051G,
            SoundSystem::H => MatrixLayout::Bs2051H,
            SoundSystem::I => MatrixLayout::Bs2051I,
            SoundSystem::J => MatrixLayout::Bs2051J,
            SoundSystem::Ext712 => MatrixLayout::Iamf712,
            SoundSystem::Ext312 => MatrixLayout::Iamf312,
            SoundSystem::Mono => MatrixLayout::Mono,
            SoundSystem::Ext916 => MatrixLayout::Iamf916,
            SoundSystem::Binaural => MatrixLayout::Binaural,
        }
    }
}

/// Static info for a channel-based element's loudspeaker_layout (§3.7.4).
#[derive(Debug)]
pub struct LoudspeakerInfo {
    /// Number of channels in the layout.
    pub channels: usize,
    /// Position in rendering (channel_layout) order of each channel in
    /// substream-decode order: coupled pairs first, then C, then LFE.
    pub decoding_map: &'static [usize],
    /// Input-side key into the rendering matrix tables.
    pub matrix: MatrixLayout,
}

/// The sound system a loudspeaker_layout corresponds to (used for layer
/// selection: a layer matching the playback sound system needs no
/// rendering conversion).
pub fn loudspeaker_sound_system(loudspeaker_layout: u8) -> Option<SoundSystem> {
    Some(match loudspeaker_layout {
        0 => SoundSystem::Mono,
        1 => SoundSystem::A,
        2 => SoundSystem::B,
        3 => SoundSystem::C,
        4 => SoundSystem::D,
        5 => SoundSystem::I,
        6 => SoundSystem::Ext712,
        7 => SoundSystem::J,
        8 => SoundSystem::Ext312,
        _ => return None,
    })
}

/// Loudspeaker layouts 0..=8 (binaural is not supported as an input;
/// expanded layouts are described by [`expanded_info`]).
pub fn loudspeaker_info(loudspeaker_layout: u8) -> Option<&'static LoudspeakerInfo> {
    static INFOS: [LoudspeakerInfo; 9] = [
        // 0: Mono
        LoudspeakerInfo {
            channels: 1,
            decoding_map: &[0],
            matrix: MatrixLayout::Mono,
        },
        // 1: Stereo
        LoudspeakerInfo {
            channels: 2,
            decoding_map: &[0, 1],
            matrix: MatrixLayout::Stereo,
        },
        // 2: 5.1
        LoudspeakerInfo {
            channels: 6,
            decoding_map: &[0, 1, 4, 5, 2, 3],
            matrix: MatrixLayout::Iamf51,
        },
        // 3: 5.1.2
        LoudspeakerInfo {
            channels: 8,
            decoding_map: &[0, 1, 4, 5, 6, 7, 2, 3],
            matrix: MatrixLayout::Iamf512,
        },
        // 4: 5.1.4
        LoudspeakerInfo {
            channels: 10,
            decoding_map: &[0, 1, 4, 5, 6, 7, 8, 9, 2, 3],
            matrix: MatrixLayout::Iamf514,
        },
        // 5: 7.1
        LoudspeakerInfo {
            channels: 8,
            decoding_map: &[0, 1, 4, 5, 6, 7, 2, 3],
            matrix: MatrixLayout::Iamf71,
        },
        // 6: 7.1.2
        LoudspeakerInfo {
            channels: 10,
            decoding_map: &[0, 1, 4, 5, 6, 7, 8, 9, 2, 3],
            matrix: MatrixLayout::Iamf712,
        },
        // 7: 7.1.4
        LoudspeakerInfo {
            channels: 12,
            decoding_map: &[0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 2, 3],
            matrix: MatrixLayout::Iamf714,
        },
        // 8: 3.1.2
        LoudspeakerInfo {
            channels: 6,
            decoding_map: &[0, 1, 4, 5, 2, 3],
            matrix: MatrixLayout::Iamf312,
        },
    ];
    INFOS.get(usize::from(loudspeaker_layout))
}

/// Static info for an expanded loudspeaker layout (`loudspeaker_layout`
/// 15, §3.6.2 `expanded_loudspeaker_layout`), after the Open Audio
/// Renderer: a layout is either one of the larger sound systems whole
/// (9.1.6, 10.2.9.3, 7.1.5.4) or a subset of a reference layout, rendered
/// through the reference's matrices restricted to the rows of the channels
/// it carries — the others are simply absent, not silent planes.
#[derive(Debug)]
pub struct ExpandedInfo {
    /// Number of channels the layout carries.
    pub channels: usize,
    /// Position in rendering order of each channel in substream-decode
    /// order: coupled pairs first, then the mono channels (libiamf
    /// `decoding_map`).
    pub decoding_map: &'static [usize],
    /// Input-side key into the rendering matrix tables: the layout itself
    /// when it is whole, its reference layout when it is a subset.
    pub matrix: MatrixLayout,
    /// For a subset, the row of `matrix` each channel takes, in rendering
    /// order (the reference layout's channel order); `None` when the
    /// layout is `matrix` whole.
    pub rows: Option<&'static [usize]>,
}

/// Expanded loudspeaker layouts 0..=19 (IAMF v1.1's 0..=12 and v2.0's
/// 13..=19); the reference layouts and channel positions are OAR's
/// (`iamf_*_spl` tables of its EAR renderer, libiamf `iamf_layouts`).
pub fn expanded_info(expanded_loudspeaker_layout: u8) -> Option<&'static ExpandedInfo> {
    const fn subset(
        decoding_map: &'static [usize],
        matrix: MatrixLayout,
        rows: &'static [usize],
    ) -> ExpandedInfo {
        ExpandedInfo {
            channels: rows.len(),
            decoding_map,
            matrix,
            rows: Some(rows),
        }
    }
    const fn whole(decoding_map: &'static [usize], matrix: MatrixLayout) -> ExpandedInfo {
        ExpandedInfo {
            channels: decoding_map.len(),
            decoding_map,
            matrix,
            rows: None,
        }
    }
    // Rows of 7.1.4: L, R, C, LFE, Lss, Rss, Lrs, Rrs, Ltf, Rtf, Ltb, Rtb.
    // Rows of 5.1.4: L, R, C, LFE, Ls, Rs, Ltf, Rtf, Ltb, Rtb.
    // Rows of 9.1.6: FL, FR, FC, LFE, BL, BR, FLc, FRc, SiL, SiR, TpFL,
    // TpFR, TpBL, TpBR, TpSiL, TpSiR.
    // Rows of 10.2.9.3: FL, FR, FC, LFE1, BL, BR, FLc, FRc, BC, LFE2, SiL,
    // SiR, TpFL, TpFR, TpFC, TpC, TpBL, TpBR, TpSiL, TpSiR, TpBC, BtFC,
    // BtFL, BtFR.
    // Rows of 7.1.5.4: L, R, C, LFE, Lss, Rss, Lrs, Rrs, Ltf, Rtf, TpC, Ltb,
    // Rtb, BtFL, BtFR, BtBL, BtBR.
    static INFOS: [ExpandedInfo; 20] = [
        // 0: LFE of 7.1.4
        subset(&[0], MatrixLayout::Iamf714, &[3]),
        // 1: Stereo-S, Ls/Rs of 5.1.4
        subset(&[0, 1], MatrixLayout::Iamf514, &[4, 5]),
        // 2: Stereo-SS, Lss/Rss of 7.1.4
        subset(&[0, 1], MatrixLayout::Iamf714, &[4, 5]),
        // 3: Stereo-RS, Lrs/Rrs of 7.1.4
        subset(&[0, 1], MatrixLayout::Iamf714, &[6, 7]),
        // 4: Stereo-TF, Ltf/Rtf of 7.1.4
        subset(&[0, 1], MatrixLayout::Iamf714, &[8, 9]),
        // 5: Stereo-TB, Ltb/Rtb of 7.1.4
        subset(&[0, 1], MatrixLayout::Iamf714, &[10, 11]),
        // 6: Top-4ch, Ltf/Rtf/Ltb/Rtb of 7.1.4
        subset(&[0, 1, 2, 3], MatrixLayout::Iamf714, &[8, 9, 10, 11]),
        // 7: 3.0ch, L/R/C of 7.1.4
        subset(&[0, 1, 2], MatrixLayout::Iamf714, &[0, 1, 2]),
        // 8: 9.1.6
        whole(
            &[6, 7, 0, 1, 8, 9, 4, 5, 10, 11, 14, 15, 12, 13, 2, 3],
            MatrixLayout::Iamf916,
        ),
        // 9: Stereo-F, FL/FR of 9.1.6
        subset(&[0, 1], MatrixLayout::Iamf916, &[0, 1]),
        // 10: Stereo-Si, SiL/SiR of 9.1.6
        subset(&[0, 1], MatrixLayout::Iamf916, &[8, 9]),
        // 11: Stereo-TpSi, TpSiL/TpSiR of 9.1.6
        subset(&[0, 1], MatrixLayout::Iamf916, &[14, 15]),
        // 12: Top-6ch, TpFL/TpFR/TpBL/TpBR/TpSiL/TpSiR of 9.1.6 (decoded
        // TpFL/TpFR, TpSiL/TpSiR, TpBL/TpBR)
        subset(
            &[0, 1, 4, 5, 2, 3],
            MatrixLayout::Iamf916,
            &[10, 11, 12, 13, 14, 15],
        ),
        // 13: 10.2.9.3
        whole(
            &[
                6, 7, 0, 1, 10, 11, 4, 5, 12, 13, 18, 19, 16, 17, 22, 23, 2, 8, 14, 15, 20, 21, 3,
                9,
            ],
            MatrixLayout::Iamf10293,
        ),
        // 14: LFE-Pair, LFE1/LFE2 of 10.2.9.3
        subset(&[0, 1], MatrixLayout::Iamf10293, &[3, 9]),
        // 15: Bottom-3ch, BtFC/BtFL/BtFR of 10.2.9.3 (decoded BtFL/BtFR, BtFC)
        subset(&[1, 2, 0], MatrixLayout::Iamf10293, &[21, 22, 23]),
        // 16: 7.1.5.4
        whole(
            &[0, 1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 2, 10, 3],
            MatrixLayout::Iamf7154,
        ),
        // 17: Bottom-4ch, BtFL/BtFR/BtBL/BtBR of 7.1.5.4
        subset(&[0, 1, 2, 3], MatrixLayout::Iamf7154, &[13, 14, 15, 16]),
        // 18: Top-1ch, TpC of 7.1.5.4
        subset(&[0], MatrixLayout::Iamf7154, &[10]),
        // 19: Top-5ch, Ltf/Rtf/TpC/Ltb/Rtb of 7.1.5.4 (decoded Ltf/Rtf,
        // Ltb/Rtb, TpC)
        subset(
            &[0, 1, 3, 4, 2],
            MatrixLayout::Iamf7154,
            &[8, 9, 10, 11, 12],
        ),
    ];
    INFOS.get(usize::from(expanded_loudspeaker_layout))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrices::M2M_TABLE;
    use crate::reconstruct::Reconstructed;
    use crate::render::render;

    #[test]
    fn expanded_tables_are_consistent() {
        for layout in 0..=19 {
            let info = expanded_info(layout).unwrap();
            assert_eq!(info.decoding_map.len(), info.channels, "layout {layout}");
            let mut seen = vec![false; info.channels];
            for &position in info.decoding_map {
                assert!(
                    !std::mem::replace(&mut seen[position], true),
                    "layout {layout}"
                );
            }
            for entry in M2M_TABLE.iter().filter(|e| e.input == info.matrix) {
                match info.rows {
                    Some(rows) => assert!(rows.iter().all(|&r| r < entry.m), "layout {layout}"),
                    None => assert_eq!(entry.m, info.channels, "layout {layout}"),
                }
            }
        }
        assert!(expanded_info(20).is_none());
    }

    #[test]
    fn a_subset_renders_as_its_reference_with_the_other_channels_silent() {
        for layout in (0..=19).filter(|&l| expanded_info(l).unwrap().rows.is_some()) {
            let info = expanded_info(layout).unwrap();
            let rows = info.rows.unwrap();
            for entry in M2M_TABLE.iter().filter(|e| e.input == info.matrix) {
                let subset: Vec<Vec<f32>> = (0..rows.len())
                    .map(|i| vec![0.25 + i as f32, -0.5])
                    .collect();
                let mut full = vec![vec![0.0f32; 2]; entry.m];
                for (plane, &row) in subset.iter().zip(rows) {
                    full[row].clone_from(plane);
                }
                let ours = render(
                    &Reconstructed::Channels {
                        matrix: info.matrix,
                        rows: Some(rows),
                        planar: subset,
                    },
                    entry.output,
                )
                .unwrap();
                let reference = render(
                    &Reconstructed::Channels {
                        matrix: info.matrix,
                        rows: None,
                        planar: full,
                    },
                    entry.output,
                )
                .unwrap();
                assert_eq!(ours, reference, "layout {layout} to {:?}", entry.output);
            }
        }
    }
}
