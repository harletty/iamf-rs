//! Split elements on the libiamf vectors that mix two channel-based
//! elements: an element handed out by `split_element`, rendered to the
//! output layout and added back, gives the mix decoded without the split.

mod common;

use iamf_codecs::DefaultFactory;
use iamf_dec::layout::{SoundSystem, loudspeaker_info};
use iamf_dec::reconstruct::Reconstructed;
use iamf_dec::render::render;
use iamf_dec::stream::{DecodedElement, StreamDecoder, StreamSettings};

fn vector(name: &str) -> Option<Vec<u8>> {
    std::fs::read(common::vectors_dir().join(format!("{name}.iamf"))).ok()
}

/// Every temporal unit, interleaved, and the split elements beside it.
fn decode(data: &[u8], split: Option<u32>) -> Vec<(Vec<f32>, Vec<DecodedElement>)> {
    let mut settings = StreamSettings::default();
    settings.layout = SoundSystem::J;
    let mut decoder =
        StreamDecoder::new_from_descriptors(data, settings, &DefaultFactory).expect("configures");
    if let Some(id) = split {
        assert!(decoder.split_element(id), "element {id} splits");
        assert!(decoder.has_rendered_elements());
    }
    let mut units = Vec::new();
    let mut unit = Vec::new();
    for chunk in data.chunks(997) {
        decoder.decode(chunk).expect("decodes");
        while decoder
            .get_output_temporal_unit_f32(&mut unit)
            .expect("renders")
        {
            units.push((unit.clone(), decoder.elements().to_vec()));
        }
    }
    decoder.signal_end_of_decoding();
    while decoder
        .get_output_temporal_unit_f32(&mut unit)
        .expect("renders")
    {
        units.push((unit.clone(), decoder.elements().to_vec()));
    }
    units
}

fn check(name: &str, split: u32, layout: u8) {
    let data = require_vectors!(vector(name), name);
    let whole = decode(&data, None);
    let parts = decode(&data, Some(split));
    assert_eq!(whole.len(), parts.len());
    assert!(!whole.is_empty());
    let channels = SoundSystem::J.channels();
    let mut peak = 0f32;
    let mut worst = 0f32;
    for ((mix, none), (rest, elements)) in whole.iter().zip(&parts) {
        assert!(none.is_empty());
        let [element] = elements.as_slice() else {
            panic!("one split element per unit, got {}", elements.len());
        };
        assert_eq!(element.audio_element_id, split);
        assert_eq!(element.loudspeaker_layout, layout);
        let info = loudspeaker_info(layout).unwrap();
        assert_eq!(element.planes.len(), info.channels);
        let rendered = render(
            &Reconstructed::Channels {
                matrix: info.matrix,
                planar: element.planes.clone(),
            },
            SoundSystem::J.matrix_layout(),
        )
        .unwrap();
        let frames = mix.len() / channels;
        assert_eq!(rest.len(), mix.len());
        for (plane, samples) in rendered.iter().enumerate() {
            assert_eq!(samples.len(), frames);
            for (frame, &s) in samples.iter().enumerate() {
                let i = frame * channels + plane;
                peak = peak.max(mix[i].abs());
                worst = worst.max((rest[i] + s - mix[i]).abs());
            }
        }
    }
    assert!(peak > 0.01, "{name}: silent");
    assert!(worst < 1e-5, "{name}: split + rest differs by {worst}");
}

/// Two-layer stereo/5.1 element 300 beside stereo element 301 (Opus).
#[test]
fn split_stereo_element_adds_back_to_the_mix() {
    check("test_000087", 301, 1);
}

/// The scalable element is handed out at its highest layer, 5.1.
#[test]
fn split_scalable_element_is_its_highest_layer() {
    check("test_000087", 300, 2);
}

#[test]
fn only_channel_based_elements_of_the_mix_split() {
    let data = require_vectors!(vector("test_000087"), "test_000087");
    let mut settings = StreamSettings::default();
    settings.layout = SoundSystem::J;
    let mut decoder =
        StreamDecoder::new_from_descriptors(&data, settings, &DefaultFactory).unwrap();
    assert!(!decoder.split_element(999));
    assert!(decoder.split_element(300));
    assert!(decoder.split_element(301));
    // Nothing left to render into the output layout.
    assert!(!decoder.has_rendered_elements());
}
