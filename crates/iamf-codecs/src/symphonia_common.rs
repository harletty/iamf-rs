//! Shared adapter: drives a symphonia `Decoder` as an IAMF
//! [`SubstreamDecoder`] (one raw codec frame per IAMF audio frame OBU).

use iamf_dec::{DecodeError, DecodedFrame, SubstreamDecoder};
use symphonia_core::audio::{AudioBuffer, AudioBufferRef, Signal};
use symphonia_core::codecs::Decoder;
use symphonia_core::conv::IntoSample;
use symphonia_core::formats::Packet;
use symphonia_core::sample::Sample;

pub(crate) struct SymphoniaSubstreamDecoder<D: Decoder> {
    decoder: D,
    channels: u8,
    sample_rate: u32,
}

/// Converts a decoded planar buffer straight into `out`, interleaved: the
/// conversion `SampleBuffer::copy_interleaved_typed` does, without the
/// intermediate buffer and its strided writes.
fn interleave_into<S>(decoded: &AudioBuffer<S>, out: &mut Vec<f32>)
where
    S: Sample + IntoSample<f32>,
{
    let channels = decoded.spec().channels.count();
    out.clear();
    match channels {
        1 => out.extend(decoded.chan(0).iter().map(|&s| s.into_sample())),
        2 => {
            let (left, right) = (decoded.chan(0), decoded.chan(1));
            out.resize(left.len().min(right.len()) * 2, 0.0);
            for ((pair, &l), &r) in out.chunks_exact_mut(2).zip(left).zip(right) {
                pair[0] = l.into_sample();
                pair[1] = r.into_sample();
            }
        }
        _ => {
            out.resize(decoded.frames() * channels, 0.0);
            for ch in 0..channels {
                let slots = out.iter_mut().skip(ch).step_by(channels);
                for (dst, &src) in slots.zip(decoded.chan(ch)) {
                    *dst = src.into_sample();
                }
            }
        }
    }
}

impl<D: Decoder> SymphoniaSubstreamDecoder<D> {
    pub(crate) fn new(decoder: D, channels: u8, sample_rate: u32) -> Self {
        Self {
            decoder,
            channels,
            sample_rate,
        }
    }
}

impl<D: Decoder> SubstreamDecoder for SymphoniaSubstreamDecoder<D> {
    fn decode(&mut self, packet: &[u8], out: &mut DecodedFrame) -> Result<(), DecodeError> {
        let packet = Packet::new_from_slice(0, 0, 0, packet);
        let decoded = self
            .decoder
            .decode(&packet)
            .map_err(|e| DecodeError::CorruptPacket(e.to_string()))?;
        match decoded {
            AudioBufferRef::U8(buf) => interleave_into(&buf, &mut out.samples),
            AudioBufferRef::U16(buf) => interleave_into(&buf, &mut out.samples),
            AudioBufferRef::U24(buf) => interleave_into(&buf, &mut out.samples),
            AudioBufferRef::U32(buf) => interleave_into(&buf, &mut out.samples),
            AudioBufferRef::S8(buf) => interleave_into(&buf, &mut out.samples),
            AudioBufferRef::S16(buf) => interleave_into(&buf, &mut out.samples),
            AudioBufferRef::S24(buf) => interleave_into(&buf, &mut out.samples),
            AudioBufferRef::S32(buf) => interleave_into(&buf, &mut out.samples),
            AudioBufferRef::F32(buf) => interleave_into(&buf, &mut out.samples),
            AudioBufferRef::F64(buf) => interleave_into(&buf, &mut out.samples),
        }
        out.channels = self.channels;
        out.sample_rate = self.sample_rate;
        Ok(())
    }

    fn reset(&mut self) {
        self.decoder.reset();
    }
}
