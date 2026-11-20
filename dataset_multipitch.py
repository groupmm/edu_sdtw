"""Sustained sinusoids with frame-activity and onset annotations.

Example:
    data = MultipitchDataset(max_polyphony=3)
    waveform, strong_frames, weak_frames, strong_onsets, weak_onsets = data[0]

Use collate_multipitch with torch.utils.data.DataLoader to batch examples.
"""

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class MultipitchDataset(Dataset):
    """Five-second mixtures whose active frequencies change at discrete times.

    Return (waveform, strong_frames, weak_frames, strong_onsets, weak_onsets).
    All tensors are float32 on CPU. Targets have four frequency channels;
    quiet frames are all zeros. Frequency labels are indexed from 0 to 3.

    * waveform: (1, N), summed, gated sinusoids plus Gaussian noise.
    * strong_frames: (N, K), active notes throughout their sustained intervals.
    * weak_frames: (M_frame, K), one activity vector per constant-state interval.
    * strong_onsets: (N, K), impulses at inactive-to-active note transitions.
    * weak_onsets: (M_onset, K), onset-group vectors separated by quiet vectors,
      including initial and final quiet vectors, as in OnsetDataset.

    Held notes keep their phase across changes in other notes. Only new notes
    produce onsets; offset-only changes add no onset group. Pulses are gated
    abruptly and summed without normalization. Notes may remain active at EOF.
    change_frames, onsets, and offsets retain sample indices for inspection.
    """

    sr = 100
    #duration = 5.0
    #freqs = (10, 20, 30, 40)

    def __init__(self, size=128, seed=0, noise_std=0.2, max_polyphony=3, freqs=(10,20,30,40), duration=5.0, constant_range=[0.1, 0.4]):
        
        self.freqs = freqs
        self.duration = duration
        self.constant_range = constant_range

        if (isinstance(max_polyphony, bool) or not isinstance(max_polyphony, int)
                or not 1 <= max_polyphony <= len(self.freqs)):
            raise ValueError("max_polyphony must be an integer from 1 to 4")
        if not np.isfinite(noise_std) or noise_std < 0:
            raise ValueError("noise_std must be finite and nonnegative")

        self.max_polyphony = max_polyphony
        self.num_classes = len(self.freqs)
        self.class_names = [f"{f} Hz" for f in self.freqs]
        self.examples = []
        self.change_frames = []
        self.onsets = []
        self.offsets = []

        rng = np.random.default_rng(seed)
        noise_rng = torch.Generator().manual_seed(seed)
        n = int(self.duration * self.sr)
        # Accurate oscillator phases, with float32 signals/targets for PyTorch.
        t = torch.arange(n, dtype=torch.float64) / self.sr
        oscillators = torch.stack([torch.sin(2 * torch.pi * f * t)
                                   for f in self.freqs], dim=-1).float()

        for _ in range(size):
            current = np.zeros(len(self.freqs), dtype=bool)
            quiet = torch.zeros(len(self.freqs), dtype=torch.float32)
            states, starts = [quiet.clone()], [0]
            onset_groups = [quiet.clone()]
            strong_onsets = torch.zeros(n, len(self.freqs), dtype=torch.float32)
            onsets, offsets = [], []
            start_t = rng.uniform(self.constant_range[0], self.constant_range[1])

            # Leave at least 150 ms after the last change to observe its state.
            while start_t < self.duration - 0.1:
                sample = int(start_t * self.sr)
                # Draw a different active set; shared notes continue sounding.
                new = current.copy()
                while np.array_equal(new, current):
                    new = np.zeros(len(self.freqs), dtype=bool)
                    count = int(rng.integers(0, max_polyphony + 1))
                    new[rng.choice(len(self.freqs), size=count, replace=False)] = True

                state = torch.tensor(new, dtype=torch.float32)
                onset = torch.tensor(new & ~current, dtype=torch.float32)
                states.append(state)
                starts.append(sample)
                strong_onsets[sample] = onset
                if onset.any():
                    onset_groups.extend([onset, quiet.clone()])
                for frequency in np.flatnonzero(new & ~current):
                    onsets.append((sample, int(frequency)))
                for frequency in np.flatnonzero(current & ~new):
                    offsets.append((sample, int(frequency)))
                current = new
                start_t += rng.uniform(self.constant_range[0], self.constant_range[1])

            strong_frames = torch.zeros(n, len(self.freqs), dtype=torch.float32)
            for left, right, state in zip(starts, starts[1:] + [n], states):
                strong_frames[left:right] = state

            waveform = (oscillators * strong_frames).sum(dim=-1)
            waveform += noise_std * torch.randn(n, generator=noise_rng)
            self.examples.append((
                waveform[None], strong_frames, torch.stack(states),
                strong_onsets, torch.stack(onset_groups),
            ))
            self.change_frames.append(torch.tensor(starts))
            self.onsets.append(onsets)
            self.offsets.append(offsets)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def collate_multipitch(batch):
    """Return the five batched tensors, then weak_frame_lengths/weak_onset_lengths.

    Waveforms and strong targets are stacked. Each weak target sequence is
    padded separately with zeros. Use its own length tensor to ignore padding;
    actual all-zero quiet targets remain part of the valid sequences.
    """
    waveforms, strong_frames, weak_frames, strong_onsets, weak_onsets = zip(*batch)
    return (
        torch.stack(waveforms), torch.stack(strong_frames),
        pad_sequence(weak_frames, batch_first=True), torch.stack(strong_onsets),
        pad_sequence(weak_onsets, batch_first=True),
        torch.tensor([len(y) for y in weak_frames]),
        torch.tensor([len(y) for y in weak_onsets]),
    )
