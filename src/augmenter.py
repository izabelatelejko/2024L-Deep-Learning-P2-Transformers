"""Augmenter for data augmentation."""

from audiomentations import TimeStretch, PitchShift, Shift, OneOf, AddColorNoise


def generate_augmenter():
    """Generate audio augmenter."""
    return OneOf(
        transforms=[
            AddColorNoise(min_f_decay=0, max_f_decay=0, p=1),  # white noise
            AddColorNoise(min_f_decay=-3.01, max_f_decay=-3.01, p=1),  # pink noise
            TimeStretch(min_rate=0.8, max_rate=0.95, leave_length_unchanged=True, p=1),
            TimeStretch(min_rate=1.05, max_rate=1.25, leave_length_unchanged=True, p=1),
            PitchShift(min_semitones=-5, max_semitones=-1, p=1),
            PitchShift(min_semitones=1, max_semitones=5, p=1),
            Shift(p=1, min_shift=-0.2, max_shift=-0.05, rollover=False),
            Shift(p=1, min_shift=0.05, max_shift=0.2, rollover=False),
        ]
    )
