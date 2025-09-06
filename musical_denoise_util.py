import numpy as np
import mlx.core as mx
import matplotlib.pyplot as plt
from diffusion_model import VAE, LatentUNet, PRED_CLAMPING, LinearNoiseScheduler, VAE_LATENT_SPACE_SIZE

def add_biases(recon, chord=0.0, key_sig=1.0):
    assert recon.shape[0] == 49 and recon.shape[1] == 64
    
    result = np.array(recon)
    min = np.min(result)
    max = np.max(result)
    phi = 0.5*(max - min)
    mid = 0.5*(max + min)
    
    scale = [1, 2, 4, 6, 7, 9, 11]
    not_scale = [0, 3, 5, 8, 10]
    
    for t in range(64):
        
        # First do key signature bias, subtract key_sig from each note which is not on the scale
        for n in not_scale:
            result[n, t] -= phi*key_sig
            result[n + 12, t] -= phi*key_sig
            result[n + 24, t] -= phi*key_sig
            result[n + 36, t] -= phi*key_sig

        # Next do chord bias
        for n in range(12, 42):
            if recon[n, t] > mid:
                result[n + 1, t] -= chord*phi
                if ((n + 3) % 12) in scale:
                    result[n + 3, t] += chord*phi
                else:
                    result[n + 4, t] += chord*phi
                result[n + 5, t] += chord*phi
                result[n + 7, t] += chord*phi

    return result

# Upscale a piano roll along time by an integer scale factor
def upscale_piano_roll(x, factor):
    old_length = x.shape[1]
    new_length = old_length*factor
    assert new_length > old_length

    out = np.zeros((49, new_length))
    
    for i in range(old_length):
        idx = i * factor
        out[:, idx:idx+factor] = x[:, i:i+1]

    return out



# Downscale a piano roll along time by summing the note potentials of groups
def downscale_cumulative(x, factor):
    old_length = x.shape[1]
    assert old_length % factor == 0
    new_length = old_length // factor
    
    out = np.zeros((49, new_length))
    
    for i in range(new_length):
        idx = i * factor
        out[:, i] = np.sum(x[:, idx:idx+factor], axis=1)

    return out


def filter_music_loudest(recon, count):
    reshaped = recon
    
    reshaped = add_biases(reshaped)
    
    strongest_img = np.zeros(recon.shape)
    
    for t in range(recon.shape[1]):  # For each column (time step)
        column = reshaped[:, t]
        
        hottest_indices = np.argsort(column)[-count:]
        for idx in hottest_indices:
            strongest_img[idx, t] = 1
    
    return strongest_img


def detect_resolution(piano_roll, maj=False, min=False):
    """
    Vectorised convolution-based loop resolution detection for a piano roll (49, 64).
    """
    assert piano_roll.shape[0] == 49, "Expected 49-note piano roll"
    
    one_octave = np.array((12, piano_roll.shape[1]))
    one_octave = piano_roll[0:12] + piano_roll[12:24] + piano_roll[24:36] + piano_roll[36:48]
    one_octave[0] += piano_roll[48]
    one_octave = one_octave > 0

    # Original dissonance kernel
    kernel = np.array([
        0,    # Tonic
        1,    # Yuck
        0.1,  # 2nd
        0,    # Minor Third
        0,    # Major Third
        0,    # Fourth
        1,    # Tritone
        0,    # Fifth
        0.1,  # Minor 6th
        0.1,  # Major 6th
        0.1,  # Minor 7th
        0.1,  # Major 7th
    ])[::-1]  # Reverse kernel

    pad_above = len(kernel) - 1

    # Pad above the piano roll
    padded_roll = np.pad(one_octave, ((0, pad_above), (0, 0)), mode="constant")

    # Perform convolution across pitch dimension for all timesteps
    # Result shape: (n_positions, timesteps)
    conv_results = np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode="valid"),
        axis=0,
        arr=padded_roll
    ) * one_octave
    
    # Sum kernel responses per timestep
    tension_scores = conv_results.sum(axis=0)

    # Bias for tonic, 4th, and 5th bass notes
    tonic = 2 if maj else (11 if min else None)
    if tonic is not None:
        tonic_pc = tonic % 12
        bass_notes = np.argmax(piano_roll > 0, axis=0)
        bass_pc = bass_notes % 12
        bias_mask = np.isin(bass_pc, [(tonic_pc + 5) % 12, (tonic_pc + 7) % 12])
        tension_scores[bias_mask] *= 0.8
        bias_mask = np.isin(bass_pc, [tonic_pc])
        tension_scores[bias_mask] *= 0.6

    # Find largest drop in tension
    deltas = np.diff(tension_scores, prepend=tension_scores[-1])
    resolution_idx = np.argmin(deltas)

    return resolution_idx, tension_scores


def create_piano_rolls(recon, recon_m1=None, maj=False, min=False, resolve=True):
    if recon_m1 is None:
        recon_m1 = recon
   
    bass = upscale_piano_roll(downscale_cumulative(recon, 8), 8)
    minim = np.min(bass)
    bass[12:49, :] = np.full((37, 64), minim)

    chords = upscale_piano_roll(downscale_cumulative(recon, 4), 4)
    minim = np.min(chords)
    chords[0:12, :] = np.full((12, 64), minim)

    minim = np.min(recon_m1)
    piano = np.full((49,64), minim)
    piano[12:49, :] = recon_m1[12:49, :]
    
    chords = filter_music_loudest(chords, 3)
    piano = filter_music_loudest(piano, 3)
    bass = filter_music_loudest(bass, 1)
    
    if resolve:
        notes = piano + bass
        resolution_idx, tension_scores = detect_resolution(notes, maj, min)
        resolution_idx = (resolution_idx // 8) * 8
        bass = np.roll(bass, -resolution_idx, axis=1)
        chords = np.roll(chords, -resolution_idx, axis=1)
        piano = np.roll(piano, -resolution_idx, axis=1)

    return bass, chords, piano




def create_start_point_with_chords(chords):
    chord_count = len(chords)
    factor = 64 // chord_count
    if 64 % factor != 0:
        print(chord_count, factor, 64 % factor)
        raise ValueError("Number of chords must be divisor of 64")
    
    img = np.zeros((49, chord_count))
    
    for t, chord in enumerate(chords):
        for n in chord:
            img[n, t] = 1
    
    return upscale_piano_roll(img, factor)


F_MAJ = [2, 26, 30, 33]
G_MIN = [4, 28, 31, 35]
A_MIN = [6, 30, 33, 37]
Bb_MAJ = [7, 19, 23, 26]
C_MAJ = [9, 21, 25, 28]
D_MIN = [11, 23, 26, 30]
E_DIM = [1, 25, 28, 31]

MAJOR_START_IMAGES = [
    create_start_point_with_chords([F_MAJ, C_MAJ, F_MAJ, C_MAJ, F_MAJ, C_MAJ, F_MAJ, C_MAJ]),
    create_start_point_with_chords([F_MAJ, Bb_MAJ, F_MAJ, Bb_MAJ, F_MAJ, Bb_MAJ, F_MAJ, Bb_MAJ]),
    create_start_point_with_chords([F_MAJ, C_MAJ, D_MIN, Bb_MAJ, F_MAJ, C_MAJ, D_MIN, Bb_MAJ]),
    create_start_point_with_chords([F_MAJ, G_MIN, Bb_MAJ, C_MAJ, F_MAJ, G_MIN, Bb_MAJ, C_MAJ]),
    create_start_point_with_chords([Bb_MAJ, Bb_MAJ, F_MAJ, D_MIN, Bb_MAJ, Bb_MAJ, F_MAJ, D_MIN]),
]

MINOR_START_IMAGES = [
    create_start_point_with_chords([D_MIN, A_MIN, D_MIN, A_MIN, D_MIN, A_MIN, D_MIN, A_MIN]),
    create_start_point_with_chords([D_MIN, G_MIN, D_MIN, G_MIN, D_MIN, G_MIN, D_MIN, G_MIN]),
    create_start_point_with_chords([D_MIN, A_MIN, F_MAJ, G_MIN, D_MIN, A_MIN, F_MAJ, G_MIN]),
    create_start_point_with_chords([D_MIN, F_MAJ, G_MIN, A_MIN, D_MIN, F_MAJ, G_MIN, A_MIN]),
    create_start_point_with_chords([G_MIN, G_MIN, D_MIN, C_MAJ, G_MIN, G_MIN, D_MIN, C_MAJ]),
]

MAJOR_GROOVE_START_IMAGE = create_start_point_with_chords([F_MAJ])
MINOR_GROOVE_START_IMAGE = create_start_point_with_chords([D_MIN])

def encode_vector(vae, start_image):
    z, _ = vae.encode(mx.array(start_image).reshape(1, 49, 64, 1))
    return z

def encode_vectors(vae, start_images):
    result = []
    for img in start_images:
        z = encode_vector(vae, img)
        result.append(z)
    return result


class StartVectors:
    def __init__(self, vae):
        self.major_theme_vectors = encode_vectors(vae, MAJOR_START_IMAGES)
        self.minor_theme_vectors = encode_vectors(vae, MINOR_START_IMAGES)
        self.major_groove_vector = encode_vector(vae, MAJOR_GROOVE_START_IMAGE)
        self.minor_groove_vector = encode_vector(vae, MINOR_GROOVE_START_IMAGE)

    def get_theme_vector(self, seed, maj, min):
        if maj:
            idx = seed % len(self.major_theme_vectors)
            return self.major_theme_vectors[idx]
        elif min:
            idx = seed % len(self.minor_theme_vectors)
            return self.minor_theme_vectors[idx]
        return mx.zeros(self.major_groove_vector.shape)

    def get_groove_vector(self, seed, maj, min):
        if maj:
            return self.major_groove_vector
        elif min:
            return self.minor_groove_vector
        return mx.zeros(self.major_groove_vector.shape)






MAJ_NOTE_NAMES = ['F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E']
MIN_NOTE_NAMES = ['D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'C', 'C#']





class Denoiser:
    def __init__(self, vae, unet, text_encoder):
        self.vae = vae
        self.unet = unet
        self.text_encoder = text_encoder
        self.cfg = 0
        self.steps = 5
        self.denoise_time = 850
        self.resolve = True
        self.set_prompt("")
        self.set_major("C")
        self.set_seed(0)
        self.start_vectors = StartVectors(vae)
        self.bass = np.zeros((49, 64))
        self.chords = np.zeros((49, 64))
        self.lead = np.zeros((49, 64))

    def clear_embed(self):
        self._emb = None
        self.full_prompt = "Not embedded"
    
    def set_seed(self, seed):
        self._seed = seed
        self._prng = mx.random.key(seed)

    def set_chromatic(self, root):
        if root in MAJ_NOTE_NAMES:
            self._key_sig = ""
            self.key = "Chromatic"
            self.offset = MAJ_NOTE_NAMES.index(root)
            self._major = False
            self._minor = False
            self.clear_embed()

    def set_major(self, root):
        if root in MAJ_NOTE_NAMES:
            self._key_sig = f", in {root} Maj"
            self.key = f"{root} Major"
            self.offset = MAJ_NOTE_NAMES.index(root)
            self._major = True
            self._minor = False
            self.clear_embed()

    def set_minor(self, root):
        if root in MIN_NOTE_NAMES:
            self._key_sig = f", in {root} min"
            self.key = f"{root} minor"
            self.offset = MIN_NOTE_NAMES.index(root)
            self._major = False
            self._minor = True
            self.clear_embed()

    def set_prompt(self, prompt):
        self._prompt = prompt
        self.clear_embed()

    def embed(self):
        if self._emb is None:
            self.full_prompt = self._prompt + self._key_sig
            vec = self.text_encoder.encode(self.full_prompt)
            self._emb = mx.array(vec, dtype=mx.float32)[None, ...]
            mx.eval(self._emb)

    def gen_theme(self):
        self.embed()
        z0 = self.start_vectors.get_theme_vector(self._seed, self._major, self._minor)
        self.gen(z0)
    
    def gen_groove(self):
        self.embed()
        z0 = self.start_vectors.get_groove_vector(self._seed, self._major, self._minor)
        self.gen(z0, self.denoise_time // 3)
    
    def gen(self, z0, dn=None):
        st = 999
        if self._major or self._minor:
            st = dn if dn is not None else self.denoise_time
        gen = self.unet.denoise_loop(
            self._emb,
            key=self._prng,
            z0=z0,
            time_start=st,
            steps=self.steps,
            cfg=self.cfg
        )
        z_list = [z for z, _ in gen]
        r = np.array(self.vae.decode(z_list[-1]).reshape((49, 64)))
        rm1 = np.array(self.vae.decode(z_list[-2]).reshape((49, 64)))
        mx.eval(r, rm1)
        bass, chords, lead = create_piano_rolls(r, rm1, self._major, self._minor, self.resolve)
        self.bass = bass
        self.chords = chords
        self.lead = lead
    
    def __str__(self):
        return f"{self.full_prompt} - {self._seed} steps:{self.steps} cfg:{self.cfg} n:{self.denoise_time}" + (" resolved" if self.resolve else "")
