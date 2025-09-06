# generate_sample.py
import os
import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt
import time
from sentence_transformers import SentenceTransformer
from diffusion_model import VAE, LatentUNet, PRED_CLAMPING, LinearNoiseScheduler, VAE_LATENT_SPACE_SIZE
from datetime import datetime
import argparse
import random
import pretty_midi
from musical_denoise_util import Denoiser










IMAGE_WIDTH=49
IMAGE_HEIGHT=64
CFG=15
START_POINT_SCALE=750
MAJ_NOTE_NAMES = ['F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E']
MIN_NOTE_NAMES = ['D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'C', 'C#']
VAE_MODEL = "vae_model.safetensors"
UNET_MODEL = "unet_model.safetensors"
ONLINE_TEXT_MODEL = "all-MiniLM-L6-v2"
LOCAL_TEXT_MODEL = "text_model"










def add_reconstruction_to_midi(midi_file, filtered_recon, velocity=80, program=52,
                               instrument_name="Reconstructed Piano", tempo_bpm=120):
    """
    Add reconstruction notes to a pretty_midi file using the 4 strongest notes per time step
    (1 from bass region, 3 from melody region).
    
    Args:
        midi_file: pretty_midi.PrettyMIDI object to modify
        reconstruction: numpy array of shape (49, 64) representing the reconstruction
        note_duration: Duration of each reconstructed note in seconds (default: 0.5)
        velocity: MIDI velocity for reconstructed notes (0-127, default: 80)
        instrument_name: Name for the reconstruction instrument (default: "Reconstructed Piano")
        tempo_bpm: Tempo in BPM for timing calculations (default: 120)
        
    Returns:
        pretty_midi.PrettyMIDI: Modified MIDI file with reconstruction notes added
        
    Note:
        - Bottom note (pixel 0) corresponds to D#1 (MIDI note 27)
        - Top note (pixel 48) corresponds to C5 (MIDI note 75)
        - Time step duration is calculated as (60/tempo_bpm) * (4/64) for 16th notes
    """
    
    # Create new instrument for reconstruction
    reconstruction_instrument = pretty_midi.Instrument(
        program=program,
        is_drum=False,
        name=instrument_name
    )
    
    recon = np.array(filtered_recon)
    
    # Calculate timing parameters
    # Assuming 64 time steps represent 4 beats (16th note resolution)
    beat_duration = 60.0 / tempo_bpm  # Duration of one beat in seconds
    time_step_duration = beat_duration  # 16th note duration
    
    # Convert reconstruction to MIDI notes
    for pitch_idx in range(49):  # Each pitch
        for time_idx in range(64):  # Each time step
            if recon[pitch_idx, time_idx] > 0:
                # Convert pitch index to MIDI note number
                # pitch_idx 0 = D#1 (MIDI 27), pitch_idx 48 = C5 (MIDI 75)
                midi_note_number = 27 + pitch_idx
                
                # Calculate note timing
                start_time = time_idx * time_step_duration
                
                # Extend the note for the length of contiguous activations
                end_idx = time_idx
                while end_idx < 64 and recon[pitch_idx, end_idx] > 0:
                    # Remove the activation so that we don't add the note again
                    recon[pitch_idx, end_idx] = 0
                    end_idx += 1
                
                end_time = end_idx * time_step_duration
                
                # Create MIDI note
                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=midi_note_number,
                    start=start_time,
                    end=end_time
                )
                
                reconstruction_instrument.notes.append(note)
    
    # Add reconstruction instrument to MIDI file
    midi_file.instruments.append(reconstruction_instrument)
    
    return midi_file










# Create a new MIDI file containing only the reconstruction notes.
def create_reconstruction_midi(denoiser, filename="reconstruction", trackname="track", velocity=80, tempo_bpm=120):
    
    bass = denoiser.bass
    chords = denoiser.chords
    piano = denoiser.lead
    offset = denoiser.offset
    
    # Create empty MIDI file
    midi_file = pretty_midi.PrettyMIDI(initial_tempo=tempo_bpm)
    
    # Add reconstruction to empty MIDI
    midi_with_reconstruction = add_reconstruction_to_midi(
        midi_file, bass, velocity, 63,
        f"{trackname}-bass", tempo_bpm
    )
    midi_with_reconstruction = add_reconstruction_to_midi(
        midi_file, chords, velocity, 63,
        f"{trackname}-chords", tempo_bpm
    )
    midi_with_reconstruction = add_reconstruction_to_midi(
        midi_file, piano, velocity, 1,
        f"{trackname}-piano", tempo_bpm
    )
    
    # Add offsets to put everything in key
    #print(midi_with_reconstruction.instruments[0].notes)
    for n in range(len(midi_with_reconstruction.instruments[0].notes)):
        # With the bass note, we want to make sure it stays inside the bass register
        x = midi_with_reconstruction.instruments[0].notes[n].pitch + offset
        if x > 38:
            x -= 12
        midi_with_reconstruction.instruments[0].notes[n].pitch = x
    #print(midi_with_reconstruction.instruments[0].notes)

    for i in range(1, 3):
        inst = midi_with_reconstruction.instruments[i]
        for n in range(len(inst.notes)):
            inst.notes[n].pitch += offset
            

    # Save to file
    midi_with_reconstruction.write(f"{filename}.mid")
    print(f"Reconstruction MIDI saved as: {filename}.mid")
    
    return midi_with_reconstruction










def make_plot(denoiser, filename):
    fig, axs = plt.subplots(2, 1, figsize=(10, 9))
    
    title = f"{denoiser}"
    
    axs[0].imshow(denoiser.lead + denoiser.bass, aspect='auto', origin='lower')
    axs[0].set_title(f"{title} - Bass + piano")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Note")
    axs[0].set_xticks([0, 8, 16, 24, 32, 40, 48, 56])

    axs[1].imshow(denoiser.chords, aspect='auto', origin='lower')
    axs[1].set_title(f"{title} - Chords")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Note")
    axs[1].set_xticks([0, 8, 16, 24, 32, 40, 48, 56])

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()










def run_script(filename, denoiser):
    try:
        print(f"Running commands from {filename}")
        with open(filename, "r") as script_file:
            script_lines = script_file.readlines()
    except:
        print(f"Error reading file {filename}")
        raise
    
    timestamp = datetime.now().isoformat(timespec='seconds').replace(":", "-")
    os.makedirs(f"output samples/{timestamp}", exist_ok=True)
    img_idx = 0
    
    for line in script_lines:
        tokens = line.rstrip().split(' ', 1)

        command, parameter = (tokens[0], tokens[1] if len(tokens) > 1 else None)
        
        def check_parameter():
            if parameter is None:
                raise RuntimeError(f"Error: {command} expects parameter")
                
        
        if command == "":
            # Empty line, skip it
            pass
        
        elif command[0] == "#":
            # This line is commented out, skip it
            pass
        
        elif command == "seed":
            check_parameter()
            seed = int(parameter)
            if seed == -1:
                seed = random.randint(0, 2**32 - 1)
            denoiser.set_seed(seed)
        
        elif command == "cfg":
            check_parameter()
            denoiser.cfg = float(parameter)
        
        elif command == "steps":
            check_parameter()
            denoiser.steps = int(parameter)
        
        elif command == "prompt":
            check_parameter()
            denoiser.set_prompt(parameter)
        
        elif command == "strength":
            check_parameter()
            denoiser.denoise_time = int(parameter)
        
        elif command == "major":
            check_parameter()
            denoiser.set_major(parameter)
        
        elif command == "minor":
            check_parameter()
            denoiser.set_minor(parameter)
        
        elif command == "resolve":
            denoiser.resolve = True
        
        elif command == "dontresolve":
            denoiser.resolve = False
        
        elif command == "groove":
            denoiser.embed()
            print(f"Generating groove {denoiser}")
            output_filename = f"output samples/{timestamp}/sample-output-{img_idx:04d}"
            img_idx += 1
            denoiser.gen_groove()
        
        elif command == "theme":
            denoiser.embed()
            print(f"Generating theme {denoiser}")
            output_filename = f"output samples/{timestamp}/sample-output-{img_idx:04d}"
            img_idx += 1
            denoiser.gen_theme()
        
        elif command == "plot":
            print(f"Plotting {denoiser}")
            make_plot(denoiser, f"{output_filename}.png")

        elif command == "midi":
            print(f"MIDI {denoiser}")
            trackname = f"{denoiser}"
            midi = create_reconstruction_midi(denoiser, filename=output_filename, trackname=trackname)

        else:
            print(f"Unknown command {command}")









os.makedirs("output samples", exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument("-steps", type=int, help="Specify a step count other than 25 steps")
parser.add_argument("-count", type=int, help="Specify the number of images to generate from the prompt")
parser.add_argument("-seed", type=int, help="Specify the seed for the random number generator")
parser.add_argument("-cfg", type=float, help="Specify classifer free guidance vector strength")
parser.add_argument("-maj", type=str, choices=MAJ_NOTE_NAMES, help="Choose a major scale")
parser.add_argument("-min", type=str, choices=MIN_NOTE_NAMES, help="Choose a minor scale")
parser.add_argument("-n", type=int, help="Choose a scale factor for denoising from the starting image")
parser.add_argument("-plot", action="store_true", help="Produce a plot of the decoded latents and the predicted noise at each step")
parser.add_argument("-groove", action="store_true", help="Produce a groove from a groove start vector")
parser.add_argument("prompt", nargs=argparse.REMAINDER, help="The prompt follows at the end of the command")
parser.add_argument("-script", type=str, help="Ignore the command line options and run this generation script instead")
args = parser.parse_args()

# Load models
print("Loading VAE")
vae = VAE()
vae.load_weights(VAE_MODEL)
vae.eval()

print("Loading UNet")
unet = LatentUNet()
unet.load_weights(UNET_MODEL)
unet.eval()

print("Loading SentenceTransformer")
if os.path.exists(LOCAL_TEXT_MODEL):
    print("Attempting to load local model")
    text_encoder = SentenceTransformer(LOCAL_TEXT_MODEL)
    print("Successfully loaded local model")
else:
    text_encoder = SentenceTransformer(ONLINE_TEXT_MODEL)
    print("Downloaded online model and saved locally")
    text_encoder.save(LOCAL_TEXT_MODEL)


print("Initialising Denoiser")
denoiser = Denoiser(vae, unet, text_encoder)


if args.script is not None:
    run_script(args.script, denoiser)

else:
    num_steps = args.steps if args.steps else 10
    prompt = " ".join(args.prompt)
    output_count = args.count if args.count else 1
    if args.cfg is not None:
        CFG = args.cfg

    if args.seed is not None:
        random_seed = args.seed
    else:
        random_seed = random.randint(0, 4294967295)

    if args.n is not None:
        START_POINT_SCALE=args.n

    timestamp = datetime.now().isoformat(timespec='seconds').replace(":", "-")

    denoiser.cfg = CFG
    denoiser.steps = num_steps
    denoiser.denoise_time = START_POINT_SCALE

    if args.maj is None and args.min is None:
        tonic = (random_seed // 2) % 12
        if random_seed % 2 == 0:
            args.maj = MAJ_NOTE_NAMES[tonic]
        else:
            args.min = MIN_NOTE_NAMES[tonic]

    if args.maj is not None:
        denoiser.set_major(args.maj)
    if args.min is not None:
        denoiser.set_minor(args.min)

    img_idx = 0
    previous_text_embedding = None

    # Encode prompt
    print(f"Let's begin")
    start_time = time.time()
    denoiser.set_prompt(prompt)
    denoiser.embed()
    text_time = time.time() - start_time
    print(f"Prompt: {prompt} -- encoding time: {text_time:.3f}s")

    for i in range(output_count):
        # Generate from noise
        start_denoise = time.time()
        
        denoiser.set_seed(random_seed)
        if args.groove:
            denoiser.gen_groove()
        else:
            denoiser.gen_theme()
        
        denoise_time = time.time() - start_denoise

        filename = f"output samples/sample-output-{timestamp}-{img_idx}"
        trackname = f"n{i}-{random_seed} cfg:{CFG} steps:{num_steps} n:{denoiser.denoise_time}"
        midi = create_reconstruction_midi(denoiser, filename=filename, trackname=trackname)
        note_count = len(midi.instruments[0].notes)

        # Plot
        if args.plot:
            make_plot(denoiser, f"{filename}.png")

        # Stats
        print(f"Denoising time: {denoise_time:.3f}s")
        print(f"Total generation time: {text_time + denoise_time:.3f}s")
        
        # Increment the seed and index so that successive generations are different
        img_idx += 1
        random_seed += 1
