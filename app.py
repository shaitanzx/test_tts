# coding=utf-8
# Qwen3-TTS Gradio 
# Supports: Voice Design, Voice Clone (Base), TTS (CustomVoice)
#import subprocess
#subprocess.run('pip install flash-attn==2.7.4.post1', shell=True)
import os
ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ["HF_HOME"] = os.path.join(ROOT, "model")
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
import gradio as gr
import numpy as np
import torch
import shutil
import librosa
LIBROSA_AVAILABLE = True
import parselmouth
PARSELMOUTH_AVAILABLE = True
from huggingface_hub import snapshot_download
import argparse
import time
import io
import soundfile as sf
import subprocess
#import sys
#from pathlib import Path
#sys.path.insert(0, str(Path(__file__).parent.parent))
#from huggingface_hub import login
HF_TOKEN = os.environ.get('HF_TOKEN')
#login(token=HF_TOKEN)

# Global model holders - keyed by (model_type, model_size)
loaded_models = {}
reference_playing_state = {"is_playing": False, "current_file": None}
# Model size options
MODEL_SIZES = ["0.6B", "1.7B"]
REFERENCE = sorted([f for f in os.listdir("reference") if f.lower().endswith(('.wav', '.mp3'))])
REF_DIR = os.path.join(ROOT, "reference")
os.makedirs(os.path.join(ROOT, "custom"), exist_ok=True)
CUSTOM_VOICE = sorted([f for f in os.listdir("custom") if f.lower().endswith(('.wav', '.mp3'))]) or [""]
CUSTOM_DIR = os.path.join(ROOT, "custom")
OUTPUT_DIR = os.path.join(ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
def get_device() -> torch.device:
    """
    Автоматически определяет доступное устройство:
    CUDA → MPS → CPU (в порядке проверки)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
DEVICE=get_device()
def change_voice_mode(voice_mode):
    return (
        gr.update(visible=(voice_mode == "predefined")),
        gr.update(visible=(voice_mode == "custom"))
    )
def read_text_for_audio(audio_path):
    """
    Ищет файл с тем же именем, что и audio_path, но с расширением .txt или .lab.
    Возвращает содержимое первого найденного файла или пустую строку.
    """
    base = os.path.splitext(audio_path)[0]
    for ext in ['.txt', '.lab']:
        text_path = base + ext
        if os.path.exists(text_path):
            try:
                with open(text_path, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except Exception as e:
                print(f"Ошибка при чтении {text_path}: {e}")
                return ""
    return ""  # ни один файл не найден
def select_ref_audio(audio_path):
    return read_text_for_audio(os.path.join(REF_DIR, audio_path))
def select_custom_audio(audio_path):
    return read_text_for_audio(os.path.join(CUSTOM_DIR, audio_path))

REFERENCE_TXT = read_text_for_audio(os.path.join(REF_DIR, REFERENCE[0]))
CUSTOM_TXT = read_text_for_audio(os.path.join(CUSTOM_DIR, CUSTOM_VOICE[0]))
def toggle_voice_audio(selected_file, voice_mode):
    global reference_playing_state
    if voice_mode == "reference":
        base_path = REF_DIR
    else: 
        base_path = CUSTOM_DIR
    
    file_path = os.path.join(base_path,selected_file)
    current_key = f"{voice_mode}_{selected_file}"
    
    if reference_playing_state["is_playing"] and reference_playing_state["current_key"] == current_key:
        reference_playing_state = {"is_playing": False, "current_key": None}
        gr.Info(f"⏸️ Stopped: {selected_file}")
        return None

    reference_playing_state = {"is_playing": True, "current_key": current_key}
    gr.Info(f"🎵 Playing: {selected_file}")
    
    return str(file_path)
def sanitize_filename(filename):
    """
    Removes potentially unsafe characters and path components from a filename
    to make it safe for use in file paths. Replaces unsafe sequences with underscores.

    Args:
        filename: The original filename string.

    Returns:
        A sanitized filename string, ensuring it's not empty and reasonably short.
    """
    if not filename:
        # Generate a unique name if the input is empty.
        return f"unnamed_file_{uuid.uuid4().hex[:8]}"

    # Remove directory separators and leading/trailing whitespace.
    base_filename = os.path.basename(filename).strip()
    if not base_filename:
        return f"empty_basename_{uuid.uuid4().hex[:8]}"

    # Define a set of allowed characters (alphanumeric, underscore, hyphen, dot, space).
    # Spaces will be replaced by underscores later.
    safe_chars = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._- "
    )
    sanitized_list = []
    last_char_was_underscore = False

    for char in base_filename:
        if char in safe_chars:
            # Replace spaces with underscores.
            sanitized_list.append("_" if char == " " else char)
            last_char_was_underscore = char == " "
        elif not last_char_was_underscore:
            # Replace any disallowed character sequence with a single underscore.
            sanitized_list.append("_")
            last_char_was_underscore = True

    sanitized = "".join(sanitized_list).strip("_")

    # Prevent names starting with multiple dots or consisting only of dots/underscores.
    if not sanitized or sanitized.lstrip("._") == "":
        return f"sanitized_file_{uuid.uuid4().hex[:8]}"

    # Limit filename length (e.g., 100 characters), preserving the extension.
    max_len = 100
    if len(sanitized) > max_len:
        name_part, ext_part = os.path.splitext(sanitized)
        # Ensure extension is not overly long itself; common extensions are short.
        ext_part = ext_part[:10]  # Limit extension length just in case.
        name_part = name_part[
            : max_len - len(ext_part) - 1
        ]  # -1 for the dot if ext exists
        sanitized = name_part + ext_part
        print(
            f"Original filename '{base_filename}' was truncated to '{sanitized}' due to length limits."
        )

    if not sanitized:  # Should not happen with previous checks, but as a failsafe.
        return f"final_fallback_name_{uuid.uuid4().hex[:8]}"

    return sanitized
def upload_reference_audio_endpoint(files):
    # upload reference audio
    ref_path = CUSTOM_DIR
    uploaded_filenames = []
    errors = []
    for file_info in files:
        if not file_info:
            continue
            
        # Extract filename from Gradio file object
        filename = os.path.basename(file_info)
        safe_filename = sanitize_filename(filename)
        destination_path = os.path.join(ref_path,safe_filename)
        try:
            if os.path.exists(destination_path):
                print(f"File '{safe_filename}' already exists.")
                uploaded_filenames.append(safe_filename)
                continue
            
            # Copy file
            shutil.copy2(file_info, destination_path)
            print(f"Saved uploaded file to: {destination_path}")
            uploaded_filenames.append(safe_filename)
                
        except Exception as e:
            errors.append({"filename": filename, "error": str(e)})
    
    all_files = sorted([f for f in os.listdir("custom") if f.lower().endswith(('.wav', '.mp3','.lab', '.txt'))]) or [""]
    return {
        "message": f"Processed {len(files)} file(s)",
        "uploaded_files": uploaded_filenames,
        "all_reference_files": all_files,
        "errors": errors
    }

def on_reference_upload(files):
    try:
        result =  upload_reference_audio_endpoint(files)
        all_files = result.get("all_reference_files", [])
        uploaded_files = result.get("uploaded_files", [])

        updated_options = sorted([f for f in os.listdir("custom") if f.lower().endswith(('.wav', '.mp3'))])
        default_selection = None
        text=None
        if uploaded_files:
            for file in reversed(uploaded_files):
                if file.lower().endswith(('.wav', '.mp3')):
                    default_selection = file
                    break
            if default_selection:
                return gr.update(choices=updated_options,value=default_selection),gr.update(value=select_custom_audio(default_selection))
            else:
                base_name = os.path.splitext(uploaded_files[-1])[0]
                for ext in ('.wav', '.mp3'):
                    audio_file = f"{base_name}{ext}"
                    if audio_file in all_files:
                        default_selection = audio_file
                        break
                if default_selection:
                    return gr.update(choices=updated_options,value=default_selection),gr.update(value=select_custom_audio(default_selection))
                else:
                    return gr.update(),gr.update()   
        else:         
            return gr.update(), gr.update()
            
    except Exception as e:
        print(f"Error in reference upload: {e}")
        return sorted([f for f in os.listdir("custom") if f.lower().endswith(('.wav', '.mp3'))]) or [""]

def get_model_path(model_type: str, model_size: str) -> str:
    """Get model path based on type and size."""
    return snapshot_download(f"Qwen/Qwen3-TTS-12Hz-{model_size}-{model_type}")


def get_model(model_type: str, model_size: str):
    """Get or load a model by type and size."""
    global loaded_models
    key = (model_type, model_size)
    if key not in loaded_models:
        from qwen_tts import Qwen3TTSModel
        model_path = get_model_path(model_type, model_size)
        loaded_models[key] = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=DEVICE,
            dtype=torch.bfloat16,
            token=HF_TOKEN,
#           attn_implementation="flash_attention_2",
        )
    return loaded_models[key]


def _normalize_audio(wav, eps=1e-12, clip=True):
    """Normalize audio to float32 in [-1, 1] range."""
    x = np.asarray(wav)

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")

    if clip:
        y = np.clip(y, -1.0, 1.0)

    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)

    return y


def _audio_to_tuple(audio):
    """Convert Gradio audio input to (wav, sr) tuple."""
    if audio is None:
        return None

    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], int):
        sr, wav = audio
        wav = _normalize_audio(wav)
        return wav, int(sr)

    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        sr = int(audio["sampling_rate"])
        wav = _normalize_audio(audio["data"])
        return wav, sr

    return None


# Speaker and language choices for CustomVoice model
SPEAKERS = [
    "Aiden", "Dylan", "Eric", "Ono_anna", "Ryan", "Serena", "Sohee", "Uncle_fu", "Vivian"
]
LANGUAGES = ["Auto", "Chinese", "English", "Japanese", "Korean", "French", "German", "Spanish", "Portuguese", "Russian"]
def encode_audio(audio_array,sample_rate,output_format,target_sample_rate):
    """
    Encodes a NumPy audio array into the specified format (Opus or WAV) in memory.
    Can resample the audio to a target sample rate before encoding if specified.

    Args:
        audio_array: NumPy array containing audio data (expected as float32, range [-1, 1]).
        sample_rate: Sample rate of the input audio data.
        output_format: Desired output format ('opus', 'wav' or 'mp3').
        target_sample_rate: Optional target sample rate to resample to before encoding.

    Returns:
        Bytes object containing the encoded audio, or None if encoding fails.
    """
    if audio_array is None or audio_array.size == 0:
        #logger.warning("encode_audio received empty or None audio array.")
        return None,"encode_audio received empty or None audio array."

    # Ensure audio is float32 for consistent processing.
    if audio_array.dtype != np.float32:
        if np.issubdtype(audio_array.dtype, np.integer):
            max_val = np.iinfo(audio_array.dtype).max
            audio_array = audio_array.astype(np.float32) / max_val
        else:  # Fallback for other types, assuming they might be float64 or similar
            audio_array = audio_array.astype(np.float32)
        print(f"Converted audio array to float32 for encoding.")

    # Ensure audio is mono if it's (samples, 1)
    if audio_array.ndim == 2 and audio_array.shape[1] == 1:
        audio_array = audio_array.squeeze(axis=1)
        print(
            "Squeezed audio array from (samples, 1) to (samples,) for encoding."
        )
    elif (
        audio_array.ndim > 1
    ):  # Multi-channel not directly supported by simple encoding path, attempt to take first channel
        print(
            f"Multi-channel audio (shape: {audio_array.shape}) provided to encode_audio. Using only the first channel."
        )
        audio_array = audio_array[:, 0]

    # Resample if target_sample_rate is provided and different from current sample_rate
    if (
        target_sample_rate is not None
        and target_sample_rate != sample_rate
        and LIBROSA_AVAILABLE
    ):
        try:
            print(
                f"Resampling audio from {sample_rate}Hz to {target_sample_rate}Hz using Librosa."
            )
            audio_array = librosa.resample(
                y=audio_array, orig_sr=sample_rate, target_sr=target_sample_rate
            )
            sample_rate = (
                target_sample_rate  # Update sample_rate for subsequent encoding
            )
        except Exception as e_resample:
            print(
                f"Error resampling audio to {target_sample_rate}Hz: {e_resample}. Proceeding with original sample rate {sample_rate}.",
            )
    elif target_sample_rate is not None and target_sample_rate != sample_rate:
        print(
            f"Librosa not available. Cannot resample audio from {sample_rate}Hz to {target_sample_rate}Hz. "
            f"Proceeding with original sample rate for encoding."
        )

    start_time = time.time()
    output_buffer = io.BytesIO()
    try:
        audio_to_write = audio_array
        rate_to_write = sample_rate

        if output_format == "opus":
            OPUS_SUPPORTED_RATES = {8000, 12000, 16000, 24000, 48000}
            TARGET_OPUS_RATE = 48000  # Preferred Opus rate.

            if rate_to_write not in OPUS_SUPPORTED_RATES:
                if LIBROSA_AVAILABLE:
                    print(
                        f"Current sample rate {rate_to_write}Hz not directly supported by Opus. "
                        f"Resampling to {TARGET_OPUS_RATE}Hz using Librosa for Opus encoding."
                    )
                    audio_to_write = librosa.resample(
                        y=audio_array, orig_sr=rate_to_write, target_sr=TARGET_OPUS_RATE
                    )
                    rate_to_write = TARGET_OPUS_RATE
                else:
                    print(
                        f"Librosa not available. Cannot resample audio from {rate_to_write}Hz for Opus encoding. "
                        f"Opus encoding may fail or produce poor quality."
                    )
                    # Proceed with current rate, soundfile might handle it or fail.
            sf.write(
                output_buffer,
                audio_to_write,
                rate_to_write,
                format="ogg",
                subtype="opus",
            )

        elif output_format == "wav":
            # WAV typically uses int16 for broader compatibility.
            # Clip audio to [-1.0, 1.0] before converting to int16 to prevent overflow.
            audio_clipped = np.clip(audio_array, -1.0, 1.0)
            audio_int16 = (audio_clipped * 32767).astype(np.int16)
            audio_to_write = audio_int16  # Use the int16 version for WAV
            sf.write(
                output_buffer,
                audio_to_write,
                rate_to_write,
                format="wav",
                subtype="pcm_16",
            )

        elif output_format == "mp3":
            audio_clipped = np.clip(audio_array, -1.0, 1.0)
            audio_int16 = (audio_clipped * 32767).astype(np.int16)
            audio_segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1,
            )
            audio_segment.export(output_buffer, format="mp3")

        else:
            #logger.error(
            #    f"Unsupported output format requested for encoding: {output_format}"
            #)
            return None,f"Unsupported output format requested for encoding: {output_format}"

        encoded_bytes = output_buffer.getvalue()
        end_time = time.time()
        print(
            f"Encoded {len(encoded_bytes)} bytes to '{output_format}' at {rate_to_write}Hz in {end_time - start_time:.3f} seconds."
        )
        return encoded_bytes

    except ImportError as ie_sf:  # Specifically for soundfile import issues
        print(
            f"The 'soundfile' library or its dependency (libsndfile) is not installed or found. "
            f"Audio encoding/saving is not possible. Please install it. Error: {ie_sf}"
        )
        return None
    except Exception as e:
        print(f"Error encoding audio to '{output_format}': {e}")
        return None
def generate_voice_design(text, language, voice_description):
    """Generate speech using Voice Design model (1.7B only)."""
    if not text or not text.strip():
        return None, "Error: Text is required."
    if not voice_description or not voice_description.strip():
        return None, "Error: Voice description is required."

    try:
        start_time = time.time()
        tts = get_model("VoiceDesign", "1.7B")
        wavs, sr = tts.generate_voice_design(
            text=text.strip(),
            language=language,
            instruct=voice_description.strip(),
            non_streaming_mode=True,
            max_new_tokens=2048,
        )

        encoded_audio_bytes = encode_audio(
            audio_array=wavs[0],
            sample_rate=sr,
            output_format="wav",
            target_sample_rate=sr,  
            )
        
        if encoded_audio_bytes is None:
            return None, "Failed to encode audio to requested format."
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        suggested_filename_base = f"qwen3_vd_{timestamp_str}"
        file_name = f"{suggested_filename_base}.wav"
        file_path = os.path.join(OUTPUT_DIR, file_name)
        
        with open(file_path, "wb") as f:
            f.write(encoded_audio_bytes)
        
        generation_time = time.time() - start_time
        
        return str(file_path), f"✅ Audio generated successfully in {generation_time:.2f}s"
       




        #return (sr, wavs[0]), "Voice design generation completed successfully!"
    except Exception as e:
        return None, f"Error: {type(e).__name__}: {e}"


def generate_voice_clone(voice_mode_radio, clone_ref_audio_drop, clone_ref_text_drop, clone_xvector,  
        custom_ref_audio_drop, custom_ref_text_drop,custom_xvector,target_text, language, model_size):
    if voice_mode_radio == 'predefined':
        audio=clone_ref_audio_drop
        ref_text=clone_ref_text_drop
        use_xvector_only=clone_xvector
        path='reference'
    else:
        audio=custom_ref_audio_drop
        ref_text=custom_ref_text_drop
        use_xvector_only=custom_xvector
        path='custom'
    start_time = time.time()
    wav, sr = librosa.load(os.path.join(ROOT,path,audio), sr=None)
    ref_audio = {"sampling_rate": sr, "data": wav}


    """Generate speech using Base (Voice Clone) model."""
    if not target_text or not target_text.strip():
        return None, "Error: Target text is required."

    audio_tuple = _audio_to_tuple(ref_audio)
    if audio_tuple is None:
        return None, "Error: Reference audio is required."

    if not use_xvector_only and (not ref_text or not ref_text.strip()):
        return None, "Error: Reference text is required when 'Use x-vector only' is not enabled."

    try:
        tts = get_model("Base", model_size)
        wavs, sr = tts.generate_voice_clone(
            text=target_text.strip(),
            language=language,
            ref_audio=audio_tuple,
            ref_text=ref_text.strip() if ref_text else None,
            x_vector_only_mode=use_xvector_only,
            max_new_tokens=2048,
        )
        encoded_audio_bytes = encode_audio(
            audio_array=wavs[0],
            sample_rate=sr,
            output_format="wav",
            target_sample_rate=sr,  
            )
        
        if encoded_audio_bytes is None:
            return None, "Failed to encode audio to requested format."
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        suggested_filename_base = f"qwen3_base_{timestamp_str}"
        file_name = f"{suggested_filename_base}.wav"
        file_path = os.path.join(OUTPUT_DIR, file_name)
        
        with open(file_path, "wb") as f:
            f.write(encoded_audio_bytes)
        
        generation_time = time.time() - start_time
        
        return str(file_path), f"✅ Audio generated successfully in {generation_time:.2f}s"
        #return (sr, wavs[0]), "Voice clone generation completed successfully!"
    except Exception as e:
        return None, f"Error: {type(e).__name__}: {e}"


def generate_custom_voice(text, language, speaker, instruct, model_size):
    """Generate speech using CustomVoice model."""
    if not text or not text.strip():
        return None, "Error: Text is required."
    if not speaker:
        return None, "Error: Speaker is required."

    try:
        tts = get_model("CustomVoice", model_size)
        start_time = time.time()
        wavs, sr = tts.generate_custom_voice(
            text=text.strip(),
            language=language,
            speaker=speaker.lower().replace(" ", "_"),
            instruct=instruct.strip() if instruct else None,
            non_streaming_mode=True,
            max_new_tokens=2048,
        )
        encoded_audio_bytes = encode_audio(
            audio_array=wavs[0],
            sample_rate=sr,
            output_format="wav",
            target_sample_rate=sr,  
            )
        
        if encoded_audio_bytes is None:
            return None, "Failed to encode audio to requested format."
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        suggested_filename_base = f"qwen3_tts_{timestamp_str}"
        file_name = f"{suggested_filename_base}.wav"
        file_path = os.path.join(OUTPUT_DIR, file_name)
        
        with open(file_path, "wb") as f:
            f.write(encoded_audio_bytes)
        
        generation_time = time.time() - start_time
        
        return str(file_path), f"✅ Audio generated successfully in {generation_time:.2f}s"
        #return (sr, wavs[0]), "Voice clone generation completed successfully!"
    except Exception as e:
        return None, f"Error: {type(e).__name__}: {e}"
def post_process_gui():
    with gr.Row():                    
        with gr.Accordion("⚙️ Postprocessing", open=False):
            with gr.Row():
                post_output = gr.Audio(
                    label="Postprocessed Audio",
                    type="filepath",
                    interactive=True,
                    visible=False,
                    show_download_button=True
                    )
            with gr.Row():
                speed_factor_slider = gr.Slider(
                    minimum=0.25,
                    maximum=4.0,
                    value=1.0,
                    step=0.05,
                    label="Speed Factor",
                    interactive=True
                    )
            with gr.Row():
                silence_trimming = gr.Checkbox(
                    label="Silence Trimming",
                    value=False,
                    interactive=True
                    )
                internal_silence_fix = gr.Checkbox(
                    label="Internal Silence Fix",
                    value=False,
                    interactive=True
                    )
                unvoiced_removal = gr.Checkbox(
                    label="Unvoiced Removal",
                    value=False,
                    interactive=True
                    )
            with gr.Row():
                 post_btn = gr.Button("🎵 PostProcessing",visible=True,interactive=True)
    return post_btn, post_output, speed_factor_slider, silence_trimming, internal_silence_fix, unvoiced_removal 

def trim_lead_trail_silence(
    audio_array: np.ndarray,
    sample_rate: int,
    silence_threshold_db: float = -40.0,
    min_silence_duration_ms: int = 100,
    padding_ms: int = 50,
) -> np.ndarray:
    """
    Trims silence from the beginning and end of a NumPy audio array using a dB threshold.

    Args:
        audio_array: NumPy array (float32) of the audio.
        sample_rate: Sample rate of the audio.
        silence_threshold_db: Silence threshold in dBFS. Segments below this are considered silent.
        min_silence_duration_ms: Minimum duration of silence to be trimmed (ms).
        padding_ms: Padding to leave at the start/end after trimming (ms).

    Returns:
        Trimmed NumPy audio array. Returns original if no significant silence is found or on error.
    """
    if audio_array is None or audio_array.size == 0:
        return audio_array

    try:
        if not LIBROSA_AVAILABLE:
            print("Librosa not available, skipping silence trimming.")
            return audio_array

        top_db_threshold = abs(silence_threshold_db)

        frame_length = 2048
        hop_length = 512

        trimmed_audio, index = librosa.effects.trim(
            y=audio_array,
            top_db=top_db_threshold,
            frame_length=frame_length,
            hop_length=hop_length,
        )

        start_sample, end_sample = index[0], index[1]

        padding_samples = int((padding_ms / 1000.0) * sample_rate)
        final_start = max(0, start_sample - padding_samples)
        final_end = min(len(audio_array), end_sample + padding_samples)

        if final_end > final_start:  # Ensure the slice is valid
            # Check if significant trimming occurred
            original_length = len(audio_array)
            trimmed_length_with_padding = final_end - final_start
            # Heuristic: if length changed by more than just padding, or if original silence was more than min_duration
            # For simplicity, if librosa.effects.trim found *any* indices different from [0, original_length],
            # it means some trimming potential was identified.
            if index[0] > 0 or index[1] < original_length:
                print(
                    f"Silence trimmed: original samples {original_length}, new effective samples {trimmed_length_with_padding} (indices before padding: {index})"
                )
                return audio_array[final_start:final_end]

        print(
            "No significant leading/trailing silence found to trim, or result would be empty."
        )
        return audio_array

    except Exception as e:
        print(f"Error during silence trimming: {e}")
        return audio_array

def fix_internal_silence(
    audio_array: np.ndarray,
    sample_rate: int,
    silence_threshold_db: float = -40.0,
    min_silence_to_fix_ms: int = 700,
    max_allowed_silence_ms: int = 300,
) -> np.ndarray:
    """
    Reduces long internal silences in a NumPy audio array to a specified maximum duration.
    Uses Librosa to split by silence.

    Args:
        audio_array: NumPy array (float32) of the audio.
        sample_rate: Sample rate of the audio.
        silence_threshold_db: Silence threshold in dBFS.
        min_silence_to_fix_ms: Minimum duration of an internal silence to be shortened (ms).
        max_allowed_silence_ms: Target maximum duration for long silences (ms).

    Returns:
        NumPy audio array with long internal silences shortened. Original if no fix needed or on error.
    """
    if audio_array is None or audio_array.size == 0:
        return audio_array

    try:
        if not LIBROSA_AVAILABLE:
            print("Librosa not available, skipping internal silence fixing.")
            return audio_array

        top_db_threshold = abs(silence_threshold_db)
        min_silence_len_samples = int((min_silence_to_fix_ms / 1000.0) * sample_rate)
        max_silence_samples_to_keep = int(
            (max_allowed_silence_ms / 1000.0) * sample_rate
        )

        non_silent_intervals = librosa.effects.split(
            y=audio_array,
            top_db=top_db_threshold,
            frame_length=2048,  # Can be tuned
            hop_length=512,  # Can be tuned
        )

        if len(non_silent_intervals) <= 1:
            print("No significant internal silences found to fix.")
            return audio_array

        fixed_audio_parts = []
        last_nonsilent_end = 0

        for i, (start_sample, end_sample) in enumerate(non_silent_intervals):
            silence_duration_samples = start_sample - last_nonsilent_end
            if silence_duration_samples > 0:
                if silence_duration_samples >= min_silence_len_samples:
                    silence_to_add = audio_array[
                        last_nonsilent_end : last_nonsilent_end
                        + max_silence_samples_to_keep
                    ]
                    fixed_audio_parts.append(silence_to_add)
                    print(
                        f"Shortened internal silence from {silence_duration_samples} to {max_silence_samples_to_keep} samples."
                    )
                else:
                    fixed_audio_parts.append(
                        audio_array[last_nonsilent_end:start_sample]
                    )
            fixed_audio_parts.append(audio_array[start_sample:end_sample])
            last_nonsilent_end = end_sample

        # Handle potential silence after the very last non-silent segment
        # This part is tricky as librosa.effects.split only gives non-silent parts.
        # The trim_lead_trail_silence should handle overall trailing silence.
        # This function focuses on *between* non-silent segments.
        if last_nonsilent_end < len(audio_array):
            trailing_segment = audio_array[last_nonsilent_end:]
            # Check if this trailing segment is mostly silence and long enough to shorten
            # For simplicity, we'll assume trim_lead_trail_silence handles the very end.
            # Or, we could append it if it's short, or shorten it if it's long silence.
            # To avoid over-complication here, let's just append what's left.
            # The primary goal is internal silences.
            # However, if the last "non_silent_interval" was short and followed by a long silence,
            # that silence needs to be handled here too.
            silence_duration_samples = len(audio_array) - last_nonsilent_end
            if silence_duration_samples > 0:
                if silence_duration_samples >= min_silence_len_samples:
                    fixed_audio_parts.append(
                        audio_array[
                            last_nonsilent_end : last_nonsilent_end
                            + max_silence_samples_to_keep
                        ]
                    )
                    print(
                        f"Shortened trailing silence from {silence_duration_samples} to {max_silence_samples_to_keep} samples."
                    )
                else:
                    fixed_audio_parts.append(trailing_segment)
        if not fixed_audio_parts:  # Should not happen if non_silent_intervals > 1
            logger.warning(
                "Internal silence fixing resulted in no audio parts; returning original."
            )
            return audio_array

        return np.concatenate(fixed_audio_parts)

    except Exception as e:
        print(f"Error during internal silence fixing: {e}")
        return audio_array

def remove_long_unvoiced_segments(
    audio_array: np.ndarray,
    sample_rate: int,
    min_unvoiced_duration_ms: int = 300,
    pitch_floor: float = 75.0,
    pitch_ceiling: float = 600.0,
) -> np.ndarray:
    """
    Removes segments from a NumPy audio array that are unvoiced for longer than
    the specified duration, using Parselmouth for pitch analysis.

    Args:
        audio_array: NumPy array (float32) of the audio.
        sample_rate: Sample rate of the audio.
        min_unvoiced_duration_ms: Minimum duration (ms) of an unvoiced segment to be removed.
        pitch_floor: Minimum pitch (Hz) to consider for voicing.
        pitch_ceiling: Maximum pitch (Hz) to consider for voicing.

    Returns:
        NumPy audio array with long unvoiced segments removed. Original if Parselmouth not available or on error.
    """
    if not PARSELMOUTH_AVAILABLE:
        print("Parselmouth not available, skipping unvoiced segment removal.")
        return audio_array
    if audio_array is None or audio_array.size == 0:
        return audio_array

    try:
        sound = parselmouth.Sound(
            audio_array.astype(np.float64), sampling_frequency=sample_rate
        )
        pitch = sound.to_pitch(pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)

        pitch_values = pitch.selected_array['frequency']
        
        voiced_frames = pitch_values > 0
        
        time_step = pitch.get_time_step()
        start_time = pitch.get_start_time()
        frame_times = [start_time + i * time_step for i in range(len(pitch_values))]
        
        segments_to_keep = []
        current_segment_start_sample = 0
        min_unvoiced_samples = int((min_unvoiced_duration_ms / 1000.0) * sample_rate)

        i = 0
        while i < len(voiced_frames):

            is_voiced_current = voiced_frames[i]
            
            j = i
            while j < len(voiced_frames) and voiced_frames[j] == is_voiced_current:
                j += 1
            
            segment_start_time = frame_times[i]
            segment_end_time = frame_times[j-1] + time_step if j < len(frame_times) else frame_times[-1] + time_step
            
            segment_start_sample = int(segment_start_time * sample_rate)
            segment_end_sample = int(segment_end_time * sample_rate)
            segment_duration_samples = segment_end_sample - segment_start_sample
            
            segment_start_sample = min(max(segment_start_sample, 0), len(audio_array))
            segment_end_sample = min(max(segment_end_sample, 0), len(audio_array))
            
            if is_voiced_current:

                if segment_start_sample < segment_end_sample:
                    segments_to_keep.append(audio_array[segment_start_sample:segment_end_sample])
                current_segment_start_sample = segment_end_sample
            else:  
                if segment_duration_samples < min_unvoiced_samples:
                   
                    if segment_start_sample < segment_end_sample:
                        segments_to_keep.append(audio_array[segment_start_sample:segment_end_sample])
                    current_segment_start_sample = segment_end_sample
                else:
                    
                    print(
                        f"Removing long unvoiced segment from {segment_start_time:.2f}s to {segment_end_time:.2f}s."
                    )
                    
                    if segment_start_sample > current_segment_start_sample:
                        segments_to_keep.append(
                            audio_array[current_segment_start_sample:segment_start_sample]
                        )
                    current_segment_start_sample = segment_end_sample
            
            i = j  

        if current_segment_start_sample < len(audio_array):
            segments_to_keep.append(audio_array[current_segment_start_sample:])

        if not segments_to_keep:
            print(
                "Unvoiced segment removal resulted in empty audio; returning original."
            )
            return audio_array

        return np.concatenate(segments_to_keep)

    except Exception as e:
        print(f"Error during unvoiced segment removal: {e}")
        return audio_array

def postprocess(audio_file,speed_factor, silence_trimming, internal_silence_fix, unvoiced_removal):
        speed_factor = float (speed_factor)
        audio_data, engine_output_sample_rate = librosa.load(audio_file, sr=None)
        if silence_trimming:
            audio_data = trim_lead_trail_silence(
                audio_data, engine_output_sample_rate
            )
        
        if internal_silence_fix:
            audio_data = fix_internal_silence(
                audio_data, engine_output_sample_rate
            )

        if unvoiced_removal:
            audio_data = remove_long_unvoiced_segments(
                audio_data, engine_output_sample_rate
            )

        if speed_factor != 1.0:
            encoded_audio_bytes = encode_audio(
                audio_array=audio_data,
                sample_rate=engine_output_sample_rate,
                output_format="wav",
                target_sample_rate=engine_output_sample_rate,
                )
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            file_name_temp = "_temp.wav"
            file_path_temp = os.path.join(OUTPUT_DIR, file_name_temp)
            with open(file_path_temp, "wb") as f:
                f.write(encoded_audio_bytes)
    
            suggested_filename_base = f"qwen3_post_{timestamp_str}"
            file_name = f"{suggested_filename_base}.wav"
            file_path = os.path.join(OUTPUT_DIR, file_name)

            os.system(f"sox {file_path_temp} {file_path} tempo -s {str(speed_factor)}")

            os.remove(file_path_temp)
        else:
            encoded_audio_bytes = encode_audio(
                audio_array=audio_data,
                sample_rate=engine_output_sample_rate,
                output_format="wav",
                target_sample_rate=engine_output_sample_rate,
                )
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            suggested_filename_base = f"qwen3_post_{timestamp_str}"
            file_name = f"{suggested_filename_base}.wav"
            file_path = os.path.join(OUTPUT_DIR, file_name)
            with open(file_path, "wb") as f:
                f.write(encoded_audio_bytes)

        return file_path
 
# Build Gradio UI
def build_ui():
    theme = gr.themes.Soft(
        font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"],
    )

    css = """
    #audio-player-container {
        display: none !important;
    }
    """

    with gr.Blocks(title="Qwen3-TTS",css=css) as demo:
        demo.load(
            None,
            None,
            js="() => {const params = new URLSearchParams(window.location.search);if (!params.has('__theme')) {params.set('__theme', 'dark');window.location.search = params.toString();}}"
        )
        gr.Markdown(
            """
# Qwen3-TTS Demo

A unified Text-to-Speech demo featuring three powerful modes:
- **Voice Design**: Create custom voices using natural language descriptions
- **Voice Clone (Base)**: Clone any voice from a reference audio
- **TTS (CustomVoice)**: Generate speech with predefined speakers and optional style instructions

Built with [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by Alibaba Qwen Team.
"""
        )

        with gr.Tabs():
            # Tab 1: Voice Design (Default, 1.7B only)
            with gr.Tab("Voice Design"):
                gr.Markdown("### Create Custom Voice with Natural Language")
                with gr.Row():
                    with gr.Column(scale=2):
                        design_text = gr.Textbox(
                            label="Text to Synthesize",
                            lines=4,
                            placeholder="Enter the text you want to convert to speech...",
                            value="It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!"
                        )
                        design_language = gr.Dropdown(
                            label="Language",
                            choices=LANGUAGES,
                            value="Auto",
                            interactive=True,
                        )
                        design_instruct = gr.Textbox(
                            label="Voice Description",
                            lines=3,
                            placeholder="Describe the voice characteristics you want...",
                            value="Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice."
                        )
                        

                    with gr.Column(scale=2):
                        #design_audio_out = gr.Audio(label="Generated Audio", type="numpy")
                        design_audio_out = gr.Audio(label="Generated Audio", type="filepath",interactive=True,visible=False,show_download_button=True)
                        design_status = gr.Textbox(label="Status", lines=2, interactive=False)
                with gr.Row():
                    design_btn = gr.Button("Generate with Custom Voice", variant="primary")
                post_btn, post_output, speed_factor_slider, silence_trimming, internal_silence_fix, unvoiced_removal = post_process_gui()               
                design_btn.click(lambda: (gr.update(interactive=False)),outputs=[design_btn]) \
                    .then(
                    generate_voice_design,
                    inputs=[design_text, design_language, design_instruct],
                    outputs=[design_audio_out, design_status]) \
                    .then(lambda: (gr.update(interactive=True),gr.update(visible=True)),outputs=[design_btn,design_audio_out])
                post_btn.click(lambda: (gr.update(interactive=False)),outputs=[post_btn]) \
                    .then(postprocess,inputs=[design_audio_out,speed_factor_slider, silence_trimming, internal_silence_fix, unvoiced_removal],outputs=[post_output]) \
                    .then(lambda: (gr.update(interactive=True),gr.update(visible=True)),outputs=[post_btn,post_output])
            # Tab 2: Voice Clone (Base)
            with gr.Tab("Voice Clone (Base)"):
                gr.Markdown("### Clone Voice from Reference Audio")
                with gr.Row():
                    with gr.Column(scale=2):
                        voice_mode_radio = gr.Radio(
                            choices=["predefined", "custom"],
                            value="predefined",
                            label="Select Voice Mode"
                            )
                        with gr.Group(visible=True) as ref_group:
                            clone_ref_audio_drop = gr.Dropdown ( 
                                label="Reference Audio",
                                choices=REFERENCE,
                                value=REFERENCE[0],
                                interactive=True,
                            )
                            with gr.Row():    
                                ref_play_btn = gr.Button("▶️ Play/Stop")
                            clone_ref_text_drop= gr.Textbox(
                                label="Reference Text",
                                lines=5,
                                value=REFERENCE_TXT,
                                autoscroll=False,
                                max_lines=5
                            )
                            clone_xvector = gr.Checkbox(
                                label="Use x-vector only (No text needed, but lower quality)",
                                value=False,
                                )

                        with gr.Group(visible=False) as custom_group:
                            custom_ref_audio_drop = gr.Dropdown ( 
                                label="Custom Audio",
                                choices=CUSTOM_VOICE,
                                value=CUSTOM_VOICE[0],
                                interactive=True
                            )
                            with gr.Row(): 
                                custom_play_btn = gr.Button("▶️ Play/Stop")
                            with gr.Row():
                                custom_upload_btn = gr.UploadButton("📁 Upload Custom Audio",
                                    file_types=[".wav", ".mp3", ".txt", ".lab"],
                                    file_count="multiple",
                                    visible=True
                                )
                            custom_ref_text_drop= gr.Textbox(
                                label="Custom Text",
                                lines=5,
                                value=CUSTOM_TXT,
                                autoscroll=False,
                                max_lines=5
                            )
                            custom_xvector = gr.Checkbox(
                                label="Use x-vector only (No text needed, but lower quality)",
                                value=False,
                                )

                        voice_mode_radio.change(
                            fn=change_voice_mode,
                            inputs=[voice_mode_radio],
                            outputs=[ref_group, custom_group]
                            )
                        clone_ref_audio_drop.change(
                            fn=select_ref_audio,
                            inputs=[clone_ref_audio_drop],
                            outputs=[clone_ref_text_drop]
                            )  

                        with gr.Row(elem_id="audio-player-container"):
                            pre_player = gr.Audio(
                                visible=True,
                                label="",
                                interactive=False,
                                show_label=False,
                                autoplay=True  
                                ) 
                        ref_play_btn.click(
                            fn=lambda file: toggle_voice_audio(file, "reference"),
                            inputs=[clone_ref_audio_drop],
                            outputs=[pre_player]
                            )
                        custom_play_btn.click(
                            fn=lambda file: toggle_voice_audio(file, "custom"),
                            inputs=[custom_ref_audio_drop],
                            outputs=[pre_player]
                            )
                        custom_upload_btn.upload(
                            fn=on_reference_upload,
                            inputs=[custom_upload_btn],
                            outputs=[custom_ref_audio_drop,clone_ref_text_drop]
                            )
                        custom_ref_audio_drop.change(
                            fn=select_custom_audio,
                            inputs=[custom_ref_audio_drop],
                            outputs=[custom_ref_text_drop]
                            )
                    with gr.Column(scale=2):
                        clone_target_text = gr.Textbox(
                            label="Target Text (Text to synthesize with cloned voice)",
                            lines=4,
                            placeholder="Enter the text you want the cloned voice to speak...",
                        )
                        with gr.Row():
                            clone_language = gr.Dropdown(
                                label="Language",
                                choices=LANGUAGES,
                                value="Auto",
                                interactive=True,
                            )
                            clone_model_size = gr.Dropdown(
                                label="Model Size",
                                choices=MODEL_SIZES,
                                value="1.7B",
                                interactive=True,
                            )
                        clone_status = gr.Textbox(label="Status", lines=2, interactive=False)
                        clone_btn = gr.Button("Clone & Generate", variant="primary")

                with gr.Row():
                    #clone_audio_out = gr.Audio(label="Generated Audio", type="numpy")
                    clone_audio_out = gr.Audio(label="Generated Audio", type="filepath",interactive=True,visible=False,show_download_button=True)
                    
                post_btn_vc, post_output_vc, speed_factor_slider_vc, silence_trimming_vc, internal_silence_fix_vc, unvoiced_removal_vc = post_process_gui()    
                clone_btn.click(lambda: (gr.update(interactive=False)),outputs=[clone_btn]) \
                    .then(
                    generate_voice_clone,
                    inputs=[voice_mode_radio, clone_ref_audio_drop, clone_ref_text_drop, clone_xvector,  
                            custom_ref_audio_drop, custom_ref_text_drop, custom_xvector, clone_target_text, clone_language, clone_model_size],
                    outputs=[clone_audio_out, clone_status]) \
                    .then(lambda: (gr.update(interactive=True),gr.update(visible=True)),outputs=[clone_btn,clone_audio_out])
                post_btn_vc.click(lambda: (gr.update(interactive=False)),outputs=[post_btn_vc]) \
                    .then(postprocess,inputs=[clone_audio_out,speed_factor_slider_vc, silence_trimming_vc, internal_silence_fix_vc, unvoiced_removal_vc],outputs=[post_output_vc]) \
                    .then(lambda: (gr.update(interactive=True),gr.update(visible=True)),outputs=[post_btn_vc,post_output_vc])
            # Tab 3: TTS (CustomVoice)
            with gr.Tab("TTS (CustomVoice)"):
                gr.Markdown("### Text-to-Speech with Predefined Speakers")
                with gr.Row():
                    with gr.Column(scale=2):
                        tts_text = gr.Textbox(
                            label="Text to Synthesize",
                            lines=4,
                            placeholder="Enter the text you want to convert to speech...",
                            value="Hello! Welcome to Text-to-Speech system. This is a demo of our TTS capabilities."
                        )
                        with gr.Row():
                            tts_language = gr.Dropdown(
                                label="Language",
                                choices=LANGUAGES,
                                value="English",
                                interactive=True,
                            )
                            tts_speaker = gr.Dropdown(
                                label="Speaker",
                                choices=SPEAKERS,
                                value="Ryan",
                                interactive=True,
                            )
                        with gr.Row():
                            tts_instruct = gr.Textbox(
                                label="Style Instruction (Optional)",
                                lines=2,
                                placeholder="e.g., Speak in a cheerful and energetic tone",
                            )
                            tts_model_size = gr.Dropdown(
                                label="Model Size",
                                choices=MODEL_SIZES,
                                value="1.7B",
                                interactive=True,
                            )
                        

                    with gr.Column(scale=2):
                        tts_audio_out = gr.Audio(label="Generated Audio", type="filepath",interactive=True,visible=False,show_download_button=True)
                        tts_status = gr.Textbox(label="Status", lines=2, interactive=False)
                        tts_btn = gr.Button("Generate Speech", variant="primary")
                post_btn_tts, post_output_tts, speed_factor_slider_tts, silence_trimming_tts, internal_silence_fix_tts, unvoiced_removal_tts = post_process_gui()    
                post_btn_tts.click(lambda: (gr.update(interactive=False)),outputs=[post_btn_tts]) \
                    .then(postprocess,inputs=[tts_audio_out,speed_factor_slider_tts, silence_trimming_tts, internal_silence_fix_tts, unvoiced_removal_tts],outputs=[post_output_tts]) \
                    .then(lambda: (gr.update(interactive=True),gr.update(visible=True)),outputs=[post_btn_tts,post_output_tts])
                tts_btn.click(lambda: (gr.update(interactive=False)),outputs=[tts_btn]) \
                    .then(
                    generate_custom_voice,
                    inputs=[tts_text, tts_language, tts_speaker, tts_instruct, tts_model_size],
                    outputs=[tts_audio_out, tts_status]) \
                    .then(lambda: (gr.update(interactive=True),gr.update(visible=True)),outputs=[tts_btn,tts_audio_out])

        

    return demo
def parse_args():
    parser = argparse.ArgumentParser(description="Chatterbox TTS Server")
    parser.add_argument("--share", action="store_true", 
                       help="Enable share link for Gradio")

    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    demo = build_ui()
    demo.launch(share=args.share,inbrowser=not args.share)
