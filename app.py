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
from huggingface_hub import snapshot_download
import argparse
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
os.makedirs( os.path.join(ROOT, "custom"), exist_ok=True)
CUSTOM_VOICE = sorted([f for f in os.listdir("custom") if f.lower().endswith(('.wav', '.mp3'))]) or [""]
CUSTOM_DIR = os.path.join(ROOT, "custom")
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
        print('---------------all files',all_files)
        print('---------------uploaded_files',uploaded_files)
        updated_options = sorted([f for f in os.listdir("custom") if f.lower().endswith(('.wav', '.mp3'))])
        default_selection = uploaded_files[0] if uploaded_files else "none"
        text=None
        #print('---------------default_selection',default_selection)
        if uploaded_files:
            for file in reversed(uploaded_files):
                if file.lower().endswith(('.wav', '.mp3')):
                    default_selection = file
                    break

            if not default_selection:
                base_name = os.path.splitext(uploaded_files[-1])[0]  # Убираем расширение
                # Ищем файл с тем же именем, но с расширением .wav или .mp3
                for ext in ('.wav', '.mp3'):
                    audio_file = f"{base_name}{ext}"
                    if audio_file in all_files:
                        default_selection = audio_file
                        break
                if default_selection==None:
                    return gr.update(),gr.update()
                else:
                    return gr.update(choices=updated_options,value=default_selection),gr.update(value=select_custom_audio(audio_file))
            else:
                return gr.update(choices=updated_options,value=default_selection),gr.update(value=select_custom_audio(default_selectionh))
        else:
            return gr.update(choices=all_files)
            
    except Exception as e:
        print(f"Error in reference upload: {e}", exc_info=True)
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
            device_map="cuda",
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

def generate_voice_design(text, language, voice_description):
    """Generate speech using Voice Design model (1.7B only)."""
    if not text or not text.strip():
        return None, "Error: Text is required."
    if not voice_description or not voice_description.strip():
        return None, "Error: Voice description is required."

    try:
        tts = get_model("VoiceDesign", "1.7B")
        wavs, sr = tts.generate_voice_design(
            text=text.strip(),
            language=language,
            instruct=voice_description.strip(),
            non_streaming_mode=True,
            max_new_tokens=2048,
        )
        return (sr, wavs[0]), "Voice design generation completed successfully!"
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
        return (sr, wavs[0]), "Voice clone generation completed successfully!"
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
        wavs, sr = tts.generate_custom_voice(
            text=text.strip(),
            language=language,
            speaker=speaker.lower().replace(" ", "_"),
            instruct=instruct.strip() if instruct else None,
            non_streaming_mode=True,
            max_new_tokens=2048,
        )
        return (sr, wavs[0]), "Generation completed successfully!"
    except Exception as e:
        return None, f"Error: {type(e).__name__}: {e}"


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
                        design_btn = gr.Button("Generate with Custom Voice", variant="primary")

                    with gr.Column(scale=2):
                        design_audio_out = gr.Audio(label="Generated Audio", type="numpy")
                        design_status = gr.Textbox(label="Status", lines=2, interactive=False)

                design_btn.click(
                    generate_voice_design,
                    inputs=[design_text, design_language, design_instruct],
                    outputs=[design_audio_out, design_status],
                )

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


         
#                        clone_ref_audio = gr.Audio(
#                            label="Reference Audio (Upload a voice sample to clone)",
#                            type="numpy",
#                        )
#                        clone_ref_text = gr.Textbox(
#                            label="Reference Text (Transcript of the reference audio)",
#                            lines=2,
#                            placeholder="Enter the exact text spoken in the reference audio...",
#                        )
#                        clone_xvector = gr.Checkbox(
#                            label="Use x-vector only (No reference text needed, but lower quality)",
#                            value=False,
#                        )
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
                        clone_btn = gr.Button("Clone & Generate", variant="primary")

                with gr.Row():
                    clone_audio_out = gr.Audio(label="Generated Audio", type="numpy")
                    clone_status = gr.Textbox(label="Status", lines=2, interactive=False)

                clone_btn.click(
                    generate_voice_clone,
                    inputs=[voice_mode_radio, clone_ref_audio_drop, clone_ref_text_drop, clone_xvector,  
                            custom_ref_audio_drop, custom_ref_text_drop, custom_xvector, clone_target_text, clone_language, clone_model_size],
                    outputs=[clone_audio_out, clone_status],
                )

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
                        tts_btn = gr.Button("Generate Speech", variant="primary")

                    with gr.Column(scale=2):
                        tts_audio_out = gr.Audio(label="Generated Audio", type="numpy")
                        tts_status = gr.Textbox(label="Status", lines=2, interactive=False)

                tts_btn.click(
                    generate_custom_voice,
                    inputs=[tts_text, tts_language, tts_speaker, tts_instruct, tts_model_size],
                    outputs=[tts_audio_out, tts_status],
                )

        

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
