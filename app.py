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
from huggingface_hub import snapshot_download

from huggingface_hub import login
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
REFERENCE_TXT = read_text_for_audio(os.path.join(REF_DIR, REFERENCE[0]))
CUSTOM_TXT = read_text_for_audio(os.path.join(CUSTOM_DIR, CUSTOM_VOICE[0]))
def toggle_voice_audio(selected_file, voice_mode):
    global reference_playing_state
    if voice_mode == "reference":
        base_path = REF_DIR
    else: 
        base_path = CUSTOM_DIR
    
    file_path = os.path.join(base_path,selected_file)
   
    
    #if not file_path.exists():
    #    gr.Error(f"❌ File not found: {selected_file}")
    #    return None, "▶️ Play/Stop", gr.update(visible=False), gr.update(visible=False)
    
    current_key = f"{voice_mode}_{selected_file}"
    
    if reference_playing_state["is_playing"] and reference_playing_state["current_key"] == current_key:
        reference_playing_state = {"is_playing": False, "current_key": None}
        gr.Info(f"⏸️ Stopped: {selected_file}")
        return None

    reference_playing_state = {"is_playing": True, "current_key": current_key}
    gr.Info(f"🎵 Playing: {selected_file}")
    
    return str(file_path)  
    
#def reset_playback_on_mode_change(voice_mode):

#    global reference_playing_state
#    reference_playing_state = {"is_playing": False, "current_key": None}
#    return "▶️ Play/Stop", "▶️ Play/Stop", gr.update(visible=False)

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


def generate_voice_clone(ref_audio, ref_text, target_text, language, use_xvector_only, model_size):
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

#    css = """
#    .gradio-container {max-width: none !important;}
#    .tab-content {padding: 20px;}
#    """

    with gr.Blocks(title="Qwen3-TTS") as demo:
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

                        pre_player = gr.Audio(
                            visible=True,
                            label="",
                            interactive=False,
                            show_label=True,
                            elem_id="reference-audio-player",
                            autoplay=True  
                            )  
                        #reference_audio_trigger = gr.Audio(
                        #    visible=False,
                        #    elem_id="reference-audio-trigger"
                        #    )
                        ref_play_btn.click(
                            fn=lambda file: toggle_voice_audio(file, "reference"),
                            inputs=[clone_ref_audio_drop],
                            outputs=[pre_player]
                            )
                        custom_play_btn.click(
                            fn=lambda file: toggle_voice_audio(file, "custom"),
                            inputs=[clone_ref_audio_drop],
                            outputs=[pre_player]
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

#                clone_btn.click(
#                    generate_voice_clone,
#                    inputs=[clone_ref_audio, clone_ref_text, clone_target_text, clone_language, clone_xvector, clone_model_size],
#                    outputs=[clone_audio_out, clone_status],
#                )

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


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(share=True)
