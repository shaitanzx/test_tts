<img width="402" height="117" alt="image" src="https://github.com/user-attachments/assets/087def0b-4978-4109-9ab4-975379a69305" />

Qwen3-TTS covers 10 major languages (Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, and Italian) as well as multiple dialectal voice profiles to meet global application needs. 

| Model | Features | Language Support | Streaming | Instruction Control |
|---|---|---|---|---|
| Qwen3-TTS-12Hz-1.7B-VoiceDesign | Performs voice design based on user-provided descriptions. | Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian | ✅ | ✅ |
| Qwen3-TTS-12Hz-1.7B-CustomVoice | Provides style control over target timbres via user instructions; supports 9 premium timbres covering various combinations of gender, age, language, and dialect. | Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian | ✅ | ✅ |
| Qwen3-TTS-12Hz-1.7B-Base | Base model capable of 3-second rapid voice clone from user audio input; can be used for fine-tuning (FT) other models. | Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian | ✅ |  |
| Qwen3-TTS-12Hz-0.6B-CustomVoice | Supports 9 premium timbres covering various combinations of gender, age, language, and dialect. | Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian | ✅ |  |
| Qwen3-TTS-12Hz-0.6B-Base | Base model capable of 3-second rapid voice clone from user audio input; can be used for fine-tuning (FT) other models. | Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian | ✅ |  |

**Voice Design**

Qwen3-TTS supports generating customized timbre identities through natural language descriptions. Users can freely input acoustic attributes, persona descriptions, background information, and other free-form descriptions, easily creating their desired voice identities

<img width="1231" height="586" alt="image" src="https://github.com/user-attachments/assets/743d9f51-9a7e-49c6-8696-8b7774c1ce0f" />

<table>
  <thead><tr><th>Control Type</th><th>Control Instruction</th></tr></thead>
  <tbody>
    <tr>
      <td rowspan="4">Acoustic Attribute Control</td>
      <td>Employ a high-pitched male voice, with the tone rising in sync with excitement, conveying the message at a rapid and energetic pace. The volume should be loud enough, almost shouting, to convey a sense of urgency. Pronunciation must be clear, precise, and distinct, making each word resonate powerfully. The overall delivery should be fluent, natural, bright, vivid, and dramatic, showcasing an extroverted, confident, and assertive personality, while conveying a majestic and grand pronouncement, overflowing with excitement.</td>
    </tr>
    <tr>
      <td>gender: Male.<br> pitch: Low male pitch with significant upward inflections for emphasis and excitement. <br>speed: Fast-paced delivery with deliberate pauses for dramatic effect. <br>volume: Loud and projecting, increasing notably during moments of praise and announcements. <br>age: Young adult to middle-aged adult. <br>clarity: Highly articulate and distinct pronunciation. <br>fluency: Very fluent speech with no hesitations. <br>accent: British English. <br>texture: Bright and clear vocal texture. <br>emotion: Enthusiastic and excited, especially when complimenting. <br>tone: Upbeat, authoritative, and performative. <br>personality: Confident, extroverted, and engaging.</td>
    </tr>
    <tr>
      <td>It reveals a sorrowful and hoarse voice, with a slow speaking speed, intense emotions and a sobbing tone. It is spoken slowly in standard Mandarin, with strong emotions, a mournful and high-pitched tone, and a large range of pitches.</td>
    </tr>
    <tr>
      <td>gender: Male. <br>pitch: Artificially high-pitched, slightly lowering after the initial laugh. <br>speed: Rapid during the laugh, then slowing to a deliberate pace. <br>volume: Loud laugh transitioning to a standard conversational level. <br>age: Young adult to middle-aged, performing a character voice. <br>clarity: Clear and distinct articulation. <br>fluency: Fluent delivery without hesitation. <br>accent: American English. <br>texture: Slightly strained and somewhat nasal quality. <br>emotion: Forced amusement shifting to feigned resignation. <br>tone: Initially playful, then shifts to a slightly put-upon tone. <br>personality: Theatrical and expressive.</td>
    </tr>
    <tr>
      <td rowspan="4">Age Control</td>
      <td>The voice is that of a cute, innocent little girl, with a high pitch and obvious fluctuations, creating a clingy, affected, and deliberately cute auditory effect.</td>
    </tr>
    <tr>
      <td>Speak as a sarcastic, assertive teenage girl: crisp enunciation, controlled volume, with vocal emphasis that conveys disdain and authority.</td>
    </tr>
    <tr>
      <td>Gender: Male.<br>Pitch: Deep male voice, stable pitch.<br>Speed: Slightly fast speech, tight rhythm.<br>Volume: Loud and forceful.<br>Age: Middle-aged or elderly.<br>Clarity: Clear pronunciation, strong and forceful sentences.<br>Fluency: Smooth and fluent delivery.<br>Accent: Standard Mandarin.<br>Tone and Texture: Deep and resonant voice, slightly hoarse.<br>Emotion: Serious and admonishing, clear instructions.<br>Intonation: Commanding tone, emphasizing decisiveness.<br>Personality: Authoritative, decisive, and brooks no argument.</td>
    </tr>
    <tr>
      <td>gender: Male. <br>pitch: Low male pitch, generally stable. <br>speed: Deliberate pace, slowing slightly after the initial exclamation. <br>volume: Starts loud, then transitions to a projected conversational volume. <br>age: Middle-aged adult. <br>clarity: High clarity with distinct pronunciation. <br>fluency: Highly fluent. <br>accent: American English. <br>texture: Resonant and slightly gravelly. <br>emotion: Initially commanding, shifting to narrative amusement. <br>tone: Authoritative start, moving to an engaging, descriptive tone. <br>personality: Confident and performative.</td>
    </tr>
    <tr>
      <td rowspan="2">Gradual Control</td>
      <td>Gender: Male<br>Pitch: Deep male voice, occasionally rising.<br>Speech Rate: Initially steady, gradually increasing in speed due to excitement.<br>Volume: Normal initial volume, gradually increasing to shouting level.<br>Age: Middle-aged male.<br>Clarity: Clear and accurate pronunciation.<br>Fluency: Coherent and natural speech.<br>Accent: Standard Mandarin pronunciation.<br>Tone and Texture: Slightly rough but powerful.<br>Emotion: Initially impatient, quickly turning to annoyance and reprimand.<br>Intonation: Interrogative and commanding, with displeasure and intimidation.<br>Personality: Impatient and easily angered, with a strong attitude.</td>
    </tr>
    <tr>
      <td>gender: Female.<br> pitch: Mid-range female pitch, rising sharply with frustration. <br>speed: Starts measured, then accelerates rapidly during emotional outburst. <br>volume: Begins conversational, escalates quickly to loud and forceful. <br>age: Young adult to middle-aged. <br>clarity: High clarity and distinct articulation throughout. <br>fluency: Highly fluent with no significant pauses or fillers. <br>accent: General American English. <br>texture: Bright and clear vocal quality. <br>emotion: Shifts abruptly from neutral acceptance to intense resentment and anger. <br>tone: Initially accepting, becomes sharply accusatory and confrontational. <br>personality: Assertive and emotionally expressive when provoked.</td>
    </tr>
    <tr>
      <td rowspan="2">Human-likeness</td>
      <td>A natural female voice, with a lively and smiling tone, mimicking someone lowering their voice when shushing you, just like in casual conversation.</td>
    </tr>
    <tr>
      <td>A relaxed, naturally expressive male voice in his late twenties to early thirties, with a moderately low pitch, casual speaking rate, and conversational volume; deliver lines with a light, self-deprecating tone, breaking into genuine, easygoing laughter at moments of embarrassment, while maintaining clear articulation and an overall warm, approachable clarity.</td>
    </tr>
    <tr>
      <td rowspan="2">Background Information</td>
      <td>Character Name: Lin Huaiyue<br>Voice Information: A loud, deep, and powerful middle-aged male voice.<br>Background: Chief consultant of a key national scientific research project, a senior strategic scientist nearing seventy. He has participated in major national scientific and technological projects, witnessing decades of trials and tribulations, and the arduous journey from lagging behind to independent innovation. Currently a lifetime honorary member of the National Science and Technology Advisory Committee, he continues to dedicate himself to cultivating young talent and providing advice for national strategic development.<br>Physical Characteristics: Tall and upright, with graying temples and a resolute expression etched in his brow. He often wears a dark Zhongshan suit or a simple formal suit, his eyes calm yet sharp, exuding authority and composure in every gesture.<br>Personality Traits: With an iron will and unwavering faith, never backing down from challenges; deeply patriotic and concerned for the future of the nation, closely linking personal destiny with the rise and fall of the country; rigorous and self-disciplined, always keeping one's word, speaking with a strong sense of responsibility and historical commitment; outwardly reserved but inwardly warm-hearted, seemingly serious but actually placing high hopes on the younger generation, willingly serving as a stepping stone for them.<br>Life Motto: "Our generation is not here to stand in the light, but to pave the way into the light."</td>
    </tr>
    <tr>
      <td>Character Name: Marcus Cole <br>Voice Profile: A bright, agile male voice with a natural upward lift, delivering lines at a brisk, energetic pace. Pitch leans high with spark, volume projects clearly—near-shouting at peaks—to convey urgency and excitement. Speech flows seamlessly, fluently, each word sharply defined, riding a current of dynamic rhythm. Background: Longtime broadcast booth announcer for national television, specializing in live interstitials and public engagement spots. His voice bridges segments, rallies action, and keeps momentum alive—from voter drives to entertainment news. <br>Presence: Late 50s, neatly groomed, dressed in a crisp shirt under studio lights. Moves with practiced ease, eyes locked on the script, energy coiled and ready. <br>Personality: Energetic, precise, inherently engaging. He doesn’t just read—he propels. Behind the speed is intent: to inform fast, to move people to act. Whether it’s “text VOTE to 5703” or a star-studded tease, he makes it feel immediate, vital.</td>
    </tr>
  </tbody>
</table>


**Voice Clone**

<img width="1235" height="622" alt="image" src="https://github.com/user-attachments/assets/73cb8f6b-d8d8-45cc-bfb5-7121107462d1" />


In this mode, you can replace the voice in your audio with a reference voice.
You can use preset voices as a reference voice, or upload your own (wav, mp3). For best quality replacement, you should also upload transcription text files (txt, lab).

**TTS (CustomVoice)**

<img width="1230" height="546" alt="image" src="https://github.com/user-attachments/assets/941a297f-bfa0-434e-908a-a390397da906" />

Timbre Control
After performing speaker-specific fine-tuning, Qwen3-TTS can maintain the target timbre while inheriting the style control capabilities and single-speaker multilingual capabilities of the base model.

<table>
  <thead>
    <tr>
      <th>Control Type</th>
      <th>Control Instruction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="6">Single Attribute Control</td>
      <td>spoke with a very sad and tearful voice.</td>
    </tr>
    <tr>
      <td>Very happy.</td>
    </tr>
    <tr>
      <td>Speak in a particularly angry tone</td>
    </tr>
    <tr>
      <td>Please speak very quietly.</td>
    </tr>
    <tr>
      <td>Speaking at an extremely slow pace</td>
    </tr>
    <tr>
      <td>low tone</td>
    </tr>
    <tr>
      <td rowspan="3">Multi-Attribute Control</td>
      <td>Gender: Female voice.<br>Pitch: High female range, with varied intonation.<br>Speed: Fast-paced, occasionally accelerating.<br>Volume: Normal conversational volume, loud laughter.<br>Clarity: Clear articulation, standard pronunciation.<br>Fluency: Speaks fluently and naturally.<br>Accent: Mandarin.<br>Timbre: Bright, slightly cheerful.<br>Mood: Pleasant and friendly, accompanied by hearty laughter.<br>Intonation: Lively, rising intonation, especially noticeable when asking questions.<br>Personality: Outgoing, cheerful, enthusiastic, and talkative.</td>
    </tr>
    <tr>
      <td>With extreme sadness and a distinctly tearful tone, she spoke slowly in a low voice, her pace slow, as if each word carried a heavy burden of pain. Her voice trembled and was suppressed, and although her words were soft, they were clear and distinct, revealing a deep-seated sorrow and helplessness.</td>
    </tr>
    <tr>
      <td>Maintaining the vocal characteristics of a young woman, the tone is clear and slightly urgent. The speaking speed starts steadily and gradually increases during the narration. The volume increases when emotions fluctuate, and the tone is raised at the end of sentences to emphasize the admonitory tone.</td>
    </tr>
    <tr>
      <td rowspan="3">Single-speaker Cross-lingual Generalization</td>
      <td>The speaker speaks fluently and naturally at a relatively fast pace, with a clear and bright voice, a slightly high pitch, and clear and standard pronunciation, giving people a happy and pleasant feeling.</td>
    </tr>
    <tr>
      <td>A deep, rich, and solid vocal register characteristic of a middle-aged woman, with full and powerful volume. Speech is delivered at a steady pace, articulation clear and precise, with fluent and confident intonation that rises slightly at the end of sentences.</td>
    </tr>
    <tr>
      <td>The voice should be that of a straightforward and slightly assertive middle-aged woman, with a slightly sharp tone, occasional pauses to emphasize the tone, a hint of dissatisfaction, and a slight increase in volume as the emotion intensifies.</td>
    </tr>
  </tbody>
</table>
For `Qwen3-TTS-12Hz-1.7B/0.6B-CustomVoice` models, the supported speaker list and speaker descriptions are provided below. We recommend using each speaker’s native language for the best quality. Of course, each speaker can speak any language supported by the model.

| Speaker | Voice Description  |  Native language |
| --- | --- | --- |
| Vivian | Bright, slightly edgy young female voice. | Chinese |
| Serena | Warm, gentle young female voice. | Chinese |
| Uncle_Fu | Seasoned male voice with a low, mellow timbre. | Chinese |
| Dylan | Youthful Beijing male voice with a clear, natural timbre. | Chinese (Beijing Dialect) |
| Eric | Lively Chengdu male voice with a slightly husky brightness. | Chinese (Sichuan Dialect) |
| Ryan | Dynamic male voice with strong rhythmic drive. | English |
| Aiden | Sunny American male voice with a clear midrange. | English |
| Ono_Anna | Playful Japanese female voice with a light, nimble timbre. | Japanese |
| Sohee | Warm Korean female voice with rich emotion. | Korean |

**Postprocessing**

<img width="1204" height="493" alt="image" src="https://github.com/user-attachments/assets/c4323cfb-d4df-4347-940b-6122da4fc333" />

After generation in each of the modes, you can additionally run the post-processing process -  removing non-voice sections at the end, middle, and throughout the generated voice, as well as selecting a speed change factor


For local use on Windows, you can download the following packages:
- for CPU only - https://github.com/shaitanzx/chatterbox_vc_mtl/releases/download/Chatterbox/chatterbox_cpu.7z
- for Nvidia GPU 10xx-40xx - https://github.com/shaitanzx/chatterbox_vc_mtl/releases/download/Chatterbox_Nv/chatterbox_nvidia.7z
- for Nvidia GPU 50xx (Be sure to download both files) - https://github.com/shaitanzx/chatterbox_vc_mtl/releases/download/Chatterbox_Nv50xx/chatterbox_nv50xx.7z.001

    and

  https://github.com/shaitanzx/chatterbox_vc_mtl/releases/download/Chatterbox_Nv50xx/chatterbox_nv50xx.7z.002


<table>
  <tr>
    <td><a href="https://colab.research.google.com/github/shaitanzx/chatterbox_vc_mtl/blob/main/chatterbox_vc_mtl.ipynb" rel="nofollow"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg"></a></td><td>Chatterbox_vc_mtl in Google Colab</td>
  </tr>
</table>

All suggestions and questions can be voiced in the [Telegram-group](https://t.me/+xlhhGmrz9SlmYzg6)

![image](https://github.com/user-attachments/assets/5cf86b6d-e378-4d85-aed1-c48920b6c107)
