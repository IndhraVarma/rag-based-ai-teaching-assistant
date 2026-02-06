from faster_whisper import WhisperModel
import os
import json

audio_path = r"D:\BTech Projects\RAG based AI Teaching Assistant\audios\12_Player Controls.mp3"
output_json = r"D:\BTech Projects\RAG based AI Teaching Assistant\outputs\12_Player_Controls.json"

if not os.path.exists(audio_path):
    raise FileNotFoundError(f"File not found: {audio_path}")

model = WhisperModel(
    "large-v2",
    device="cpu",
    compute_type="int8"
)

segments, info = model.transcribe(
    audio_path,
    language="hi",
    task="translate"
)

segments_json = []
for i, seg in enumerate(segments):
    segments_json.append({
        "id": i,
        "start": round(seg.start, 2),
        "end": round(seg.end, 2),
        "text": seg.text.strip()
    })

os.makedirs(os.path.dirname(output_json), exist_ok=True)

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(
        {
            "audio_file": os.path.basename(audio_path),
            "language": info.language,
            "duration": info.duration,
            "segments": segments_json
        },
        f,
        ensure_ascii=False,
        indent=4
    )

print(f"✅ Transcription saved to: {output_json}")
