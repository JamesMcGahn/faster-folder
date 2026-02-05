import subprocess
import argparse
import sys
from faster_whisper import WhisperModel
from pathlib import Path
from tqdm import tqdm
import math


def whisper_folder():
    AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".avi", ".mp4", ".ts"}
    SKIP_FILES = {"00. Professor.avi"}

    parser = argparse.ArgumentParser(
        description="Whisper the Folder of Audio or Video Files. If a Folder is a Video folder, the files will be converted to WAV files first."
    )

    parser.add_argument("--model", default="medium")
    parser.add_argument("--language", default="en")
    parser.add_argument("--directory", default="./")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--single-file", default=None)
    parser.add_argument("--convert-wav-only", action="store_true")
    parser.add_argument("--vad-filter", action="store_true")

    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--chunk-length", type=int, default=60)
    parser.add_argument("--min-silence-ms", type=int, default=2500)
    parser.add_argument("--compute-type", default="auto")
    parser.add_argument("--initial-prompt", default="transcribe")
    parser.add_argument("--keep-wav-files", action="store_true")
    parser.add_argument("--file-count", action="store_true")

    args = parser.parse_args()

    MODEL = args.model
    LANGUAGE = args.language
    AUDIO_DIR = Path(args.directory)
    START = args.start - 1
    SINGLE_FILE = args.single_file
    CONVERT_TO_WAV_ONLY = args.convert_wav_only
    BEAM_SIZE = args.beam_size
    CHUNK_LENGTH = args.chunk_length
    MIN_SILENCE = args.min_silence_ms
    COMPUTE_TYPE = args.compute_type
    INITIAL_PROMPT = args.initial_prompt
    KEEP_WAV_FILES = args.keep_wav_files
    FILE_COUNT = args.file_count
    VAD_FILTER = args.vad_filter

    # print(MODEL, LANGUAGE, AUDIO_DIR, START, SINGLE_FILE, CONVERT_TO_WAV_ONLY)
    print(f"Starting from file {START+1}")
    # ---- File discovery ----
    Path
    if SINGLE_FILE:
        file = Path(SINGLE_FILE)
        single_path = file.exists()
        files = [file] if single_path else [AUDIO_DIR / SINGLE_FILE]

    else:
        files = [
            p
            for p in AUDIO_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS
        ]
        files.sort()

    total_files = len(files)

    if FILE_COUNT:
        print(f"Total number of Files to Whisper: {total_files}")
        return

    if START < 0 or START > len(files) - 1:
        print("❌ Start Position Inputed is out of the range of files!")
        return

    pbar_files = tqdm(
        total=total_files,
        desc="Files",
        position=0,
        colour="blue",
        leave=True,
        unit="sec",
        initial=START,
    )

    count = START + 1
    print(f"📦 Loading model: {MODEL}")

    model = WhisperModel(MODEL, compute_type=COMPUTE_TYPE)

    for file_path in files[START:]:
        created_wav_file = False
        if file_path.name in SKIP_FILES:
            print(f"Skipping {file_path.name}")
            count += 1
            continue

        if not file_path.exists():
            print("❌ Skipping File - File no longer exists at path given.")
            count += 1
            continue

        print(f"⏳ Processing {count}/{total_files}: {file_path}")

        # ---- Convert to WAV ----
        print(f"🎙️ {count}/{total_files}: Converting to wav (16k mono)")

        out_wav = file_path.with_suffix(".wav")
        if file_path.suffix.lower() != ".wav":
            created_wav_file = True
            subprocess.run(
                [
                    "ffmpeg",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(file_path),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    str(out_wav),
                ],
                check=True,
            )

            print(f"✅ WAV Conversion Done {count}/{total_files}: {out_wav}\n")

            if CONVERT_TO_WAV_ONLY:
                count += 1
                continue

        print(f"🏗️  {count}/{total_files}: Starting Whisper")

        def format_srt_time(seconds: float) -> str:
            ms = int((seconds - int(seconds)) * 1000)
            s = int(seconds) % 60
            m = (int(seconds) // 60) % 60
            h = int(seconds) // 3600
            return f"{h:02}:{m:02}:{s:02},{ms:03}"

        print(f"🎧 Transcribing: {out_wav}")

        segments, info = model.transcribe(
            str(out_wav),
            task="transcribe",
            language=LANGUAGE,
            chunk_length=CHUNK_LENGTH,
            beam_size=BEAM_SIZE,
            condition_on_previous_text=False,
            vad_filter=VAD_FILTER,
            vad_parameters={"min_silence_duration_ms": MIN_SILENCE},
            word_timestamps=False,
        )

        srt_lines = []
        txt_lines = []
        index = 1
        total_duration = float(info.duration or 0.0)
        last_logged_pct = -1
        last_end = 0

        progress_accumulation = 0.0

        with tqdm(
            total=int(total_duration),
            unit="sec",
            unit_scale=False,
            desc="Transcribing",
            position=0,
            colour="green",
            leave=True,
        ) as pbar_audio:

            for seg in segments:
                if not seg.text:
                    continue

                start = format_srt_time(seg.start)
                end = format_srt_time(seg.end)

                srt_lines.extend(
                    [
                        str(index),
                        f"{start} --> {end}",
                        seg.text,
                        "",
                    ]
                )

                txt_lines.append(seg.text)

                # pct = math.ceil(min(1.0, seg.end / total_duration) * 100)
                # delta = pct - last_logged_pct

                delta = max(0.0, seg.end - last_end)
                last_end = seg.end

                progress_accumulation += delta
                whole_seconds = int(progress_accumulation)

                if whole_seconds > 0:
                    pbar_audio.update(whole_seconds)
                    progress_accumulation -= whole_seconds
                # if delta > 0:
                #     pbar_audio.update(delta)
                #     last_logged_pct = pct

                index += 1
        remaining = pbar_audio.total - pbar_audio.n
        if remaining > 0:
            pbar_audio.update(remaining)

            if not srt_lines:
                print("❌ Empty transcript")
                count += 1
                continue

            print("")
            out_txt = out_wav.with_suffix(".txt")
            out_srt = out_wav.with_suffix(".srt")
            print(f"📝 Starting Writing")
            out_txt.write_text("\n".join(txt_lines), encoding="utf-8")
            out_srt.write_text("\n".join(srt_lines), encoding="utf-8")

            print(f"📝 Wrote: {out_txt}")
            print(f"📝 Wrote: {out_srt}")
            print(f"✅ Done {count}/{total_files}\n")

            if created_wav_file and not KEEP_WAV_FILES and out_wav.is_file():
                out_wav.unlink()

            count += 1
        pbar_files.update(1)

    pbar_files.close()
    print("👋 Done.")


def main():
    try:
        whisper_folder()
    except KeyboardInterrupt as e:
        print("\n")
        print("Quitting!!")
        sys.exit(130)


if __name__ == "__main__":
    main()
