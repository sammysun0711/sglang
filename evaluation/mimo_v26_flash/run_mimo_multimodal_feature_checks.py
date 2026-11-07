#!/usr/bin/env python3
"""Check MiMo media grounding and DP isolation against a real-prefill server.

Test media is prepared automatically and cached beside this script. Results
default to the script directory and are replaced on each run. Each evaluation
writes summary.json and summary.md, and prints the Markdown results table.
"""

import argparse
import base64
import hashlib
import html
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont


def run(command):
    subprocess.run(command, check=True, capture_output=True, timeout=60)


def prepare_media(directory):
    directory.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 70)
    for word, color in [("APPLE", "red"), ("BANANA", "blue")]:
        path = directory / f"{word.lower()}.png"
        if not path.exists():
            image = Image.new("RGB", (512, 512), "white")
            draw = ImageDraw.Draw(image)
            draw.text((256, 90), word, fill="black", font=font, anchor="mm")
            draw.rectangle((120, 190, 392, 440), fill=color)
            image.save(path)

    for name, words in [
        ("forward", ["apple", "banana"]),
        ("reverse", ["banana", "apple"]),
    ]:
        path = directory / f"{name}.mp4"
        if not path.exists():
            # Two seconds of each image, at 2 fps; default 1-fps sampling must
            # still see both states, in the correct order.
            frames = directory / name
            frames.mkdir(exist_ok=True)
            for i in range(8):
                (frames / f"{i:02d}.png").write_bytes(
                    (directory / f"{words[i // 4]}.png").read_bytes()
                )
            run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-framerate",
                    "2",
                    "-i",
                    str(frames / "%02d.png"),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(path),
                ]
            )

    for name, sentence in [
        ("number", "Please remember the number forty two."),
        ("secret", "The secret word is orange."),
    ]:
        speech = directory / f"{name}_source.wav"
        if not speech.exists():
            run(["espeak-ng", "-v", "en-us", "-s", "160", "-w", str(speech), sentence])
        wav = directory / f"{name}_16k.wav"
        if not wav.exists():
            run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-i",
                    str(speech),
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    str(wav),
                ]
            )
        mp3 = directory / f"{name}.mp3"
        if not mp3.exists():
            run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-i",
                    str(speech),
                    "-ar",
                    "24000",
                    "-ac",
                    "1",
                    str(mp3),
                ]
            )

    av_path = directory / "short_audio_tail.mp4"
    if not av_path.exists():
        run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(directory / "forward.mp4"),
                "-i",
                str(directory / "secret_source.wav"),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                str(av_path),
            ]
        )

    natural = directory / "natural_speech.mp3"
    natural_url = "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/audios/Trump_WEF_2018_10s.mp3"
    if not natural.exists():
        response = requests.get(natural_url, timeout=60)
        response.raise_for_status()
        natural.write_bytes(response.content)

    manifest = {
        "natural_audio_source": natural_url,
        "files": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.iterdir()
            if p.is_file() and p.suffix in {".png", ".mp4", ".wav", ".mp3"}
        },
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def media(directory, filename):
    path = directory / filename
    kind, mime = {
        ".png": ("image_url", "image/png"),
        ".mp4": ("video_url", "video/mp4"),
        ".wav": ("audio_url", "audio/wav"),
        ".mp3": ("audio_url", "audio/mpeg"),
    }[path.suffix]
    return {
        "type": kind,
        kind: {
            "url": f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()
        },
    }


def cases():
    word = "Read the large printed word. Reply with only that word."
    order = "Which large printed word appears first, and which appears last? Reply with only the two words, in order."
    secret = "Which secret word was spoken? Reply with only that word."
    return [
        dict(
            name="image_apple_dp0",
            files=["apple.png"],
            question=word,
            patterns=[r"\bapple\b"],
            dp=0,
        ),
        dict(
            name="image_banana_dp1",
            files=["banana.png"],
            question=word,
            patterns=[r"\bbanana\b"],
            dp=1,
        ),
        dict(
            name="image_red",
            files=["apple.png"],
            question="What color is the large filled square? Reply with only the color.",
            patterns=[r"\bred\b"],
            dp=0,
        ),
        dict(
            name="image_blue",
            files=["banana.png"],
            question="What color is the large filled square? Reply with only the color.",
            patterns=[r"\bblue\b"],
            dp=1,
        ),
        dict(
            name="multiple_images",
            files=["apple.png", "banana.png"],
            question="Read the large word in each image, first image then second. Reply with only the two words in order.",
            patterns=[r"apple.*banana"],
            dp=0,
        ),
        dict(
            name="video_forward",
            files=["forward.mp4"],
            question=order,
            patterns=[r"apple.*banana"],
            dp=1,
        ),
        dict(
            name="video_reverse",
            files=["reverse.mp4"],
            question=order,
            patterns=[r"banana.*apple"],
            dp=0,
        ),
        dict(
            name="audio_wav_16k",
            files=["number_16k.wav"],
            question="Transcribe the speech. Return only the transcription.",
            patterns=[r"(?:forty[ -]two|\b42\b)"],
            dp=0,
        ),
        dict(
            name="audio_mp3",
            files=["secret.mp3"],
            question=secret,
            patterns=[r"\borange\b"],
            dp=1,
        ),
        dict(
            name="video_short_audio",
            files=["short_audio_tail.mp4"],
            question="What secret word is spoken, and which printed word is visible at the end? Reply with only those two words, spoken then printed.",
            patterns=[r"orange.*banana"],
            dp=0,
        ),
        dict(
            name="natural_audio",
            files=["natural_speech.mp3"],
            question="Transcribe the speech. Return only the transcription.",
            patterns=[r"\bprivilege\b", r"\bforum\b"],
            dp=1,
        ),
        dict(
            name="repeated_image_other_dp",
            files=["apple.png"],
            question=word,
            patterns=[r"\bapple\b"],
            dp=1,
        ),
        dict(
            name="text_control",
            files=[],
            question="Reply with exactly the word ORBIT.",
            patterns=[r"\borbit\b"],
            dp=1,
        ),
        dict(
            name="chunked_image",
            files=["apple.png"],
            question=("background " * 20000) + "\n" + word,
            patterns=[r"\bapple\b"],
            dp=0,
            min_prompt_tokens=16385,
        ),
    ]


def normalize(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def check_case(case, args, model):
    start = time.monotonic()
    content = [media(args.media_dir, f) for f in case["files"]]
    content.append({"type": "text", "text": case["question"]})
    payload = dict(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0,
        max_tokens=256,
        routed_dp_rank=case["dp"],
        chat_template_kwargs={"enable_thinking": False},
    )
    result = dict(
        name=case["name"],
        files=case["files"],
        routed_dp_rank=case["dp"],
        patterns=case["patterns"],
        passed=False,
    )
    try:
        response = requests.post(
            args.base_url.rstrip("/") + "/v1/chat/completions",
            json=payload,
            timeout=300,
        )
        result["http_status"] = response.status_code
        response.raise_for_status()
        data = response.json()
        answer = data["choices"][0]["message"].get("content") or ""
        result.update(response=data, answer=answer)
        result["passed"] = all(
            re.search(p, answer.lower(), re.S) for p in case["patterns"]
        )
        if "min_prompt_tokens" in case:
            result["passed"] &= (
                data.get("usage", {}).get("prompt_tokens", 0)
                >= case["min_prompt_tokens"]
            )
    except Exception as error:
        result["error"] = repr(error)
    result["elapsed_seconds"] = time.monotonic() - start
    (args.output / (case["name"] + ".json")).write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "response"}), flush=True)
    return result


def format_markdown_summary(summary):
    def cell(value):
        text = html.escape(str(value), quote=False)
        text = text.replace("\\", "\\\\").replace("|", "\\|")
        return text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")

    status = "PASS" if summary["passed"] else "FAIL"
    lines = [
        "## MiMo multimodal feature checks",
        "",
        f"**{status}: {summary['passed_cases']}/{summary['total_cases']} checks passed.**",
        "",
        "| Test case | DP rank | Result | Response / error |",
        "| --- | ---: | --- | --- |",
    ]
    for result in summary["results"]:
        detail = result.get("error") or result.get("answer", "")
        if result.get("matches_serial") is False:
            detail += " (does not match the normalized serial response)"
        cells = [
            result["name"],
            result["routed_dp_rank"],
            "Pass" if result["passed"] else "Fail",
            detail,
        ]
        lines.append("| " + " | ".join(cell(value) for value in cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30001")
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir,
        help="Result directory (default: this script's directory); overwrite prior results.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare cached test media without sending requests or replacing results.",
    )
    args = parser.parse_args()
    args.output = args.output.expanduser().resolve()
    args.media_dir = script_dir / ".cache" / "mimo_multimodal_media"
    all_cases = cases()
    mixed_names = ["image_apple_dp0", "text_control", "audio_mp3", "video_forward"]
    mixed_cases = [
        dict(c, name="mixed_" + c["name"])
        for c in all_cases
        if c["name"] in mixed_names
    ]
    if not args.prepare_only:
        args.output.mkdir(parents=True, exist_ok=True)
        # Remove only this script's results so an interrupted retry cannot leave
        # an old successful summary or stale cases alongside new results.
        result_names = ["summary.json", "summary.md", "server_info.json"] + [
            case["name"] + ".json" for case in all_cases + mixed_cases
        ]
        for name in result_names:
            (args.output / name).unlink(missing_ok=True)
        print(f"Results: {args.output} (overwriting previous results)", flush=True)
    prepare_media(args.media_dir)
    if args.prepare_only:
        print(args.media_dir / "manifest.json")
        return
    models = requests.get(args.base_url.rstrip("/") + "/v1/models", timeout=30)
    models.raise_for_status()
    model = models.json()["data"][0]["id"]
    info = requests.get(args.base_url.rstrip("/") + "/server_info", timeout=30)
    info.raise_for_status()
    (args.output / "server_info.json").write_text(
        json.dumps(info.json(), indent=2) + "\n"
    )
    results = [check_case(case, args, model) for case in all_cases]
    baseline = {r["name"]: r for r in results}
    with ThreadPoolExecutor(max_workers=len(mixed_cases)) as pool:
        mixed_results = list(
            pool.map(lambda c: check_case(c, args, model), mixed_cases)
        )
    for result in mixed_results:
        original = baseline[result["name"].removeprefix("mixed_")]
        result["matches_serial"] = normalize(result.get("answer", "")) == normalize(
            original.get("answer", "")
        )
        result["passed"] &= result["matches_serial"]
        (args.output / (result["name"] + ".json")).write_text(
            json.dumps(result, indent=2) + "\n"
        )
    results += mixed_results
    summary = dict(
        passed=all(r["passed"] for r in results),
        passed_cases=sum(r["passed"] for r in results),
        total_cases=len(results),
        results=results,
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    markdown = format_markdown_summary(summary)
    (args.output / "summary.md").write_text(markdown)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}), flush=True)
    print(markdown, flush=True)
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
