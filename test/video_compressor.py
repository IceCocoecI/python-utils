from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Sequence


DEFAULT_VIDEO_CODEC = "libx264"
DEFAULT_PRESET = "medium"
DEFAULT_GENERIC_SUFFIX = "compressed"
DEFAULT_BITRATE_ONLY_SUFFIX = "bitrate_only"
DEFAULT_TARGET_SIZE_MB = 10.0
MANUAL_TEST_VIDEO_PATH = "/home/cz/视频/视频压缩测试/001.mp4"


@dataclass(frozen=True)
class MediaInfo:
    path: Path
    duration_seconds: float
    width: int
    height: int
    fps: float
    total_bitrate_bps: int
    video_bitrate_bps: int
    audio_bitrate_bps: int


class VideoCompressor:
    @staticmethod
    def _resolve_input_path(input_path: str | Path) -> Path:
        resolved = Path(input_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Input video does not exist: {resolved}")
        if not resolved.is_file():
            raise ValueError(f"Input path is not a file: {resolved}")
        return resolved

    @staticmethod
    def _run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(command, check=True, text=True, capture_output=True)

    @staticmethod
    def _safe_int(value: object) -> int:
        if value in (None, "", "N/A"):
            return 0
        return int(float(str(value)))

    @staticmethod
    def _parse_fps(value: str | None) -> float:
        if not value or value == "0/0":
            return 0.0
        return float(Fraction(value))

    @staticmethod
    def _format_kbps(value: float) -> str:
        formatted = f"{value:.2f}".rstrip("0").rstrip(".")
        return f"{formatted}k"

    @classmethod
    def _build_output_path(
        cls,
        input_path: Path,
        output_name_suffix: str,
        output_path: str | Path | None = None,
    ) -> Path:
        if output_path:
            resolved = Path(output_path).expanduser().resolve()
        else:
            resolved = input_path.with_name(
                f"{input_path.stem}_{output_name_suffix}{input_path.suffix}"
            )

        if resolved == input_path:
            raise ValueError("Output path must be different from input path.")

        if not resolved.exists():
            return resolved

        counter = 2
        while True:
            candidate = resolved.with_name(
                f"{resolved.stem}_{counter}{resolved.suffix}"
            )
            if not candidate.exists():
                return candidate
            counter += 1

    @classmethod
    def probe_media_info(cls, input_path: str | Path) -> MediaInfo:
        resolved = cls._resolve_input_path(input_path)
        command = [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(resolved),
        ]
        result = cls._run_command(command)
        payload = json.loads(result.stdout)

        streams = payload.get("streams", [])
        format_info = payload.get("format", {})
        video_stream = next(
            (stream for stream in streams if stream.get("codec_type") == "video"),
            None,
        )
        if not video_stream:
            raise ValueError(f"No video stream found in: {resolved}")

        audio_stream = next(
            (stream for stream in streams if stream.get("codec_type") == "audio"),
            None,
        )

        duration_seconds = float(
            format_info.get("duration")
            or video_stream.get("duration")
            or 0
        )
        if duration_seconds <= 0:
            raise ValueError(f"Failed to read video duration: {resolved}")

        total_bitrate_bps = cls._safe_int(format_info.get("bit_rate"))
        video_bitrate_bps = cls._safe_int(video_stream.get("bit_rate"))
        audio_bitrate_bps = (
            cls._safe_int(audio_stream.get("bit_rate")) if audio_stream else 0
        )

        if video_bitrate_bps <= 0 and total_bitrate_bps > 0:
            video_bitrate_bps = max(total_bitrate_bps - audio_bitrate_bps, 0)

        return MediaInfo(
            path=resolved,
            duration_seconds=duration_seconds,
            width=int(video_stream["width"]),
            height=int(video_stream["height"]),
            fps=cls._parse_fps(
                video_stream.get("avg_frame_rate")
                or video_stream.get("r_frame_rate")
            ),
            total_bitrate_bps=total_bitrate_bps,
            video_bitrate_bps=video_bitrate_bps,
            audio_bitrate_bps=audio_bitrate_bps,
        )

    @classmethod
    def compress_video(
        cls,
        input_path: str | Path,
        target_width: int | None = None,
        target_height: int | None = None,
        target_fps: float | None = None,
        target_video_bitrate_kbps: float | None = None,
        output_path: str | Path | None = None,
        output_name_suffix: str = DEFAULT_GENERIC_SUFFIX,
        video_codec: str = DEFAULT_VIDEO_CODEC,
        preset: str = DEFAULT_PRESET,
    ) -> Path:
        if all(
            value is None
            for value in (
                target_width,
                target_height,
                target_fps,
                target_video_bitrate_kbps,
            )
        ):
            raise ValueError(
                "At least one of target_width, target_height, target_fps, "
                "or target_video_bitrate_kbps must be provided."
            )

        resolved_input = cls._resolve_input_path(input_path)
        resolved_output = cls._build_output_path(
            input_path=resolved_input,
            output_name_suffix=output_name_suffix,
            output_path=output_path,
        )

        video_filters: list[str] = []
        if target_width and target_height:
            video_filters.append(f"scale={target_width}:{target_height}")
        elif target_width:
            video_filters.append(f"scale={target_width}:-2")
        elif target_height:
            video_filters.append(f"scale=-2:{target_height}")

        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(resolved_input),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            video_codec,
            "-preset",
            preset,
            "-pix_fmt",
            "yuv420p",
        ]

        if video_filters:
            command.extend(["-vf", ",".join(video_filters)])
        if target_fps is not None:
            command.extend(["-r", str(target_fps)])
        if target_video_bitrate_kbps is not None:
            if target_video_bitrate_kbps <= 0:
                raise ValueError("target_video_bitrate_kbps must be greater than 0.")
            bitrate_value = cls._format_kbps(target_video_bitrate_kbps)
            command.extend(
                [
                    "-b:v",
                    bitrate_value,
                    "-maxrate",
                    bitrate_value,
                    "-bufsize",
                    cls._format_kbps(target_video_bitrate_kbps * 2),
                ]
            )

        command.extend(
            [
                "-c:a",
                "copy",
                str(resolved_output),
            ]
        )

        subprocess.run(command, check=True)
        return resolved_output

    @classmethod
    def calculate_target_video_bitrate_kbps(
        cls,
        input_path: str | Path,
        target_size_mb: float = DEFAULT_TARGET_SIZE_MB,
    ) -> float:
        if target_size_mb <= 0:
            raise ValueError("target_size_mb must be greater than 0.")

        media_info = cls.probe_media_info(input_path)
        target_total_bits = target_size_mb * 1024 * 1024 * 8
        target_total_bitrate_bps = target_total_bits / media_info.duration_seconds
        target_video_bitrate_bps = target_total_bitrate_bps - media_info.audio_bitrate_bps

        if target_video_bitrate_bps <= 0:
            raise ValueError(
                "Audio stream already exceeds the requested file size budget."
            )

        return max(target_video_bitrate_bps / 1000, 100.0)

    @classmethod
    def compress_video_bitrate_only(
        cls,
        input_path: str | Path,
        target_size_mb: float = DEFAULT_TARGET_SIZE_MB,
        target_video_bitrate_kbps: float | None = None,
        output_path: str | Path | None = None,
        output_name_suffix: str = DEFAULT_BITRATE_ONLY_SUFFIX,
        video_codec: str = DEFAULT_VIDEO_CODEC,
        preset: str = DEFAULT_PRESET,
    ) -> Path:
        if target_video_bitrate_kbps is None:
            target_video_bitrate_kbps = cls.calculate_target_video_bitrate_kbps(
                input_path=input_path,
                target_size_mb=target_size_mb,
            )
        elif target_video_bitrate_kbps <= 0:
            raise ValueError("target_video_bitrate_kbps must be greater than 0.")

        return cls.compress_video(
            input_path=input_path,
            target_video_bitrate_kbps=target_video_bitrate_kbps,
            output_path=output_path,
            output_name_suffix=output_name_suffix,
            video_codec=video_codec,
            preset=preset,
        )

    @classmethod
    def compress_video_bitrate_only_to_target_size(
        cls,
        input_path: str | Path,
        target_size_mb: float = DEFAULT_TARGET_SIZE_MB,
        output_path: str | Path | None = None,
        output_name_suffix: str = DEFAULT_BITRATE_ONLY_SUFFIX,
        video_codec: str = DEFAULT_VIDEO_CODEC,
        preset: str = DEFAULT_PRESET,
    ) -> Path:
        return cls.compress_video_bitrate_only(
            input_path=input_path,
            target_size_mb=target_size_mb,
            output_path=output_path,
            output_name_suffix=output_name_suffix,
            video_codec=video_codec,
            preset=preset,
        )

    @classmethod
    def manual_test_generic_compression(cls) -> Path:
        input_path = Path(MANUAL_TEST_VIDEO_PATH).expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(
                "Update MANUAL_TEST_VIDEO_PATH to a real local video before running the manual test. "
                f"Current value: {input_path}"
            )

        output_path = cls.compress_video(
            input_path=input_path,
            target_width=1280,
            target_height=720,
            target_fps=30,
            target_video_bitrate_kbps=2000,
        )
        print(f"Input video: {input_path}")
        print(f"Output video: {output_path}")
        return output_path

    @classmethod
    def manual_test_bitrate_only_compression(cls) -> Path:
        input_path = Path(MANUAL_TEST_VIDEO_PATH).expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(
                "Update MANUAL_TEST_VIDEO_PATH to a real local video before running the manual test. "
                f"Current value: {input_path}"
            )

        output_path = cls.compress_video_bitrate_only(
            input_path=input_path,
            target_size_mb=DEFAULT_TARGET_SIZE_MB,
        )
        print(f"Input video: {input_path}")
        print(f"Output video: {output_path}")
        return output_path


if __name__ == "__main__":
    VideoCompressor.manual_test_bitrate_only_compression()
