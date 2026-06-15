# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# src/flow_factory/logger/abc.py
import json
import os
import imageio
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Optional, Dict, Any, List

from PIL import Image as PILImage

from ..hparams import *
from .formatting import LogFormatter, LogImage, LogVideo, LogTable


class Logger(ABC):
    platform: Any

    def __init__(self, config: Arguments):
        self.config = config
        self.clean_up_freq = 10
        self._pending_cleanup: List[Dict] = []
        self._init_platform()

    @abstractmethod
    def _init_platform(self):
        pass

    # ---- local media / metrics helpers ----

    @property
    def _should_save_locally(self) -> bool:
        return getattr(self.config.log_args, 'save_media_locally', False)

    @property
    def _should_log_jsonl(self) -> bool:
        return getattr(self.config.log_args, 'log_metrics_jsonl', True)

    @property
    def _logs_dir(self) -> str:
        if not hasattr(self, '_logs_dir_cache'):
            path = os.path.join(
                self.config.log_args.save_dir,
                self.config.log_args.run_name,
                'logs',
            )
            os.makedirs(path, exist_ok=True)
            self._logs_dir_cache = path
        return self._logs_dir_cache

    @staticmethod
    def _sanitize_key(key: str) -> str:
        return key.replace('/', '_')

    def _save_image_file(self, key: str, img_obj: LogImage, step: int) -> str:
        """Save a LogImage locally, return relative path."""
        fmt = getattr(self.config.log_args, 'image_save_format', 'png').lower()
        quality = getattr(self.config.log_args, 'image_save_quality', 90)
        ext = 'jpg' if fmt == 'jpg' else 'png'
        sanitized = self._sanitize_key(key)
        step_dir = os.path.join(self._logs_dir, 'images', f'step_{step:06d}')
        os.makedirs(step_dir, exist_ok=True)
        filepath = os.path.join(step_dir, f'{sanitized}.{ext}')

        img = img_obj.get_pil().convert('RGB')
        if fmt == 'jpg':
            img.save(filepath, format='JPEG', quality=quality)
        else:
            img.save(filepath, format='PNG')
        return os.path.relpath(filepath, self._logs_dir)

    def _save_video_file(self, key: str, vid_obj: LogVideo, step: int) -> str:
        """Save a LogVideo as MP4, return relative path."""
        sanitized = self._sanitize_key(key)
        step_dir = os.path.join(self._logs_dir, 'videos', f'step_{step:06d}')
        os.makedirs(step_dir, exist_ok=True)
        filepath = os.path.join(step_dir, f'{sanitized}.mp4')
        arr = vid_obj.get_numpy()  # THWC uint8
        imageio.mimwrite(filepath, [f for f in arr], fps=vid_obj.fps, format='FFMPEG',
                         codec='libx264', pixelformat='yuv420p')
        return os.path.relpath(filepath, self._logs_dir)

    def _extract_and_save_media(self, data: Dict, step: int):
        """Walk dict, save media to local files, and remove from dict."""
        entries = []

        def _walk(key: str, value: Any):
            if isinstance(value, (LogImage, LogVideo)):
                if isinstance(value, LogImage):
                    path = self._save_image_file(key, value, step)
                else:
                    path = self._save_video_file(key, value, step)
                entry = {'step': step, 'key': key, 'path': path, 'caption': value.caption}
                if value.metadata:
                    entry.update(value.metadata)
                entries.append(entry)
                return None
            elif isinstance(value, LogTable):
                for row_idx, row in enumerate(value.rows):
                    for col_idx, item in enumerate(row):
                        if item is not None:
                            col_name = value.columns[col_idx] if col_idx < len(value.columns) else str(col_idx)
                            _walk(f'{key}/{col_name}/{row_idx}', item)
                return None
            elif isinstance(value, list):
                return [_walk(f'{key}/{i}', v) for i, v in enumerate(value)]
            elif isinstance(value, dict):
                return {k: _walk(f'{key}/{k}', v) for k, v in value.items()}
            return value

        for k in list(data.keys()):
            data[k] = _walk(k, data[k])

        # Remove sample-list keys (now all-None after media extraction)
        for k in list(data.keys()):
            if k == 'train_samples' or k.startswith('eval/') and k.endswith('/samples'):
                del data[k]

        if entries:
            filepath = os.path.join(self._logs_dir, 'media.jsonl')
            with open(filepath, 'a') as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    def _write_metrics_jsonl(self, scalars: Dict[str, Any], step: int):
        record = {'step': step}
        for k, v in scalars.items():
            scalar = LogFormatter.to_scalar(v)
            if scalar is not None:
                record[k] = scalar
            elif isinstance(v, (list, dict)):
                record[k] = v
        if len(record) > 1:  # more than just 'step'
            filepath = os.path.join(self._logs_dir, 'metrics.jsonl')
            with open(filepath, 'a') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

    # ---- main log flow ----

    def log_data(
        self,
        data: Dict[str, Any],
        step: int,
        keys: Optional[str] = None,
    ):
        # 1. Process rules (Mean, Paths, wrappers) into IR
        formatted_dict = LogFormatter.format_dict(data)

        # 2. [NEW] Save media locally if configured (also removes media from dict)
        if self._should_save_locally:
            self._extract_and_save_media(formatted_dict, step)

        # 3. [NEW] Write scalar metrics to local JSONL
        if self._should_log_jsonl:
            self._write_metrics_jsonl(formatted_dict, step)

        # 4. Remove non-scalar values (nested structures only meaningful for local JSONL)
        formatted_dict = {k: v for k, v in formatted_dict.items() if not isinstance(v, (list, dict))}

        # 5. Filter keys if requested
        if keys:
            valid_keys = keys.split(',')
            formatted_dict = {k: v for k, v in formatted_dict.items() if k in valid_keys}

        # 6. Convert IR to Platform Objects
        final_dict = {}
        for k, v in formatted_dict.items():
            converted = self._recursive_convert(v)
            if isinstance(converted, dict):
                final_dict.update(converted)
            else:
                final_dict[k] = converted

        # 7. Actual Logging (filter out None from removed media)
        final_dict = {k: v for k, v in final_dict.items() if v is not None}
        if final_dict:
            self._log_impl(final_dict, step)

        # 8. Cleanup temporary files periodically
        if not self._should_save_locally:
            if len(self._pending_cleanup) >= self.clean_up_freq:
                first_data = self._pending_cleanup.pop(0)
                self._cleanup_temp_files(first_data)
            self._pending_cleanup.append(formatted_dict)

    def _recursive_convert(
        self,
        value: Any,
        height: Optional[int] = None,
        width: Optional[int] = None
    ) -> Any:
        """Recursively convert IR objects to platform objects."""
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return [self._recursive_convert(v, height, width) for v in value if v is not None]
        return self._convert_to_platform(value, height, width)

    def _cleanup_temp_files(self, data: Dict):
        for value in data.values():
            if isinstance(value, (LogImage, LogVideo, LogTable)):
                value.cleanup()
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, (LogImage, LogVideo, LogTable)):
                        item.cleanup()

    @abstractmethod
    def _convert_to_platform(
        self,
        value: Any,
        height: Optional[int] = None,
        width: Optional[int] = None
    ) -> Any:
        """
        Convert a single IR object to platform-specific object.

        Args:
            value: IR object (LogImage, LogVideo, LogTable) or pass-through value.
            height: Optional target height for resize (aspect-ratio preserved if width is None).
            width: Optional target width for resize (aspect-ratio preserved if height is None).

        Returns:
            Platform-specific object (e.g., wandb.Image, swanlab.Video).
        """
        pass

    @abstractmethod
    def _log_impl(self, data: Dict, step: int):
        pass
