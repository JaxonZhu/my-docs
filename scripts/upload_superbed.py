#!/usr/bin/env python3
"""Upload one image to Superbed and print a Markdown image link.

The API key comes from SUPERBED_API_KEY or a hidden terminal prompt.
No credentials or upload history are written to disk.
"""

import argparse
import getpass
import json
import mimetypes
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.parse


UPLOAD_URL = "https://www.superbed.cn/upload"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="本地图片路径")
    parser.add_argument("--folder", default="blog", help="图床目录，如 blog/2026/evo-1")
    parser.add_argument("--alt", help="Markdown 图片说明，默认使用文件名")
    parser.add_argument("--url-only", action="store_true", help="仅输出原图 URL")
    args = parser.parse_args()

    path = args.image.expanduser()
    if not path.is_file():
        parser.error(f"图片不存在：{path}")
    content_type = mimetypes.guess_type(path.name)[0] or ""
    if not content_type.startswith("image/"):
        parser.error("请提供 PNG、JPEG、WebP 等图片；此脚本不上传视频。")
    if any(c in str(path) for c in "\r\n"):
        parser.error("图片路径不能包含换行。")
    curl = shutil.which("curl")
    if not curl:
        parser.error("需要 curl（macOS 已自带）。")

    api_key = os.environ.get("SUPERBED_API_KEY", "").strip()
    if not api_key:
        if not sys.stdin.isatty():
            parser.error("请在交互终端运行并输入密钥，或设置 SUPERBED_API_KEY。")
        api_key = getpass.getpass("Superbed API Key（输入不显示）：").strip()
    if not api_key or any(c in api_key for c in "\r\n"):
        parser.error("API Key 为空或包含换行。")

    # Feed the key through stdin, never a process argument or a temporary file.
    escaped_key = api_key.replace("\\", "\\\\").replace('"', '\\"')
    escaped_path = str(path.resolve()).replace("\\", "\\\\").replace('"', '\\"')
    command = [
        curl, "--disable", "--config", "-", "--silent", "--show-error",
        "--connect-timeout", "20", "--max-time", "120", "--request", "POST",
        "--header", "Accept: application/json",
        "--form", f'file=@"{escaped_path}";type={content_type}',
        "--form-string", f'categories={args.folder.strip("/")}',
        "--write-out", "\n%{http_code}", UPLOAD_URL,
    ]
    # Do not follow redirects, load ~/.curlrc, or retry an upload automatically.
    try:
        response = subprocess.run(
            command, input=f'form-string = "token={escaped_key}"\n',
            text=True, capture_output=True, timeout=130, check=False,
        )
        if response.returncode:
            raise ValueError(response.stderr.strip() or "curl 请求失败。")
        body, status = response.stdout.rsplit("\n", 1)
        try:
            result = json.loads(body)
        except ValueError:
            raise ValueError(f"HTTP {status}：服务返回了非 JSON 内容。") from None
        if not isinstance(result, dict):
            raise ValueError("服务返回了非预期的 JSON 结构。")
        if not status.startswith("2"):
            detail = result.get("msg") or result.get("detail") or result.get("message") or "请求被拒绝。"
            raise ValueError(f"HTTP {status}：{detail}")
        if result.get("err") != 0:
            raise ValueError(str(result.get("msg", "上传失败，未返回详细原因。")))
        url = result.get("url")
        if not isinstance(url, str):
            raise ValueError("服务未返回图片 URL。")
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https" or not parsed.netloc or any(c in url for c in "\r\n"):
            raise ValueError("服务未返回有效的 HTTPS 图片地址。")
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        message = str(exc).replace(api_key, "[redacted]")[:300]
        print(f"上传未确认成功：{message}", file=sys.stderr)
        return 1

    if args.url_only:
        print(url)
    else:
        alt = args.alt if args.alt is not None else path.stem
        alt = alt.replace("\r", " ").replace("\n", " ")
        alt = alt.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
        url = url.replace("<", "%3C").replace(">", "%3E")
        print(f"![{alt}](<{url}>)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyboardInterrupt, EOFError):
        print("\n已取消。", file=sys.stderr)
        sys.exit(130)
