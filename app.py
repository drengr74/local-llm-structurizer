"""Структурирование строк по схеме из schemas.json через локальную Ollama.

Запуск интерфейса:
    python app.py

Запуск в терминале, как раньше:
    python app.py --cli
    python app.py --cli --schema person
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import ollama

MODEL_NAME = "llama3.2"
BASE_DIR = Path(__file__).resolve().parent
SCHEMAS_PATH = BASE_DIR / "schemas.json"
INPUT_PATH = BASE_DIR / "input.txt"
OUTPUT_PATH = BASE_DIR / "output.json"
WEB_PATH = BASE_DIR / "web" / "index.html"
UI_HOST = "127.0.0.1"
UI_PORT = 8765

# Ответ модели иногда оборачивается в ```json ... ```.
_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*\n?(.*?)\n?```\s*$",
    re.DOTALL | re.IGNORECASE,
)
_SCHEMA_LABELS = ("person", "product", "unknown")
_DETECT_PROMPT = (
    "Analyze the text and determine which schema it belongs to: 'person' or 'product'. "
    "If it belongs to neither, return 'unknown'. "
    "Answer with ONLY one word: person, product, or unknown"
)


def load_schemas(path: Path) -> dict[str, Any]:
    """Загружает словарь схем из JSON-файла."""
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"В {path} ожидается непустой объект со схемами.")
    return data


def read_lines(path: Path) -> list[str]:
    """Читает непустые строки входного файла."""
    with path.open(encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def build_system_prompt(schema_name: str, schema: dict[str, Any]) -> str:
    """Собирает системный промпт так, чтобы модель вернула только JSON по схеме."""
    fields = schema.get("fields", {})
    description = schema.get("description", schema_name)
    field_lines = "\n".join(f"- {name}: {hint}" for name, hint in fields.items())
    keys = ", ".join(fields.keys())
    return (
        "Ты извлекаешь структурированные данные из текста.\n"
        f"Схема «{schema_name}»: {description}\n"
        "Поля:\n"
        f"{field_lines}\n\n"
        "Правила ответа:\n"
        f"- Верни один JSON-объект только с ключами: {keys}.\n"
        "- Если значение в тексте не найдено, поставь null.\n"
        "- Числа возвращай числами, не строками.\n"
        "- Если текст не относится к этой схеме, всё равно верни объект "
        "с теми же ключами и null там, где данных нет.\n"
        "- Никакого markdown, комментариев и текста вне JSON."
    )


def strip_markdown(raw: str) -> str:
    """Убирает обёртку ```json ... ``` и лишние пробелы."""
    text = raw.strip()
    fenced = _FENCE_RE.match(text)
    if fenced:
        return fenced.group(1).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_json_object(raw: str) -> dict[str, Any]:
    """Парсит ответ модели в словарь. Допускает мусор вокруг единственного объекта."""
    cleaned = strip_markdown(raw)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end <= start:
            raise
        parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Модель вернула JSON, но это не объект.")
    return parsed


def align_to_schema(data: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Оставляет только поля схемы и подставляет null для пропусков."""
    fields = schema.get("fields", {})
    return {name: data.get(name) for name in fields}


def detect_schema(text: str) -> str:
    """Короткий запрос в Ollama: person, product или unknown."""
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": _DETECT_PROMPT},
            {"role": "user", "content": text},
        ],
        options={"temperature": 0, "num_predict": 8},
    )
    raw = strip_markdown(str(response["message"]["content"]))
    token = raw.strip().strip("`'\".").lower()
    token = token.split()[0] if token else ""
    token = token.strip(".,:;!")
    if token in _SCHEMA_LABELS:
        return token
    for label in _SCHEMA_LABELS:
        if re.search(rf"\b{label}\b", raw, flags=re.IGNORECASE):
            return label
    return "unknown"


def extract_with_ollama(text: str, schema_name: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Отправляет строку в llama3.2 и возвращает объект по выбранной схеме."""
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": build_system_prompt(schema_name, schema)},
            {"role": "user", "content": text},
        ],
        format="json",
        options={"temperature": 0},
    )
    content = response["message"]["content"]
    parsed = parse_json_object(content)
    return align_to_schema(parsed, schema)


def process_one(
    index: int,
    line: str,
    schemas: dict[str, Any],
    schema_name: str | None,
) -> dict[str, Any]:
    """Обрабатывает одну строку. Ошибка возвращается в записи, а не наружу."""
    mode = "manual" if schema_name else "auto"
    active_schema = schema_name or "unknown"
    try:
        if mode == "auto":
            active_schema = detect_schema(line)
            print(f"[{index}] схема: {active_schema}")
            if active_schema == "unknown":
                return {
                    "line_number": index,
                    "source": line,
                    "detected_schema": mode,
                    "schema": "unknown",
                    "message": "данные не распознаны",
                }
            schema = schemas[active_schema]
        else:
            if schema_name is None:
                raise ValueError("Не выбрана схема.")
            schema = schemas[schema_name]

        if not isinstance(schema, dict) or "fields" not in schema:
            raise ValueError(f"Схема «{active_schema}» должна содержать объект fields.")

        data = extract_with_ollama(line, active_schema, schema)
        print(f"[{index}] ok")
        return {
            "line_number": index,
            "source": line,
            "detected_schema": mode,
            "schema": active_schema,
            "data": data,
        }
    except Exception as error:  # noqa: BLE001 — строка не должна ронять весь прогон
        print(f"[{index}] ошибка: {error}", file=sys.stderr)
        return {
            "line_number": index,
            "source": line,
            "detected_schema": mode,
            "schema": active_schema,
            "error": str(error),
        }


def process_lines(
    lines: list[str],
    schemas: dict[str, Any],
    schema_name: str | None,
) -> list[dict[str, Any]]:
    """Обрабатывает строки по одной. Ошибка в строке не останавливает остальные.

    schema_name is None в режиме auto: схема определяется для каждой строки.
    """
    return [process_one(index, line, schemas, schema_name) for index, line in enumerate(lines, start=1)]


def save_results(path: Path, results: list[dict[str, Any]]) -> None:
    """Сохраняет список объектов в output.json."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
        file.write("\n")


def parse_args(schema_names: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Структурирует строки по схеме через Ollama. Без --cli открывает окно в браузере.",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Обработать input.txt в терминале, без интерфейса.",
    )
    parser.add_argument(
        "--schema",
        nargs="?",
        const="auto",
        default="auto",
        choices=[*schema_names, "auto"],
        help="Схема из schemas.json. Без аргумента или auto — определить схему для каждой строки.",
    )
    return parser.parse_args()


def split_lines(text: str) -> list[str]:
    """Делит текст интерфейса на непустые строки."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def read_output() -> list[Any]:
    """Возвращает прошлый output.json или пустой список, если файла ещё нет."""
    if not OUTPUT_PATH.exists():
        return []
    with OUTPUT_PATH.open(encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, list) else []


def build_state(schemas: dict[str, Any]) -> dict[str, Any]:
    """Данные для первой отрисовки интерфейса."""
    try:
        source = INPUT_PATH.read_text(encoding="utf-8")
    except OSError:
        source = ""
    try:
        previous = read_output()
    except (OSError, json.JSONDecodeError):
        previous = []
    return {
        "model": MODEL_NAME,
        "schemas": schemas,
        "source": source,
        "results": previous,
    }


class AppHandler(BaseHTTPRequestHandler):
    """Локальный интерфейс и API обработки строк."""

    schemas: dict[str, Any] = {}

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            try:
                page = WEB_PATH.read_bytes()
            except OSError as error:
                self._send_json(500, {"error": f"Не найден интерфейс: {error}"})
                return
            self._send(200, page, "text/html; charset=utf-8")
            return
        if path == "/api/state":
            self._send_json(200, build_state(self.schemas))
            return
        self._send_json(404, {"error": "Не найдено"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path != "/api/process":
            self._send_json(404, {"error": "Не найдено"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(400, {"error": "Некорректный запрос"})
            return

        schema_choice = str(payload.get("schema", "auto"))
        allowed = {*self.schemas.keys(), "auto"}
        if schema_choice not in allowed:
            self._send_json(400, {"error": "Неизвестная схема"})
            return
        lines = split_lines(str(payload.get("text", "")))
        if not lines:
            self._send_json(400, {"error": "Нет строк для обработки"})
            return

        manual_name = None if schema_choice == "auto" else schema_choice
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        results: list[dict[str, Any]] = []
        try:
            for index, line in enumerate(lines, start=1):
                item = process_one(index, line, self.schemas, manual_name)
                results.append(item)
                self.wfile.write(
                    (json.dumps({"type": "line", "result": item, "total": len(lines)}, ensure_ascii=False) + "\n").encode(
                        "utf-8"
                    )
                )
                self.wfile.flush()
            save_results(OUTPUT_PATH, results)
            self.wfile.write(
                (json.dumps({"type": "done", "count": len(results)}, ensure_ascii=False) + "\n").encode("utf-8")
            )
            self.wfile.flush()
        except (OSError, BrokenPipeError) as error:
            print(f"Не удалось отдать результат: {error}", file=sys.stderr)


def serve_ui() -> None:
    """Поднимает локальный интерфейс и открывает его в браузере."""
    if not WEB_PATH.exists():
        print(f"Не найден файл интерфейса: {WEB_PATH}", file=sys.stderr)
        sys.exit(1)

    port = UI_PORT
    server: ThreadingHTTPServer | None = None
    for candidate in range(UI_PORT, UI_PORT + 20):
        try:
            server = ThreadingHTTPServer((UI_HOST, candidate), AppHandler)
            port = candidate
            break
        except OSError:
            continue
    if server is None:
        print("Не удалось занять порт для интерфейса.", file=sys.stderr)
        sys.exit(1)

    url = f"http://{UI_HOST}:{port}"
    print(f"Интерфейс: {url}")
    print("Закрыть: Ctrl+C")
    threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановлено.")
    finally:
        server.server_close()


def run_cli(schemas: dict[str, Any], schema_choice: str) -> None:
    """Прежний режим: прочитать input.txt и записать output.json."""
    manual_name: str | None = None if schema_choice == "auto" else schema_choice
    if manual_name is not None:
        schema = schemas[manual_name]
        if not isinstance(schema, dict) or "fields" not in schema:
            print(f"Схема «{manual_name}» должна содержать объект fields.", file=sys.stderr)
            sys.exit(1)

    try:
        lines = read_lines(INPUT_PATH)
    except OSError as error:
        print(f"Не удалось прочитать {INPUT_PATH}: {error}", file=sys.stderr)
        sys.exit(1)

    if not lines:
        print(f"В {INPUT_PATH} нет строк для обработки.", file=sys.stderr)
        sys.exit(1)

    results = process_lines(lines, schemas, manual_name)
    try:
        save_results(OUTPUT_PATH, results)
    except OSError as error:
        print(f"Не удалось записать {OUTPUT_PATH}: {error}", file=sys.stderr)
        sys.exit(1)

    print(f"Готово: {len(results)} строк → {OUTPUT_PATH}")


def main() -> None:
    try:
        schemas = load_schemas(SCHEMAS_PATH)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"Не удалось загрузить схемы: {error}", file=sys.stderr)
        sys.exit(1)

    AppHandler.schemas = schemas
    args = parse_args(list(schemas.keys()))
    if args.cli:
        run_cli(schemas, args.schema)
        return
    serve_ui()


if __name__ == "__main__":
    main()
