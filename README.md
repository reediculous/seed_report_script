# Структура

- `data_parser.py` — парсинг `.txt` файлов с замерами.
- `stats_analysis.py` — описательная статистика, ANOVA/Краскела-Уоллиса, post-hoc, CLD-буквы.
- `plotting.py` — построение графиков (boxplot, гистограммы и т.д.).
- `report_builder.py` — сборка PDF-отчёта.
- `generate_report.py` — главная точка входа.
- `raw_data/example_data/` — пример входных данных.
- `output/example_report.pdf` — пример итогового отчёта.

# Установка

Нужен Python 3.10+.

Требуется запустить следующие команды:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

# Формат входных данных

Файлы складываются в папку `raw_data/` (или любую другую, указанную через `--raw-dir`).

Имя файла должно соответствовать шаблону:

```
<вариант>_<повторность>.txt
```

Например: `НСК-29-19_100sec_1.txt` — вариант `НСК-29-19_100sec`, повторность `1`.

Внутри файла — по одной строке на семя; в строке через табуляцию идут числовые замеры (длина проростка, корней и т.д.). Пустая строка просто игнорируется. Количество непроросших задаётся либо строками с одним 0, либо записью 0=N

Посмотрите пример в `raw_data/example_data/`.

# Запуск

Стандартный запуск (берёт всё из `raw_data/`, сохраняет в `output/report.pdf`):

```bash
python generate_report.py
```

С параметрами:

```bash
python generate_report.py --raw-dir raw_data/example_data --output output/my_report.pdf --title "Отчёт по варианту НСК-29-19"
```

Аргументы:

- `--raw-dir` — папка с `.txt` файлами (по умолчанию `raw_data`).
- `--output` — путь к итоговому PDF (по умолчанию `output/report.pdf`).
- `--title` — заголовок отчёта.

# Результат

После запуска появятся:

- PDF-отчёт по указанному пути (`output/report.pdf` по умолчанию).
- Папка `output/plots/` с PNG-графиками, которые встраиваются в отчёт.

Эталон того, как выглядит готовый отчёт, — `output/example_report.pdf`.

# Тесты

```bash
python -m pytest tests/
```
