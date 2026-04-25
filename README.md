[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project — Прогнозирование депрессии у учащихся

**Студент:** Щепотьев Андрей Александрович

**Группа:** БИВ238


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуски](#быстрый-старт)
4. [Данные](#данные)
5. [Результаты](#результаты)
7. [Отчёт](#отчёт)


## Описание задачи

<!-- Кратко опишите задачу: что предсказываем, какой датасет, метрика качества -->

**Задача:** бинарная классификация

**Датасет:** [Student Depression Dataset](https://www.kaggle.com/datasets/hopesb/student-depression-dataset/data)

Датасет найден на Kaggle, содержит 27900 строк, 17 столбцов

**Целевая переменная:** Depression (0/1)

**Метрики:** F1-score


## Структура репозитория
Опишите структуру проекта, сохранив при этом верхнеуровневые папки. Можно добавить новые при необходимости.
```
.
├── data
│   ├── processed               # Очищенные и обработанные данные
│   └── raw                     # Исходные файлы
├── models                      # Сохранённые модели 
├── notebooks
│   ├── 01_eda.ipynb            # EDA
│   ├── 02_baseline.ipynb       # Baseline-модель
├── presentation                # Презентация для защиты
├── report
├── src
├── tests
├── requirements.txt
└── README.md
```

## Запуск

Этот блок замените способом запуска вашего сервиса.
```bash
# 1. Клонировать репозиторий
git clone <url>
cd <repo-name>

# 2. Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Установить зависимости
pip install -r requirements.txt
```

## Данные
- `data/raw/` — исходные файлы
- `data/processed/` — предобработанные данные


## Результаты
Здесь коротко выпишите результаты.
| Модель | F1-Score (класс 1) | Accuracy | Примечание |
|--------|--------------------|----------|------------|
| Baseline | 0.87               | 0.85     | Логистическая регрессия |
| Ансамбль (Voting) | 0.88               | 0.85     | Лучшая комбинация моделей |

## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
