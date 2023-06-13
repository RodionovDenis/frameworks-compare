# frameworks-compare
[![python: 3.10](https://img.shields.io/badge/python-3.10-44cc12?style=flat-square&logo=python)](https://www.python.org/downloads/release/python-3106/)

```
git clone https://github.com/RodionovDenis/frameworks-compare.git
cd frameworks-compare
```
Создаем виртуальную среду

```
python -m venv env
```

Активируем ее
```
Linux:
source env/bin/activate

Windows:
./env/Scripts/activate
```
Устанавливаем всё необходимое

```
pip install -U -r requirements.txt
```

Скачиваем датасеты. Для этого запускаем скрипт:

```
python data/loader.py
```

После завершения в `data` появится папка `datasets`. Чтобы обновить список датасетов (периодически их список будет менятся), нужно запустить этот скрипт еще раз.

Запускаем эксперимент, проверяем, что всё работает (запускать с корневой папки репозитория):

`python experiments/svc.py`
