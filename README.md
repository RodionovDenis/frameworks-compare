# frameworks-compare
Compare frameworks for hyperparams tune

```
git clone https://github.com/RodionovDenis/frameworks-compare.git
cd frameworks-compare
```
Создаем виртуальную среду, активируем и устанавливаем всё необходимое

```
python -m venv env
source env/bin/activate
pip install -U -r requirements.txt
```

Скачиваем датасеты. Для этого запускаем скрипт:

```
python data/loader.py
```

После завершения в `data` появится папка `datasets`. Чтобы обновить список датасетов (периодически их список будет менятся), нужно запустить этот скрипт еще раз (важно запускать, находясь в корневой папки репозитория).

Запускаем эксперимент, убеждаемся в том, что всё работает:

`python -m experiments.svc`
