# frameworks-compare
Compare frameworks for hyperparams tune

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
