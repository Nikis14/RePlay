# RePlay

В библиотеке RePlay реализовано множество алгоритмов для создания моделей рекомендательных систем. Инструменты библиотеки позволяют проводить эксперименты по сравнению различных алгоритмов и параметров модели по выбранным метрикам и времени работы.  

### Алгоритмы, реализованные в RePlay:
* Popular Recommender
* Wilson Recommender
* Random Recommender
* K-Nearest Neighbours
* Classifier Recommender
* Alternating Least Squares
* Neural Matrix Factorization
* SLIM
* ADMM SLIM
* Mult-VAE
* Word2Vec Recommender
* Обертка LightFM
* Обертка implicit

Больше информации об алгоритмах - в документации к replay.

### Метрики
В библиотеке реализованы основные метрики для оценки качества рекомендательных систем, такие как **HitRate, Precision, MAP, Recall, ROC-AUC, MRR, NDCG, Surprisal, Unexpectedness, Coverage**. Кроме того пользователь может создать свою метрику.
Все метрики рассчитываются для первых K объектов в рекомендации.

### Сценарии 
Библиотека позволяет запускать эксперименты по сравнению эффективности отдельных алгоритмов на пользовательских данных для выбора лучшего алгоритма и его параметров.
Кроме того, функциональность библиотеки позволяет запускать сценарии, объединяющие в себе основные этапы создания рекомендательной системы:

* разбиение данных сплиттером на обучающую и валидационную выборки
* подбор гипер-параметров с помощью optuna
* расчёт метрик качества для полученных моделей-кандидатов
* обучение на всём объёме данных с подобранными гипер-параметрами и отгрузка рекомендаций (batch production)

## Как начать пользоваться библиотекой

### Установка
Рекомендуется использовать Unix машину с python >= 3.6.

```bash
git clone https://sbtatlas.sigma.sbrf.ru/stash/scm/ailab/replay.git
cd replay
pip install poetry
poetry install
```

### Проверка работы библиотеки
Можно проверить корректность установки, запустив тесты. 
Из директории `replay`:
```bash
pytest ./tests
```

### Документация
Документация по библиотеке пока не выложена в открытом доступе, но ее можно сформировать самостоятельно после установки пакета.
Из директории `replay`:
```bash
cd ./docs
mkdir -p _static
make clean html
```
Документация будет доступна в `replay/docs/_build/html/index.html`


## Как присоединиться к разработке
Тестирующий скрипт, используемый для формирования документации, проверки codestyle и запуска тестов, предполагает, что в директории с библиотекой создана виртуальная среда python с именем venv. Поэтому ниже в коде есть создание среды, которое можно выполнить другим удобным образом. 
Клонируем репозиторий:
```bash
git clone https://sbtatlas.sigma.sbrf.ru/stash/scm/ailab/replay.git
cd replay
```
Создаем и активируем виртуальную среду:
```bash
python3 -m venv .
source ./bin/activate
```
Устанавливаем библиотеку:
```bash
pip install poetry
poetry install
```

## Проверка работы библиотеки
Можно проверить корректность установки, запустив тесты. 
```bash
pytest ./tests
```
Кроме того, можно запустить тестирующий скрипт, который сформирует документацию, проверит код pylint, pycodestyle и запустит тесты.
Для корректной работы скрипта в директории `replay` должна быть создана виртуальная среда venv.
```bash
./test_package.sh
```

## Проверка качества кода
Настрой pre-commit hooks для автоматического форматирования и проверки кода:  
Из директории `replay`:
```bash
pre-commit install
```
