# StyleGAN-NADA: адаптация генератора лиц в аниме-домен

Проект адаптирует предобученный StyleGAN2-ADA, генерирующий фотореалистичные
лица, к новому домену по текстовому описанию. Для обучения не требуется датасет
аниме: сигнал даёт CLIP, а один и тот же latent-код подаётся в замороженный
`G_source` и обучаемый `G_target`.

![Результат после 500 шагов](assets/training_step_499.jpg)

В верхнем ряду показаны исходные лица, в нижнем - результаты адаптированного
генератора для тех же latent-кодов.

## Постановка задачи

StyleCLIP хорошо подходит для редактирования отдельного изображения или
направления в latent/style space, но сам по себе не переводит всю
распределённую генерацию StyleGAN в новый домен. В этом проекте используется
идея StyleGAN-NADA: обучается копия генератора, а directional CLIP loss
согласует направление изменения изображений с направлением между текстами
`photo portrait` и `anime portrait`.

## Что находится в репозитории

```text
configs/anime_t4.yaml                 конфигурация эксперимента
notebooks/01_train_colab.ipynb       обучение в Colab
notebooks/02_inference_colab.ipynb   валидация и инференс без обучения
src/stylegan_nada_anime/             функции, классы и основная логика
scripts/train.py                     CLI для обучения
scripts/infer.py                     CLI для фото или случайных seed
scripts/validate.py                  CLIP/structure-валидация
app.py                               Gradio-сервис только для инференса
REPORT.md                            полный текст отчёта
REPORT.docx / REPORT.pdf             оформленные версии отчёта
assets/                              реальные результаты запуска ноутбука
```

## Веса моделей

| Модель | Ссылка | Назначение |
|---|---|---|
| FFHQ StyleGAN2-ADA 256 px | https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res256-mirror-paper256-noaug.pkl | исходный `G_source`, скачивается автоматически |
| Адаптированный генератор этого проекта | YOUR_MODEL_WEIGHTS_URL | заменить после загрузки `stylegan_nada_anime_generator.pkl` на Google Drive/Hugging Face |
| Официальная reference-модель StyleGAN-NADA anime | https://huggingface.co/rinong/stylegan-nada-models/blob/main/anime.pt | пример весов оригинальной rosinality-реализации; формат несовместим с данным NVlabs-пайплайном |

> Единственное действие перед публикацией GitHub: загрузить полученный `.pkl`
> и заменить `YOUR_MODEL_WEIGHTS_URL`. Сам файл весов отсутствовал в переданном
> `.ipynb`, поэтому корректную ссылку невозможно создать автоматически.

## Быстрый запуск обучения в Google Colab

1. Откройте `notebooks/01_train_colab.ipynb` через кнопку **Open in Colab**.
2. В первой ячейке замените `YOUR_USERNAME` на имя GitHub-аккаунта.
3. Включите T4 GPU.
4. Запускайте ячейки сверху вниз.

На T4 используется разрешение 256 px, batch size 1 и один CLIP ViT-B/32.
Результаты, checkpoints и итоговый `.pkl` сохраняются в
`/content/stylegan_nada_anime`. Повторный запуск автоматически продолжает
обучение с последнего checkpoint.

## Валидационный ноутбук: только инференс

`notebooks/02_inference_colab.ipynb`:

- клонирует проект и StyleGAN2-ADA;
- скачивает готовый `.pkl` по Google Drive/Hugging Face URL либо принимает файл;
- генерирует фиксированную сетку по seed;
- считает CLIP direction alignment, target CLIP similarity и structure L1;
- принимает реальный портрет, выполняет W+ проекцию и возвращает аниме-версию.

В нём нет вызова `StyleGANNADATrainer` и не выполняется обновление весов.

## Локальный запуск через Conda

```bash
git clone https://github.com/YOUR_USERNAME/stylegan-nada-anime.git
cd stylegan-nada-anime
conda env create -f environment.yml
conda activate stylegan-nada-anime
pip install -e .
git clone --depth 1 https://github.com/NVlabs/stylegan2-ada-pytorch.git vendor/stylegan2-ada-pytorch
```

Обучение:

```bash
python scripts/train.py --config configs/anime_t4.yaml
```

Перед локальным запуском поменяйте `stylegan_repo` и `output_dir` в YAML на
локальные пути.

Инференс для случайных seed:

```bash
python scripts/infer.py \
  --model weights/stylegan_nada_anime_generator.pkl \
  --stylegan-repo vendor/stylegan2-ada-pytorch \
  --output-dir outputs/random
```

Инференс для фотографии:

```bash
python scripts/infer.py \
  --model weights/stylegan_nada_anime_generator.pkl \
  --stylegan-repo vendor/stylegan2-ada-pytorch \
  --input examples/person.jpg \
  --projection-steps 300 \
  --output-dir outputs/person
```

## Как устроено обучение

1. `G_source` загружается из FFHQ и полностью замораживается.
2. `G_target` создаётся как копия `G_source`.
3. Mapping network, affine-параметры и ToRGB не обучаются.
4. Короткая W+-оптимизация оценивает, какие resolution-блоки сильнее участвуют
   в требуемом текстовом изменении.
5. Обновляются только convolution weights выбранных блоков.
6. Основной loss состоит из directional CLIP loss, negative-prompt margin и
   low-frequency structure loss.
7. Сохраняются preview, checkpoints, JSON-история и итоговый генератор.

При первом выборе в переданном запуске были активированы блоки
`4, 8, 64, 128`; около 9.29 млн параметров из 18.84 млн кандидатов получили
право на обновление. Набор пересчитывается каждые 50 шагов.

## Примеры результатов

![Случайные лица до и после](assets/random_source_and_anime.jpg)

![График обучения](assets/training_curve.png)

Наблюдаемый результат: модель уверенно меняет глаза, форму носа и рта,
добавляет line art и cel-shading, сохраняя позу и общую композицию. Ограничения
видны в виде лишних линий под подбородком, стилизации фона и частичной потери
индивидуальности. Эти недостатки отдельно разобраны в отчёте.

## Веб-сервис и запись экрана

Сервис выполняет только инференс и загружает готовые веса при старте:

```bash
export MODEL_PATH=weights/stylegan_nada_anime_generator.pkl
export STYLEGAN_REPO=vendor/stylegan2-ada-pytorch
python app.py
```

Для демонстрационного видео:

1. Запустите `app.py` или последнюю ячейку inference-ноутбука.
2. Начните запись экрана.
3. Покажите, что обучение не запускается.
4. Загрузите фронтальный портрет.
5. Запустите преобразование и покажите три изображения: вход, реконструкция,
   аниме.
6. Завершите запись после появления результата.

## Ссылки

- StyleGAN-NADA: https://stylegan-nada.github.io/
- Официальный код StyleGAN-NADA: https://github.com/rinongal/StyleGAN-nada
- StyleCLIP: https://github.com/orpatashnik/StyleCLIP
- StyleGAN2-ADA PyTorch: https://github.com/NVlabs/stylegan2-ada-pytorch

## Ограничения

- Метод зависит от формулировки prompt и знаний CLIP.
- Сильная смена геометрии может ухудшить сохранение личности.
- Реальные фото ограничены качеством GAN inversion.
- Центрированный crop не заменяет FFHQ landmark alignment.
- 256 px экономит память T4, но ограничивает мелкие детали.
- Веса и данные могут наследовать смещения исходных моделей.
