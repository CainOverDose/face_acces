# utils/visualizer.py

# Импорт библиотеки OpenCV (cv2) для рисования на изображениях.
import cv2

# Импорт стандартной библиотеки logging.
# Она используется для вывода сообщений о состоянии программы (отладка, ошибки).
import logging

# --- Глобальные переменные / Константы модуля (цвета, шрифты, толщины линий) ---

# Цвета в формате BGR (Blue, Green, Red), используемом OpenCV.
# (0, 255, 0) - зеленый
# (0, 0, 255) - красный
# (255, 0, 0) - синий
COLOR_KNOWN = (0, 255, 0)    # Цвет рамки для известного (распознанного) лица
COLOR_UNKNOWN = (0, 0, 255)  # Цвет рамки для неизвестного лица
COLOR_TEXT_BG = (0, 0, 0)    # Цвет фона для текста (черный)

# Шрифт для отображения текста.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 1

# Толщина линии для прямоугольника.
RECT_THICKNESS = 2

# --- Функции модуля ---

def draw_face_box_and_label(frame, box, label: str, confidence: float = None):
    """
     Рисует прямоугольник вокруг лица и добавляет подпись (label) на кадре.

     Эта функция берет кадр, координаты прямоугольника, метку (например, ID пользователя или "Неизвестное лицо")
     и (опционально) уверенность, и отрисовывает их на изображении для визуализации.

     Args:
         frame (numpy.ndarray): Кадр изображения (NumPy массив), на котором нужно нарисовать элементы.
                                Это изображение будет *изменено* функцией.
         box (list or tuple): Список или кортеж из 4-х целых чисел [x1, y1, x2, y2],
                              представляющий координаты ограничивающего прямоугольника (левый верхний, правый нижний).
         label (str): Текст для подписи (например, "ID_001", "John Doe", "Неизвестное лицо").
         confidence (float, optional): Уверенность модели (например, 0.85). Если указана, будет добавлена к подписи.

     Raises:
         TypeError: Если frame не является numpy.ndarray или box не является списком/кортежем.
         ValueError: Если box не содержит 4 элемента или содержит некорректные координаты.
     """
    # Проверка типа frame.
    if not isinstance(frame, type(None)) and not hasattr(frame, 'ndim'):
        error_msg = f"Аргумент 'frame' должен быть массивом NumPy (numpy.ndarray). Получено: {type(frame)}"
        logging.error(error_msg)
        raise TypeError(error_msg)

    if frame is None:
        logging.warning("Попытка рисовать на пустом кадре (frame is None).")
        return # Нечего рисовать

    # Проверка типа box.
    if not isinstance(box, (list, tuple)):
        error_msg = f"Аргумент 'box' должен быть списком или кортежем. Получено: {type(box)}"
        logging.error(error_msg)
        raise TypeError(error_msg)

    # Проверка длины box.
    if len(box) != 4:
        error_msg = f"Аргумент 'box' должен содержать 4 координаты [x1, y1, x2, y2]. Получено: {len(box)} элементов"
        logging.error(error_msg)
        raise ValueError(error_msg)

    # Проверка, что все элементы box - целые числа.
    try:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    except (ValueError, TypeError):
        error_msg = f"Координаты в 'box' должны быть числами. Получено: {box}"
        logging.error(error_msg)
        raise ValueError(error_msg)

    # Выбор цвета рамки в зависимости от метки.
    # Если метка содержит "Неизвестный" или "Unknown" (регистронезависимо), используем красный цвет.
    # Иначе - зеленый. Это можно изменить в зависимости от логики распознавания.
    color = COLOR_UNKNOWN if 'Неизвестный'.lower() in label.lower() or 'Unknown'.lower() in label.lower() else COLOR_KNOWN

    # --- 1. Рисуем прямоугольник ---
    # cv2.rectangle(img, pt1, pt2, color, thickness)
    # img: изображение
    # pt1: верхний левый угол (x1, y1)
    # pt2: нижний правый угол (x2, y2)
    # color: цвет в формате BGR
    # thickness: толщина линии
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, RECT_THICKNESS)

    # --- 2. Подготовим текст для подписи ---
    # Если confidence указана, добавим её к метке.
    text = f"{label}" + (f" ({confidence:.2f})" if confidence is not None else "")

    # --- 3. Рисуем фон для текста (прямоугольник) ---
    # Сначала нужно измерить размер текста, чтобы нарисовать фон нужного размера.
    # cv2.getTextSize(text, font, fontScale, thickness) возвращает размеры текста и базовую линию.
    (text_width, text_height), baseline = cv2.getTextSize(text, FONT_FACE, FONT_SCALE, FONT_THICKNESS)

    # Позиция текста - чуть выше прямоугольника лица.
    text_x = x1
    text_y = y1 - baseline - 1 # -1 для небольшого отступа

    # Координаты фона текста.
    bg_x1, bg_y1 = text_x, text_y - text_height
    bg_x2, bg_y2 = text_x + text_width, text_y + baseline

    # Рисуем прямоугольник для фона текста.
    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), COLOR_TEXT_BG, thickness=cv2.FILLED)

    # --- 4. Рисуем сам текст ---
    # cv2.putText(img, text, org, fontFace, fontScale, color, thickness)
    # org: координаты нижней левой точки текста
    cv2.putText(frame, text, (text_x, text_y), FONT_FACE, FONT_SCALE, color, FONT_THICKNESS)

    logging.debug(f"Отрисован прямоугольник и подпись '{label}' для bbox {box}.")

# --- Модуль как исполняемый скрипт (опционально, для тестирования) ---
# if __name__ == "__main__":
#     # Пример использования функции для тестирования.
#     import numpy as np
#     # Создадим пустое "изображение" 480x640x3 (BGR)
#     test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
#     draw_face_box_and_label(test_frame, [100, 100, 200, 200], "Test User", 0.95)
#     # cv2.imshow("Test Visualization", test_frame)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()