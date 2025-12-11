# utils/camera.py

# Импорт библиотеки OpenCV (cv2) для работы с видеозахватом.
import cv2

# Импорт стандартной библиотеки logging.
# Она используется для вывода сообщений о состоянии программы (отладка, ошибки).
import logging

# --- Классы модуля ---

class CameraCapture: 
    """
     Класс для управления видеозахватом с камеры (локальной или IP-камеры).

     Этот класс инкапсулирует логику подключения к камере, захвата кадров
     и корректного освобождения ресурсов.
     """

    def __init__(self, source): 
        """
         Инициализирует объект CameraCapture.

         Args:
             source: Источник видео.
                     Может быть:
                     - Целым числом (например, 0) для доступа к локальной камере (обычно встроенной).
                     - Строкой с URL (например, "rtsp://user:password@ip:port/stream") для IP-камеры.
         """
        self.source = source 
        # Объект VideoCapture из OpenCV, который управляет соединением с камерой.
        self.cap = None
        # Флаг, показывающий, успешно ли установлено соединение.
        self.is_connected = False
        # Попытка подключения при инициализации.
        self.connect()

    def connect(self):
        """
         Пытается подключиться к камере по указанному источнику.

         Устанавливает соединение и обновляет флаг is_connected.
         """
        try:
            # Создаем объект VideoCapture с указанным источником.
            self.cap = cv2.VideoCapture(self.source)

            # Проверяем, успешно ли открылось соединение.
            # isOpened() возвращает True, если VideoCapture успешно инициализирован
            # и может читать кадры (например, камера доступна).
            if self.cap.isOpened():
                logging.info(f"Подключение к камере установлено. Источник: {self.source}")
                self.is_connected = True
            else:
                logging.error(f"Не удалось подключиться к камере. Источник: {self.source}")
                self.is_connected = False
                # Закрываем объект, если соединение не установлено.
                self.cap.release()
                self.cap = None
        except Exception as e:
            # Обработка любых исключений при попытке подключения.
            logging.error(f"Ошибка при подключении к камере {self.source}: {e}")
            self.is_connected = False
            if self.cap:
                self.cap.release()
                self.cap = None

    def read_frame(self):
        """
         Считывает один кадр с камеры.

         Returns:
             tuple: (success, frame), где:
                    - success (bool): True, если кадр успешно захвачен, False в противном случае.
                    - frame (numpy.ndarray or None): Захваченный кадр (массив NumPy) или None, если кадр не получен.
         """
        if not self.is_connected or self.cap is None:
            logging.warning("Попытка захвата кадра, но камера не подключена.")
            return False, None

        try:
            # Метод read() захватывает один кадр.
            # Возвращает кортеж (ret, frame).
            # ret: boolean, True если кадр успешно захвачен.
            # frame: numpy array, сам кадр изображения.
            ret, frame = self.cap.read()

            if ret:
                # Кадр успешно захвачен.
                # logging.debug("Кадр успешно захвачен.")
                return True, frame
            else:
                # Кадр не захвачен (например, камера отключилась).
                logging.warning("Не удалось захватить кадр с камеры.")
                # Попробуем обновить состояние подключения.
                self.is_connected = self.cap.isOpened()
                return False, None

        except Exception as e:
            # Обработка исключений при захвате кадра.
            logging.error(f"Ошибка при захвате кадра с камеры {self.source}: {e}")
            return False, None

    def release(self):
        """
         Освобождает ресурсы, связанные с видеозахватом.

         Этот метод должен быть вызван, когда работа с камерой завершена,
         чтобы корректно закрыть соединение и освободить камеру для других приложений.
         """
        if self.cap:
            logging.info(f"Освобождение ресурсов камеры. Источник: {self.source}")
            self.cap.release()
            self.cap = None
            self.is_connected = False
        else:
            logging.debug("Попытка освободить ресурсы камеры, но объект VideoCapture не инициализирован.")

    def __del__(self):
        """
         Деструктор класса. Вызывается при удалении объекта.
         Гарантирует, что ресурсы будут освобождены, даже если пользователь забудет вызвать release().
         """
        self.release()

# --- Функции модуля (альтернативный подход, если класс не нужен) ---

# def open_camera(source):
#     """
#      Открывает соединение с камерой.
#
#      Args:
#          source: Источник видео (int для локальной камеры, str для IP-камеры).
#
#      Returns:
#          cv2.VideoCapture or None: Объект VideoCapture или None в случае ошибки.
#     """
#     try:
#         cap = cv2.VideoCapture(source)
#         if cap.isOpened():
#             logging.info(f"Подключение к камере установлено. Источник: {source}")
#             return cap
#         else:
#             logging.error(f"Не удалось подключиться к камере. Источник: {source}")
#             cap.release()
#             return None
#     except Exception as e:
#         logging.error(f"Ошибка при подключении к камере {source}: {e}")
#         return None
#
# def read_frame_from_cap(cap):
#     """
#      Считывает один кадр с указанного объекта VideoCapture.
#
#      Args:
#          cap (cv2.VideoCapture): Объект видеозахвата.
#
#      Returns:
#          tuple: (success, frame), как в методе класса.
#     """
#     if cap and cap.isOpened():
#         ret, frame = cap.read()
#         if ret:
#             return True, frame
#         else:
#             logging.warning("Не удалось захватить кадр.")
#             return False, None
#     else:
#         logging.warning("Объект VideoCapture не инициализирован или закрыт.")
#         return False, None
#
# def close_camera(cap):
#     """
#      Закрывает соединение с камерой.
#
#      Args:
#          cap (cv2.VideoCapture): Объект видеозахвата.
#     """
#     if cap:
#         cap.release()
#         logging.info("Соединение с камерой закрыто.")

# --- Модуль как исполняемый скрипт (опционально, для тестирования) ---
# if __name__ == "__main__":
#     # Пример использования класса для тестирования.
#     cam = CameraCapture(0) # Попытка подключиться к локальной камере
#     if cam.is_connected:
#         success, frame = cam.read_frame()
#         if success:
#             print("Кадр успешно захвачен.")
#             # cv2.imshow("Test Feed", frame) # Для визуальной проверки
#             # cv2.waitKey(1) # Краткая задержка
#         else:
#             print("Не удалось захватить кадр.")
#     else:
#         print("Не удалось подключиться к камере.")
#     cam.release() # Важно освободить ресурсы