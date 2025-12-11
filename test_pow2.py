import os
# Добавляем разрешение на дублирование библиотек OMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
from ultralytics import YOLO
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- НАСТРОЙКИ КАМЕРЫ И МОДЕЛИ ---

# RTSP URL для основного потока Dahua (2560x1440)
# Замените данные на ваши реальные
RTSP_URL = "rtsp://admin:pioneer5800@192.168.87.73:554/cam/realmonitor?channel=1&subtype=0"

# Путь к вашей модели
MODEL_PATH = 'models/yolo11m-pose.pt' 

# Используем аппаратное ускорение CUDA
# Ultralytics автоматически использует onnxruntime-gpu или торч с CUDA, если они установлены и доступны.
DEVICE = 0 # 0 означает использование GPU (CUDA)

# --- НАСТРОЙКИ ОКНА ПРОСМОТРА ---

WINDOW_NAME = "YOLO Face Detection on Dahua Cam"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1280, 720) # Размер окна на экране
cv2.moveWindow(WINDOW_NAME, 0, 0)       # Перемещение в угол первого монитора


# --- ЗАПУСК ПОТОКА И ДЕТЕКЦИИ ---

try:
    # 1. Загрузка модели с указанием устройства (GPU)
    model = YOLO(MODEL_PATH)
    logging.info("Модель YOLO успешно загружена.")
    
    # 2. Запуск обработки с использованием аппаратного ускорения
    # Ultralytics сам обрабатывает декодирование через PyAV/FFmpeg с CUDA, если это возможно.
    results_generator = model.predict(
        source=RTSP_URL,
        device=DEVICE,    # Указываем использовать GPU
        stream=True,      # Потоковая обработка
        show=False,       # Ручное управление окном
        conf=0.5,         # Порог уверенности
        save=False
    )
    
    logging.info(f"Подключение к RTSP потоку с аппаратным ускорением GPU: {DEVICE}")

    # 3. Вручную обрабатываем кадры
    for result in results_generator:
        frame_with_detections = result.plot()
        cv2.imshow(WINDOW_NAME, frame_with_detections)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    logging.error(f"Произошла ошибка во время выполнения: {e}")
    logging.info("Убедитесь, что IP-адрес, логин и пароль указаны верно, и камера доступна по сети.")
    logging.info("Также проверьте установку onnxruntime-gpu или PyTorch с поддержкой CUDA.")

finally:
    # Закрываем все окна после завершения
    cv2.destroyAllWindows()
