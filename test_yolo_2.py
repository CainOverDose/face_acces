import cv2
from ultralytics import YOLO

# 1. Загрузите вашу модель (убедитесь, что путь правильный, как мы обсуждали ранее)
model = YOLO('models/yolov9m-face-lindevs.pt') 

# 2. Определите источник видеопотока (0 для веб-камеры)
video_source = "rtsp://admin:pioneer5800@192.168.87.73:554/cam/realmonitor?channel=1&subtype=0"

# --- Добавленные шаги для управления окном ---

# 3. Создаем именованное окно с флагом, который позволяет изменять его размер вручную
WINDOW_NAME = "YOLO Face Detection"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# 4. Перемещаем окно в определенные координаты (опционально)
# Если вы знаете координаты верхнего левого угла нужного монитора:
# Например, x=0, y=0 для первого монитора.
cv2.moveWindow(WINDOW_NAME, 0, 0) 

# 5. Устанавливаем желаемый размер окна
# Например, 1280x720 пикселей.
cv2.resizeWindow(WINDOW_NAME, 1280, 720) 

# 6. Запускаем обработку, но отключаем автоматическое отображение (show=False)
# Мы будем отображать кадры вручную
results_generator = model.predict(source=video_source, stream=True, show=False, conf=0.5, save=False)

# 7. Вручную обрабатываем кадры из генератора
for result in results_generator:
    # result.plot() возвращает кадр с нарисованными детекциями (Numpy array)
    frame_with_detections = result.plot()
    
    # Отображаем кадр в созданном нами окне
    cv2.imshow(WINDOW_NAME, frame_with_detections)
    
    # Проверяем нажатие клавиши 'q' для выхода
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 8. Закрываем все окна после завершения
cv2.destroyAllWindows()
