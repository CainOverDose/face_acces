import cv2

# URL вашей камеры Dahua
rtsp_url = "rtsp://admin:pioneer5800@192.168.87.73:554/cam/realmonitor?channel=1&subtype=0"

# Попытка подключения
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("❌ Не удалось подключиться к камере")
    # Пробуем альтернативный метод
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        print("❌ Не удалось подключиться даже с CAP_FFMPEG")
        print("Проверьте:")
        print("1. Правильность URL")
        print("2. Доступность камеры из терминала: ffplay '" + rtsp_url + "'")
        print("3. Настройки сети VirtualBox (сетевой мост)")
        exit()

print("✅ Подключение к камере успешно!")
print(f"Разрешение: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

# Чтение одного кадра для проверки
ret, frame = cap.read()
if ret:
    print("✅ Кадр успешно получен")
    cv2.imwrite("test_frame.jpg", frame)
    print("📸 Кадр сохранен в test_frame.jpg")
else:
    print("❌ Не удалось получить кадр")

# Очистка
cap.release()