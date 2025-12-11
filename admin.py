# admin.py

# Импорт модулей нашего проекта
from utils.camera import CameraCapture
from detectors.yolo_face import detect_faces
from recognizers.deepface_verify import extract_face_embedding
from db.face_database import initialize_database, add_user

# Импорт стандартной библиотеки logging.
import logging

# Импорт стандартной библиотеки для чтения конфигурации и файловых путей.
import os
# DEFAULT_DB_PATH = "face_data.db" # Пусть будет здесь, если используется как константа
DEFAULT_DB_PATH = os.getenv("FACE_DB_PATH", "face_data.db") # Пример с env переменной, если нужно

# --- Функции скрипта ---

def register_new_user():
    """
     Основная функция скрипта admin.py для регистрации нового пользователя.

     Выполняет следующие шаги:
     1. Запрашивает имя нового пользователя.
     2. Подключается к камере.
     3. Ждет, пока на кадре появится одно лицо.
     4. Обрезает изображение лица.
     5. Извлекает вектор (embedding) с помощью DeepFace.
     6. Сохраняет имя и вектор в базу данных.
     """
    print("--- Регистрация нового пользователя ---")

    # 1. Запрашиваем имя нового пользователя.
    name = input("Введите имя нового пользователя: ").strip()
    if not name:
        print("Имя не может быть пустым.")
        return

    # Генерируем уникальный ID. Простой способ - использовать имя или его хэш.
    # В реальных системах ID часто генерируется UUID или берётся из БД.
    # Для MVP используем имя, заменив пробелы на подчеркивания.
    user_id = name.replace(" ", "_").lower()

    # 2. Подключаемся к камере. Используем локальную камеру (0) или читаем из config.yaml.
    # Для MVP захардкодим.
    camera_source = "rtsp://admin:pioneer5800@192.168.87.73:554/cam/realmonitor?channel=1&subtype=0" # или читать из config.yaml
    camera = CameraCapture(source=camera_source)

    if not camera.is_connected:
        print(f"Ошибка: Не удалось подключиться к камере по источнику {camera_source}.")
        camera.release()
        return

    print(f"Камера подключена. Пожалуйста, поставьте лицо перед камерой.")
    print("Нажмите Enter в терминале, когда будете готовы сделать снимок.")
    print("Чтобы выйти без регистрации, нажмите Ctrl+C в другом терминале или закройте этот скрипт.")
    # Примечание: Классический способ получения нажатия клавиши без GUI - input(). Это блокирует выполнение.

    # --- ИМПОРТИРУЕМ CV2 и SQLITE3 ЗДЕСЬ ---
    import cv2
    import sqlite3

    embedding = None
    face_image_for_embedding = None

    try:
        # Ждём ввода от пользователя
        input("Нажмите Enter, чтобы сделать снимок...")

        print("Сделан снимок для регистрации...")

        # Считываем кадр с камеры в момент нажатия Enter.
        success, frame = camera.read_frame()
        if not success:
            print("Не удалось захватить кадр с камеры. Проверьте подключение.")
            return # Прерываем функцию, если кадр не получен

        # На этом кадре нужно найти лицо и извлечь вектор.
        # Обнаруживаем лица на кадре.
        # ПРЕДПОЛАГАЕМ, ЧТО ПУТЬ К МОДЕЛИ УКАЗАН ВЕРНО, НАПРИМЕР, В models/
        # ИСПРАВЬТЕ ПУТЬ К МОДЕЛИ, ЕСЛИ НЕОБХОДИМО
        model_path = 'models/yolov9m-face-lindevs.pt' # Измените путь, если yolov8n-face.pt не в корне или models/
        if not os.path.exists(model_path):
             # Попробуем найти в текущей директории или models
             model_path_local = "yolov8n-face.pt"
             model_path_models = os.path.join("models", "yolov8n-face.pt")
             if os.path.exists(model_path_models):
                 model_path = model_path_models
             elif os.path.exists(model_path_local):
                 model_path = model_path_local
             else:
                 print(f"Ошибка: Файл модели YOLO не найден по пути: {model_path_local}, {model_path_models}")
                 return # Прерываем, если модель не найдена

        faces = detect_faces(frame, model_path=model_path) # Используем найденный/указанный путь

        if len(faces) == 1:
            # 5. Обрезаем изображение лица.
            box = faces[0]['box']
            x1, y1, x2, y2 = box
            face_image_for_embedding = frame[y1:y2, x1:x2]

            # 6. Извлекаем вектор с помощью DeepFace.
            try:
                embedding = extract_face_embedding(
                    face_image_for_embedding,
                    model_name="ArcFace",
                    enforce_detection=False
                )
                if embedding is not None:
                    print(f"Вектор лица успешно извлечен. Длина: {len(embedding)}")
                else:
                    print("Не удалось извлечь вектор лица. Проверьте качество изображения.")
            except Exception as e:
                print(f"Ошибка при извлечении вектора: {e}")
                embedding = None
        elif len(faces) == 0:
            print("На кадре не обнаружено лиц. Пожалуйста, поставьте лицо перед камерой и повторите регистрацию.")
        else:
            print(f"На кадре обнаружено {len(faces)} лиц. Пожалуйста, останьтесь один на кадре и повторите регистрацию.")

    except KeyboardInterrupt:
        print("\nРегистрация отменена пользователем (Ctrl+C).")
        return # Прерываем функцию при прерывании
    finally:
        # 7. Освобождаем ресурсы камеры.
        camera.release()
        # cv2.destroyAllWindows() # Не нужно, так как окно не создавалось


    # 8. Сохраняем вектор в базу данных, если он был успешно извлечен.
    if embedding is not None:
        try:
            # Инициализируем БД (создаст, если не существует).
            initialize_database(DEFAULT_DB_PATH)

            # Добавляем пользователя.
            add_user(DEFAULT_DB_PATH, user_id, name, embedding)
            print(f"Пользователь '{name}' (ID: {user_id}) успешно зарегистрирован в системе.")
        except sqlite3.IntegrityError:
            print(f"Ошибка: Пользователь с ID '{user_id}' уже существует в базе данных.")
        except Exception as e:
            print(f"Ошибка при сохранении пользователя в базу данных: {e}")
    else:
        print("Регистрация не завершена: вектор лица не был получен.")


def main():
    """
     Точка входа в скрипт admin.py.
     """
    # Настройка логирования (опционально, можно настроить в отдельном файле или через config.yaml).
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Запуск процесса регистрации.
    register_new_user()


# --- Точка входа ---
if __name__ == "__main__":
    main()