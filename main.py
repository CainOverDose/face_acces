# main.py

# Импорт модулей нашего проекта
from utils.camera import CameraCapture
from detectors.yolo_face import detect_faces
from recognizers.deepface_verify import extract_face_embedding
from db.face_database import get_user_embedding, initialize_database # initialize_database для проверки существования БД
from utils.visualizer import draw_face_box_and_label

# Импорт стандартной библиотеки logging.
import logging

# Импорт numpy для вычислений
import numpy as np

# Импорт стандартной библиотеки для чтения конфигурации.
# Пока что захардкодим значения, но в реальности они будут читаться из config.yaml.
# Создадим "фиктивный" объект config для симуляции.
class Config:
    class Camera:
        source = "rtsp://admin:pioneer5800@192.168.87.73:554/cam/realmonitor?channel=1&subtype=0"  # или "rtsp://..." для IP-камеры
        fps = 10
    class Detection:
        model_path = 'models/yolov9m-face-lindevs.pt'
    class Recognition:
        model_name = "ArcFace"
        database_path = "face_data.db"
        # Порог для косинусного расстояния (1 - косинусное схожение)
        # Значения: 0.0 (идентичные) - 2.0 (противоположные)
        # Обычно работает в диапазоне 0.3 - 0.6
        verification_threshold = 0.6 # Пример порога для косинусного расстояния

config = Config()

# --- Функции скрипта ---

def find_best_match(new_embedding, db_path: str, threshold: float):
    """
     Сравнивает новый вектор с векторами из базы данных и находит лучшее совпадение
     с использованием косинусного расстояния.

     Args:
         new_embedding (list): Вектор лица, который нужно сравнить.
         db_path (str): Путь к файлу базы данных.
         threshold (float): Порог косинусного расстояния. Если наименьшее расстояние < threshold,
                            считается, что лицо известно (совпадение найдено).
                            Косинусное расстояние = 1 - косинусное схожение.
                            Значения: 0.0 - идентичные векторы, 2.0 - противоположно направленные.

     Returns:
         tuple: (user_id: str or None, best_distance: float or None)
                user_id: ID пользователя, если совпадение найдено и находится в пределах порога.
                best_distance: Наименьшее косинусное расстояние до найденного совпадения.
                Если совпадение не найдено (или БД пуста), возвращает (None, None).
     """
    import sqlite3
    import json

    best_match_id = None
    best_distance = float('inf') # Минимальное косинусное расстояние

    # Преобразуем новый вектор в numpy array для удобства вычислений
    new_embedding_np = np.array(new_embedding)

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        select_all_sql = "SELECT user_id, embedding_json FROM users"
        cursor.execute(select_all_sql)
        rows = cursor.fetchall()

        logging.info(f"[DEBUG] Найдено {len(rows)} записей в БД для сравнения.")

        for row in rows:
            user_id, embedding_str = row
            db_embedding = json.loads(embedding_str)
            # Преобразуем вектор из БД в numpy array
            db_embedding_np = np.array(db_embedding)

            # --- Вычисляем косинусное сходство ---
            # cosine_similarity = (A . B) / (||A|| * ||B||)
            dot_product = np.dot(new_embedding_np, db_embedding_np)
            norm_new = np.linalg.norm(new_embedding_np)
            norm_db = np.linalg.norm(db_embedding_np)

            # Проверим, чтобы нормы не были нулевыми (вдруг ошибка в данных)
            if norm_new == 0 or norm_db == 0:
                logging.warning(f"Обнаружена векторная норма 0 для пользователя {user_id}. Пропуск.")
                continue

            cosine_similarity = dot_product / (norm_new * norm_db)

            # --- Преобразуем в косинусное *расстояние* ---
            # Косинусное расстояние = 1 - косинусное схожение
            cosine_distance = 1 - cosine_similarity

            logging.info(f"[DEBUG] Сравнение с {user_id}: кос. расстояние = {cosine_distance:.4f}")

            # Проверяем, является ли это лучшим совпадением на данный момент.
            if cosine_distance < best_distance:
                best_distance = cosine_distance
                best_match_id = user_id

    except sqlite3.Error as e:
        logging.error(f"Ошибка при поиске совпадений в базе данных {db_path}: {e}")
        return None, None
    finally:
        if conn:
            conn.close()

    # Проверяем, находится ли лучшее совпадение в пределах порога.
    if best_match_id is not None and best_distance < threshold:
        logging.info(f"[DEBUG] Лучшее совпадение: {best_match_id}, кос. расстояние: {best_distance:.4f}, порог: {threshold} -> Совпадение найдено.")
        return best_match_id, best_distance
    else:
        if best_match_id is not None:
             logging.info(f"[DEBUG] Лучшее совпадение: {best_match_id}, кос. расстояние: {best_distance:.4f}, порог: {threshold} -> Не прошло порог.")
        else:
             logging.info(f"[DEBUG] Совпадений не найдено в БД.")
        return None, None


def main_loop():
    """
     Основной цикл программы для распознавания лиц.

     Выполняет следующие шаги в цикле:
     1. Считывает кадр с камеры.
     2. Обнаруживает лица на кадре.
     3. Для каждого лица извлекает вектор.
     4. Сравнивает вектор с базой данных.
     5. Визуализирует результаты (рамки, подписи) и выводит в консоль.
     """
    print("--- Запуск основного цикла распознавания ---")

    # Инициализируем базу данных на случай, если она не существует.
    initialize_database(config.Recognition.database_path)

    # Подключаемся к камере.
    camera = CameraCapture(source=config.Camera.source)

    if not camera.is_connected:
        logging.error(f"Не удалось подключиться к камере по источнику {config.Camera.source}.")
        return

    print("Камера подключена. Нажмите 'q' на окне камеры, чтобы выйти.")

    try:
        while True:
            # 1. Считываем кадр с камеры.
            success, frame = camera.read_frame()
            if not success:
                logging.warning("Не удалось захватить кадр. Проверьте камеру.")
                # Можно попробовать переподключиться или просто выйти.
                # Пока выйдем.
                break

            # 2. Обнаруживаем лица на кадре.
            faces = detect_faces(frame, model_path=config.Detection.model_path)

            # 3. Обрабатываем каждое обнаруженное лицо.
            for face_info in faces:
                box = face_info['box']

                # 4. Обрезаем изображение лица.
                x1, y1, x2, y2 = box
                face_image = frame[y1:y2, x1:x2]

                # 5. Извлекаем вектор (embedding) из обрезанного изображения.
                embedding = extract_face_embedding(
                    face_image,
                    model_name=config.Recognition.model_name,
                    enforce_detection=False
                )

                if embedding is not None:
                    # 6. Сравниваем вектор с базой данных.
                    matched_user_id, match_distance = find_best_match( # Переименовал переменную
                        embedding,
                        config.Recognition.database_path,
                        config.Recognition.verification_threshold
                    )

                    # 7. Визуализация и вывод в консоль.
                    if matched_user_id is not None:
                        label = f"{matched_user_id}"
                        # Выводим косинусное расстояние и порог для ясности
                        print(f"[INFO] Найдено лицо: {matched_user_id}, кос. расст.: {match_distance:.4f}, порог: {config.Recognition.verification_threshold}")
                        draw_face_box_and_label(frame, box, label, face_info['confidence']) # Рисуем с уверенностью YOLO
                    else:
                        # Проверим, нашлась ли *какая-то* близкая запись, но не прошедшая порог
                        # find_best_match в текущей версии возвращает (None, None), если не прошло порог
                        # Для простоты, просто покажем "Неизвестный", если (None, None)
                        label = "Неизвестный"
                        print(f"[INFO] Найдено лицо: {label} (нет совпадений в БД или не прошло порог)") # Общий вывод
                        draw_face_box_and_label(frame, box, label, face_info['confidence'])
                else:
                    # Если вектор не удалось извлечь, просто рисуем рамку как "неизвестное".
                    label = "Неизвестный (ошибка вектора)"
                    logging.warning(f"Не удалось извлечь вектор для лица на bbox {box}")
                    draw_face_box_and_label(frame, box, label, face_info['confidence'])

            # Показываем кадр с наложенными элементами.
            # cv2.imshow("Face Recognition Feed", frame)

            # 8. Обработка клавиш для выхода.
            # key = cv2.waitKey(1) & 0xFF
            # if key == ord('q'):
            #     print("Выход по команде пользователя.")
            #     break

    except KeyboardInterrupt:
        print("\nВыход по команде Ctrl+C.")
    finally:
        # 9. Освобождаем ресурсы камеры.
        camera.release()
        # cv2.destroyAllWindows()


def main():
    """
     Точка входа в скрипт main.py.
     """
    # Настройка логирования.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Запуск основного цикла.
    main_loop()


# --- Точка входа ---
if __name__ == "__main__":
    # Импорт cv2 необходим здесь, так как он используется для отображения.
    import cv2
    main()