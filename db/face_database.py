# db/face_database.py

# Импорт стандартной библиотеки sqlite3 для работы с базой данных SQLite.
import sqlite3

# Импорт стандартной библиотеки logging.
# Она используется для вывода сообщений о состоянии программы (отладка, ошибки).
import logging

# Импорт стандартной библиотеки json для сериализации/десериализации вектора.
# Вектор (embedding) представляет собой список чисел (list of floats).
# SQLite не имеет встроенного типа для хранения списков.
# Поэтому вектор нужно преобразовать в строку (например, JSON) для сохранения
# и обратно в список при чтении.
import json

# --- Константы ---

# SQL-запрос для создания таблицы пользователей.
# user_id: Уникальный идентификатор пользователя (например, строка "user_001").
# name: Имя пользователя (для удобства).
# embedding_json: Вектор лица, сериализованный в формате JSON и сохраненный как TEXT.
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    embedding_json TEXT NOT NULL
);
"""

# --- Функции модуля ---

def initialize_database(db_path: str):
    """
     Инициализирует базу данных SQLite, создавая таблицу users, если она не существует.

     Эта функция должна быть вызвана один раз при запуске приложения
     или когда необходимо убедиться, что структура базы данных актуальна.

     Args:
         db_path (str): Путь к файлу базы данных SQLite (например, "face_data.db").

     Raises:
         sqlite3.Error: Для любых ошибок, связанных с SQLite.
     """
    try:
        # Устанавливаем соединение с базой данных.
        # Если файл db_path не существует, он будет создан автоматически.
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Выполняем SQL-запрос для создания таблицы.
        # "IF NOT EXISTS" означает, что таблица будет создана только если её ещё нет.
        cursor.execute(CREATE_TABLE_SQL)
        logging.info(f"База данных инициализирована. Таблица 'users' готова. Путь: {db_path}")

        # Сохраняем изменения в базе данных.
        conn.commit()
    except sqlite3.Error as e:
        # Обработка ошибок SQLite.
        error_msg = f"Ошибка при инициализации базы данных {db_path}: {e}"
        logging.error(error_msg)
        # Важно выбросить исключение, чтобы вызывающий код узнал о проблеме.
        raise sqlite3.Error(error_msg)
    finally:
        # Всегда закрываем соединение, чтобы освободить ресурсы.
        if conn:
            conn.close()


def add_user(db_path: str, user_id: str, name: str, embedding: list):
    """
     Добавляет нового пользователя (ID, имя, вектор лица) в базу данных.

     Args:
         db_path (str): Путь к файлу базы данных SQLite.
         user_id (str): Уникальный идентификатор пользователя.
         name (str): Имя пользователя.
         embedding (list): Вектор (embedding) лица, представляющий собой список чисел (float).

     Raises:
         ValueError: Если embedding не является списком или пуст.
         sqlite3.IntegrityError: Если user_id уже существует (нарушение PRIMARY KEY).
         sqlite3.Error: Для любых других ошибок, связанных с SQLite.
     """
    # Проверка, является ли embedding списком и не пуст ли он.
    # Это базовая проверка на ограниченный случай.
    if not isinstance(embedding, list) or len(embedding) == 0:
        error_msg = f"Вектор лица (embedding) должен быть непустым списком. Получено: {type(embedding)}, длина: {len(embedding) if isinstance(embedding, list) else 'N/A'}"
        logging.error(error_msg)
        raise ValueError(error_msg)

    try:
        # Устанавливаем соединение с базой данных.
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Сериализуем вектор в формат JSON для хранения в TEXT поле.
        embedding_str = json.dumps(embedding)

        # Подготовленный SQL-запрос INSERT.
        # Использование ? позволяет избежать SQL-инъекций.
        insert_sql = "INSERT INTO users (user_id, name, embedding_json) VALUES (?, ?, ?)"

        # Выполняем запрос, передавая параметры.
        cursor.execute(insert_sql, (user_id, name, embedding_str))
        logging.info(f"Пользователь '{name}' с ID '{user_id}' успешно добавлен в базу данных.")

        # Сохраняем изменения в базе данных.
        conn.commit()
    except sqlite3.IntegrityError as e:
        # Обработка ошибки, если user_id уже существует.
        error_msg = f"Пользователь с ID '{user_id}' уже существует в базе данных. Ошибка: {e}"
        logging.error(error_msg)
        raise sqlite3.IntegrityError(error_msg)
    except sqlite3.Error as e:
        # Обработка любых других ошибок SQLite.
        error_msg = f"Ошибка при добавлении пользователя '{name}' в базу данных {db_path}: {e}"
        logging.error(error_msg)
        raise sqlite3.Error(error_msg)
    finally:
        # Всегда закрываем соединение.
        if conn:
            conn.close()


def get_user_embedding(db_path: str, user_id: str) -> list or None:
    """
     Извлекает вектор (embedding) лица пользователя по его ID из базы данных.

     Args:
         db_path (str): Путь к файлу базы данных SQLite.
         user_id (str): Уникальный идентификатор пользователя.

     Returns:
         list or None: Вектор (embedding) лица в виде списка чисел (float),
                       или None, если пользователь с таким ID не найден.

     Raises:
         sqlite3.Error: Для любых ошибок, связанных с SQLite.
     """
    try:
        # Устанавливаем соединение с базой данных.
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Подготовленный SQL-запрос SELECT.
        select_sql = "SELECT embedding_json FROM users WHERE user_id = ?"

        # Выполняем запрос, передавая user_id.
        cursor.execute(select_sql, (user_id,))
        row = cursor.fetchone() # Получаем одну строку результата.

        if row:
            # Если строка найдена, первый столбец (embedding_json) - это наша сериализованная строка.
            embedding_str = row[0]
            # Десериализуем строку JSON обратно в список.
            embedding = json.loads(embedding_str)
            logging.debug(f"Вектор для пользователя '{user_id}' успешно получен из базы данных.")
            return embedding
        else:
            # Если строка не найдена, возвращаем None.
            logging.info(f"Пользователь с ID '{user_id}' не найден в базе данных.")
            return None

    except sqlite3.Error as e:
        # Обработка ошибок SQLite.
        error_msg = f"Ошибка при получении вектора для пользователя '{user_id}' из базы данных {db_path}: {e}"
        logging.error(error_msg)
        raise sqlite3.Error(error_msg)
    finally:
        # Всегда закрываем соединение.
        if conn:
            conn.close()

# --- Модуль как исполняемый скрипт (опционально, для тестирования) ---
# if __name__ == "__main__":
#     # Пример использования модуля для тестирования.
#     db_file = "test_face_data.db"
#     initialize_database(db_file)
#
#     # Пример вектора (в реальности это будет длинный список float)
#     sample_embedding = [0.1, 0.2, 0.3]
#     add_user(db_file, "test_user_1", "Test User", sample_embedding)
#
#     retrieved_embedding = get_user_embedding(db_file, "test_user_1")
#     print(retrieved_embedding)