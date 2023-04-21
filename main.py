import string
from pyspark import SparkContext, SparkConf

# Функция для загрузки стоп-слов из файла
def load_stop_words(file_path):
    with open(file_path, "r") as file:
        stop_words = [line.strip() for line in file.readlines()]
    return set(stop_words)

# Функция для обработки строки (удаление знаков препинания и приведение к нижнему регистру)
def process_line(line):
    return line.translate(str.maketrans("", "", string.punctuation)).lower().split()

# Функция для фильтрации стоп-слов
def filter_stop_words(stop_words):
    return lambda word: word not in stop_words

# Функция для вычисления статистики длины слов
def word_length_stats(words):
    word_lengths = words.map(lambda word: len(word))
    total_words = word_lengths.count()
    total_word_length = word_lengths.reduce(lambda x, y: x + y)
    
    min_length = word_lengths.min()
    max_length = word_lengths.max()
    avg_length = total_word_length / total_words

    print(f"Min word length: {min_length}")
    print(f"Max word length: {max_length}")
    print(f"Average word length: {avg_length}")

# Функция для вычисления статистики длины предложений
def sentence_length_stats(text_file):
    sentences = text_file.map(lambda line: len(process_line(line)))
    total_sentences = sentences.count()
    total_sentence_length = sentences.reduce(lambda x, y: x + y)
    
    min_length = sentences.min()
    max_length = sentences.max()
    avg_length = total_sentence_length / total_sentences

    print(f"Min sentence length: {min_length}")
    print(f"Max sentence length: {max_length}")
    print(f"Average sentence length: {avg_length}")

# Основная функция для анализа текста
def word_count_top10(file_path, stop_words_file):
    # Загрузка стоп-слов
    stop_words = load_stop_words(stop_words_file)

    # Настройка Spark
    conf = SparkConf().setAppName("WordCount").setMaster("local")
    sc = SparkContext(conf=conf)

    # Чтение текстового файла
    text_file = sc.textFile(file_path)

    # Преобразование строк в слова
    words = text_file.flatMap(process_line)

    # Фильтрация стоп-слов
    filtered_words = words.filter(filter_stop_words(stop_words))

    print("\nWord length statistics:")
    word_length_stats(filtered_words)

    print("\nSentence length statistics:")
    sentence_length_stats(text_file)

    # Подсчет частоты каждого слова
    word_counts = filtered_words.countByValue()

    # Сортировка элементов словаря по значению count в порядке убывания
    sorted_word_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)

    print("\nTop 10 words:")
    for word, count in sorted_word_counts[:10]:
        print(f"{word}: {count}")

    # Остановка Spark контекста
    sc.stop()

# Запуск программы
if __name__ == "__main__":
    file_path = "example.txt"
    stop_words_file = "stop_words.txt"
    word_count_top10(file_path, stop_words_file)
