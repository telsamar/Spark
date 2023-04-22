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

def save_results_to_file(results, file_path):
    with open(file_path, "w") as file:
        for line in results:
            file.write(f"{line}\n")

def word_length_stats(words):
    word_lengths = words.map(lambda word: len(word))
    total_words = word_lengths.count()
    total_word_length = word_lengths.reduce(lambda x, y: x + y)
    
    min_length = word_lengths.min()
    max_length = word_lengths.max()
    avg_length = total_word_length / total_words

    return [
        f"Min word length: {min_length}",
        f"Max word length: {max_length}",
        f"Average word length: {avg_length}"
    ]

def sentence_length_stats(text_file):
    sentences = text_file.map(lambda line: len(process_line(line)))
    total_sentences = sentences.count()
    total_sentence_length = sentences.reduce(lambda x, y: x + y)
    
    min_length = sentences.min()
    max_length = sentences.max()
    avg_length = total_sentence_length / total_sentences

    return [
        f"Min sentence length: {min_length}",
        f"Max sentence length: {max_length}",
        f"Average sentence length: {avg_length}"
    ]

# Основная функция для анализа текста
def word_count_top25(file_path, stop_words_file):
    stop_words = load_stop_words(stop_words_file)

    conf = SparkConf().setAppName("WordCount").setMaster("local")
    sc = SparkContext(conf=conf)

    text_file = sc.textFile(file_path)
    words = text_file.flatMap(process_line)

    filtered_words = words.filter(filter_stop_words(stop_words))

    results = []

    results.append("Word length statistics:")
    word_length_stats_result = word_length_stats(filtered_words)
    results.extend(word_length_stats_result)

    results.append("\nSentence length statistics:")
    sentence_length_stats_result = sentence_length_stats(text_file)
    results.extend(sentence_length_stats_result)

    word_counts = filtered_words.countByValue()
    sorted_word_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)

    top25_words = []
    for word, count in sorted_word_counts[:25]:
        top25_words.append(f"{word}: {count}")

    results.append("\nWords count top25 statistics:")
    results.extend(top25_words)

    save_results_to_file(results, "results/result.txt")

    sc.stop()

# Запуск программы
if __name__ == "__main__":
    file_path = "example.txt"
    stop_words_file = "stop_words.txt"
    word_count_top25(file_path, stop_words_file)
