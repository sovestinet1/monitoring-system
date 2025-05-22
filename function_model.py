from transformers import pipeline

# Создаём pipeline на основе обученной модели
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

def detect_adrs(text):
    print("🔍 Анализ текста:", text)
    results = ner_pipeline(text)
    for r in results:
        entity = r['entity_group']
        word = r['word']
        score = r['score']
        print(f"📌 Найдено: '{word}' → {entity} (уверенность: {score:.2f})")
    return results
