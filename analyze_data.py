import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных из CSV файла
data = pd.read_csv('data/uploads/sample_upload.csv')

# Подсчет частоты реакций
reaction_counts = data['reaction'].value_counts()

# Построение графика
plt.figure(figsize=(10, 6))
reaction_counts.plot(kind='bar', color='skyblue')
plt.title('Частота реакций на препараты')
plt.xlabel('Реакция')
plt.ylabel('Количество')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show() 