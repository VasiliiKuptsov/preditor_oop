import pandas as pd
import numpy as np

np.random.seed(42)  # Для воспроизводимости

# Параметры генерации
n_rows = 10000
companies = ['ElectroTech', 'GadgetWorld', 'HomeSystems', 'AudioMasters', 'OfficePlus']
products = {
    'ElectroTech': ['Smartphone X%d' % i for i in range(200, 215)],
    'GadgetWorld': ['Wireless Earbuds %s' % v for v in ['Pro', 'Lite', 'Air', 'Max']],
    'HomeSystems': ['Smart TV %d"' % s for s in [32, 43, 50, 55, 65]],
    'AudioMasters': ['Headphones %s' % t for t in ['Elite', 'Pro', 'Studio', 'Travel']],
    'OfficePlus': ['Mouse %s' % c for c in ['Wireless', 'Gaming', 'Ergo', 'Compact']]
}

# Генерация данных
data = []
for _ in range(n_rows):
    company = np.random.choice(companies)
    product = np.random.choice(products[company])

    # Генерация цены в зависимости от категории
    if company == 'ElectroTech':
        price = np.random.normal(800, 150)
    elif company == 'AudioMasters':
        price = np.random.uniform(50, 400)
    else:
        price = np.random.uniform(20, 1000)
    price = max(10, round(abs(price), 2))

    add_cost = round(price * np.random.uniform(0.02, 0.15), 2)
    count = int(max(1, np.random.poisson(100 - price / 20)))

    data.append([price, count, add_cost, company, product])

# Создание DataFrame
df = pd.DataFrame(data, columns=['price', 'count', 'add_cost', 'company', 'product'])

# Сохранение в CSV
df.to_csv('product_prices_10k.csv', index=False)
print("Файл 'product_prices_10k.csv' успешно создан!")