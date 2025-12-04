import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

df = pd.read_csv('Pokemon.csv')

features = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']
target = 'Legendary'

X = df[features]
y = df[target].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

names = {
    0: "COMUM",
    1: "LENDÁRIO"
}

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('names.pkl', 'wb') as f:
    pickle.dump(names, f)

print("Arquivos 'model.pkl' e 'names.pkl' gerados com sucesso!")