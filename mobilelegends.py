import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st


# Mengakses dataset dari GitHub
# Load dataset dari URL
url = "https://raw.githubusercontent.com/Riyaludin/SkillPredictor/refs/heads/main/mobile_legend.csv"
try:
    df = pd.read_csv(url)
except Exception as e:
    st.error(f"Failed to load dataset. Error: {e}")

# Encode fitur kategori
le_role = LabelEncoder()
le_rank = LabelEncoder()
le_skill = LabelEncoder()

# Latih LabelEncoder
df['Role'] = le_role.fit_transform(df['Role'])
df['HighestRank'] = le_rank.fit_transform(df['HighestRank'])
df['Category'] = le_skill.fit_transform(df['Category'])

# Menyiapkan fitur (X) dan target (y)
X = df[['Role', 'Winrate', 'HighestRank', 'TotalMatch']]
y = df['Category']

# Memisahkan data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Melatih model Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluasi model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Tampilan evaluasi
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Integrasi pada Streamlit
# Streamlit UI
st.title("Mobile Legends Skill Predictor")
st.write("Masukkan data pengguna untuk memprediksi kategori skill.")

# Input data dari pengguna
role = st.selectbox("Pilih Role", le_role.classes_)
win_rate = st.number_input("Masukan Winrate (dalam %)",min_value=1, step=1)
highest_rank = st.selectbox("Pilih Rank Tertinggi", le_rank.classes_)
total_matches = st.number_input("Masukkan Total Match", min_value=1, step=1)

# Encode input pengguna
role_encoded = le_role.transform([role])[0]
highest_rank_encoded = le_rank.transform([highest_rank])[0]
user_input = [[role_encoded, win_rate, highest_rank_encoded, total_matches]]

# Prediksi skill berdasarkan input pengguna
if st.button("Prediksi"):
    predicted_skill = model.predict(user_input)
    skill_category = le_skill.inverse_transform(predicted_skill)[0]
    st.write(f"Kategori skill berdasarkan data Anda: **{skill_category}**")
