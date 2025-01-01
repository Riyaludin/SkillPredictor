import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Kode Anda untuk mempersiapkan data dan model tetap sama 
# Misalnya, pastikan Anda sudah mengimpor data dan melatih model RandomForest di bawah ini

# Definisikan LabelEncoder
roles = ["Explaner", "Goldlaner", "Jungler", "Midlaner", "Roamer"]
le_role = LabelEncoder()
le_role.fit(roles)

# Sama dengan rank
rank = ["warrior", "elite", "master", "grandmaster", "epic", "legend", "mythic"]
le_rank = LabelEncoder()
le_rank.fit(rank)

# Category skill
skill_categories = ['pro', 'lumayan', 'dark system']
le_skill = LabelEncoder()
le_skill.fit(skill_categories)  

# Misalnya ini adalah model RandomForest yang sudah terlatih
model = RandomForestClassifier(n_estimators=100, random_state=42)
# Anda harus memiliki model yang sudah dilatih sebelumnya di sini dengan df, X_train, y_train

# Menggunakan streamlit untuk antarmuka pengguna
st.title("Prediksi Kategori Skill Mobile Legends")

# Input Data
role_input = st.selectbox("Pilih Role", roles)  # Dropdown untuk memilih Role
win_rate_input = st.number_input("Masukkan Winrate (%)", min_value=0.0, max_value=100.0, value=00.0, step=0.1)
rank_input = st.selectbox("Pilih Rank", rank)  # Dropdown untuk memilih Rank
total_matches_input = st.number_input("Masukkan Total Matches", min_value=0, value=0)

# Kode untuk Prediksi
if st.button("Prediksi Kategori Skill"):
    # Encoding inputan yang diterima pengguna
    encoded_role = le_role.transform([role_input])[0]
    encoded_rank = le_rank.transform([rank_input])[0]
    
    # Menyiapkan input untuk prediksi (menggabungkan data)
    input_data = pd.DataFrame([[encoded_role, win_rate_input, encoded_rank, total_matches_input]],
    columns=["Role", "Winrate", "HighestRank", "TotalMatch"])
    
    # Lakukan prediksi
    prediction = model.predict(input_data)
    predicted_category = le_skill.inverse_transform(prediction)
    
    # Menampilkan hasil prediksi
    st.write(f"Prediksi kategori skill Anda adalah: {predicted_category[0]}")

