import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from rapidfuzz import process, fuzz
import re
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

### BƯỚC 1: Đọc và lọc dữ liệu ###
df = pd.read_csv('E:/Python codeptit/Bai_tap_lon/Bai_1/results.csv')
df_filtered = df[df['Min'] > 900].copy()

### BƯỚC 2: Crawl dữ liệu từ web ###
PATH = 'D:/chromedriver/chromedriver-win64/chromedriver-win64/chromedriver.exe'
service = Service(executable_path=PATH)
driver = webdriver.Chrome(service=service)

all_data = []
for page in range(1, 23):
    if page == 1:
        url = 'https://www.footballtransfers.com/us/values/players/most-valuable-soccer-players/playing-in-uk-premier-league'
    else:
        url = f'https://www.footballtransfers.com/us/values/players/most-valuable-soccer-players/playing-in-uk-premier-league/{page}'
    driver.get(url)
    time.sleep(2)

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    table = soup.find('tbody', {'id': 'player-table-body'})

    rows = table.find_all('tr')
    print(f"✅ Trang {page}: {len(rows)} cầu thủ")

    for row in rows:
        cols = row.find_all('td')
        data = []
        for idx, col in enumerate(cols):
            if idx == 2:  # Cột "Player"
                span = col.find('span')
                name = span.get_text(strip=True) if span else col.get_text(strip=True)
                data.append(name)
            else:
                data.append(col.get_text(strip=True))
        if data:
            all_data.append(data)

driver.quit()

### BƯỚC 3: Tạo DataFrame từ dữ liệu crawl ###
df_values = pd.DataFrame(all_data)
df_values.columns = ['Skill', 'Rank', 'Player', 'Age', 'Club', 'MarketValue']

## BƯỚC 4: Làm sạch tên cầu thủ ###
def clean_player_name(name):
    name = str(name)

    # Tên bị lặp (ví dụ: CaicedoCaicedo)
    match = re.match(r'^([A-Za-zÀ-ÿ\.\- ]+?)\1', name)
    if match:
        return match.group(1).strip()

    # Lấy 2 từ đầu tiên
    words = name.split()
    if len(words) >= 2:
        return f"{words[0]} {words[1]}"
    return name

# Áp dụng cho cả dữ liệu từ web và từ file
df_values['CleanedPlayer'] = df_values['Player'].apply(clean_player_name)
# df_values.to_csv('df_values.csv', index = False, encoding= 'utf - 8 - sig')
df_filtered['CleanedPlayer'] = df_filtered['Player'].apply(clean_player_name)

### BƯỚC 5: Fuzzy match tên ###
web_players = df_values['CleanedPlayer'].tolist()
value_dict = dict(zip(df_values['CleanedPlayer'], df_values['MarketValue']))

def get_market_value_fuzzy(player_name):
    match, score, _ = process.extractOne(player_name, web_players, scorer=fuzz.token_sort_ratio)
    if score >=85:  
        return value_dict[match]
    return 'N/A'

df_filtered['MarketValue'] = df_filtered['CleanedPlayer'].apply(get_market_value_fuzzy)


### BƯỚC 6: Xuất kết quả ###
final_df = df_filtered[['Player', 'Min', 'MarketValue']].copy()
final_df.insert(0, 'stt', range(1, len(final_df) + 1))

final_df.to_csv('E:/Python codeptit/Bai_tap_lon/Bai_4/results_ex4.csv', index=False, encoding='utf-8-sig')
print("\n✅ Đã lưu results_ex4.csv thành công!")
print(final_df.head(10))

#---
# Định nghĩa một giá trị random_state cố định để dễ điều chỉnh
RANDOM_STATE = 42

# 1. Đọc dữ liệu
stats_df = pd.read_csv('E:/Python codeptit/Bai_tap_lon/Bai_1/results.csv')  # Dữ liệu thống kê cầu thủ
transfers_df = pd.read_csv('E:/Python codeptit/Bai_tap_lon/Bai_4/results_ex4.csv')  # Dữ liệu giá trị chuyển nhượng

# 2. Kết hợp dữ liệu
data = pd.merge(stats_df, transfers_df[['Player', 'MarketValue']], on='Player', how='inner')

# 3. Lọc cầu thủ có thời gian thi đấu trên 900 phút
data = data[data['Min'] > 900]

# 4. Làm sạch cột MarketValue (chuyển từ chuỗi thành số)
def clean_market_value(value):
    if value == 'N/A' or pd.isna(value):
        return np.nan
    value = value.replace('€', '').strip()
    if 'M' in value:
        return float(value.replace('M', '')) * 1000000
    elif 'K' in value:
        return float(value.replace('K', '')) * 1000
    return float(value)

data['MarketValue'] = data['MarketValue'].apply(clean_market_value)

# 5. Loại bỏ các hàng có MarketValue là NaN
data = data.dropna(subset=['MarketValue'])

# 6. Đổi tên cột để đồng nhất với định dạng trong mã
data.rename(columns={
    'Cmp%': 'Passing_Total_Cmp%',
    'Att Pen': 'Possession_Att Pen',
    'Cmp%.2': 'Passing_Medium_Cmp%',
    'PrgP_x': 'Standard_PrgP',
    'SoT/90': 'Shooting_SoT/90',
    'xG': 'Standard_xG/90',
    'Gls': 'Standard_Gls/90',
    'Cmp%.3': 'Passing_Long_Cmp%'
}, inplace=True)

# 7. Kiểm tra lại tên cột sau khi đổi
print("Danh sách các cột sau khi đổi tên:")
print(data.columns.tolist())

# 8. Làm sạch các cột đặc trưng
# Xử lý cột Age
def clean_age(age):
    if pd.isna(age):
        return np.nan
    age_str = str(age).split('-')[0]
    return float(age_str)

data['Age'] = data['Age'].apply(clean_age)

# Xử lý các cột phần trăm
def clean_percentage(value):
    if pd.isna(value):
        return np.nan
    return float(str(value).replace('%', ''))

percentage_cols = ['Passing_Total_Cmp%', 'Passing_Medium_Cmp%', 'Passing_Long_Cmp%']
for col in percentage_cols:
    data[col] = data[col].apply(clean_percentage)

# Chuyển đổi các cột còn lại thành số
numeric_cols = ['Possession_Att Pen', 'Standard_PrgP', 'Shooting_SoT/90', 'Standard_xG/90', 'Standard_Gls/90']
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# 9. Loại bỏ các hàng có giá trị NaN trong các cột đặc trưng
features = ['Age', 'Passing_Total_Cmp%', 'Possession_Att Pen', 'Passing_Medium_Cmp%', 
            'Standard_PrgP', 'Shooting_SoT/90', 'Standard_xG/90', 'Standard_Gls/90', 
            'Passing_Long_Cmp%']
data = data.dropna(subset=features)

# 10. Kiểm tra dữ liệu sau khi làm sạch
for col in features:
    print(f"\nCột: {col}")
    print(f"Kiểu dữ liệu: {data[col].dtype}")
    print(f"Một số giá trị mẫu: {data[col].head().tolist()}")

# 11. Chọn đặc trưng và mục tiêu
target = 'MarketValue'
X = data[features]
y = data[target]

# 12. Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 13. Khởi tạo các mô hình
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42)
}

# 14. Lưu trữ kết quả đánh giá
mse_scores = {}
mae_scores = {}
r2_scores = {}

# 15. Huấn luyện và đánh giá từng mô hình
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse_scores[name] = mean_squared_error(y_test, y_pred)
    mae_scores[name] = mean_absolute_error(y_test, y_pred)
    r2_scores[name] = r2_score(y_test, y_pred)

# 16. Vẽ biểu đồ so sánh hiệu suất
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Biểu đồ MSE
axes[0].bar(mse_scores.keys(), mse_scores.values(), color='skyblue')
axes[0].set_title('Mean Squared Error (MSE)')
axes[0].tick_params(axis='x', rotation=45)

# Biểu đồ MAE
axes[1].bar(mae_scores.keys(), mae_scores.values(), color='lightgreen')
axes[1].set_title('Mean Absolute Error (MAE)')
axes[1].tick_params(axis='x', rotation=45)

# Biểu đồ R² Score
axes[2].bar(r2_scores.keys(), r2_scores.values(), color='salmon')
axes[2].set_title('R² Score')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# 17. Vẽ biểu đồ Feature Importance cho XGBoost
xgb_model = models['XGBoost']
feature_importance = xgb_model.feature_importances_
feature_names = features

# Tạo DataFrame cho Feature Importance
importance_df = pd.DataFrame({
    'Features': feature_names,
    'Importance Score': feature_importance * 100 
})
importance_df = importance_df.sort_values(by='Importance Score', ascending=True)

# Vẽ biểu đồ
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Features'], importance_df['Importance Score'], color='dodgerblue')
plt.title('Feature Importance (Tuned XGBoost)')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()

# 18. In ra giá trị Feature Importance chính xác
print("\nFeature Importance:")
for _, row in importance_df.iterrows():
    print(f"{row['Features']}: {row['Importance Score']:.1f}")

# 19. Lưu Feature Importance vào file
importance_df.to_csv('E:/Python codeptit/Bai_tap_lon/Bai_4/feature_importance_1.csv', index=False)