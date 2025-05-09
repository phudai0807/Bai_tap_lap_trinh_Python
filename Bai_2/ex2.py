import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Đọc dữ liệu từ file results.csv
df = pd.read_csv("E:/Python codeptit/Bai_tap_lon/Bai_1/results.csv")

# Xem qua dữ liệu
print("Danh sách các cột:")
print(df.columns.tolist())
print("\nDữ liệu mẫu:")
print(df.head())

# Các cột không phải số (bỏ qua khi phân tích)
non_numeric_cols = ['Player', 'Nation', 'Squad', 'Pos', 'Age']

# Mở file để ghi kết quả
with open("E:/Python codeptit/Bai_tap_lon/Bai_2/top_3.txt", 'w', encoding='utf-8') as f:
    for column in df.columns:
        if column not in non_numeric_cols and df[column].dtype in [np.float64, np.int64]:  # Chỉ xét cột số
            f.write(f"Top 3 and Bottom 3 in {column}:\n")
            
            # Sắp xếp giảm dần để tìm top 3
            top_3 = df[['Player', 'Squad', column]].sort_values(by=column, ascending=False).head(3)
            f.write("Top 3:\n")
            for _, row in top_3.iterrows():
                f.write(f"{row['Player']} ({row['Squad']}): {row[column]}\n")
            
            # Sắp xếp tăng dần để tìm thấp nhất 3
            lowest_3 = df[['Player', 'Squad', column]].sort_values(by=column, ascending=True).head(3)
            f.write("Bottom 3:\n")
            for _, row in lowest_3.iterrows():
                f.write(f"{row['Player']} ({row['Squad']}): {row[column]}\n")
            f.write("\n")

# Chọn các chỉ số quan trọng để tính
key_stats = ['Gls', 'Ast', 'xG', 'xAG', 'Tkl', 'Int', 'Cmp%']

# Tạo danh sách để lưu kết quả
results = []

# Tính cho toàn bộ cầu thủ
row_all = [0, 'all']
for stat in key_stats:
    median_all = df[stat].median()
    mean_all = df[stat].mean()
    std_all = df[stat].std()
    row_all.extend([median_all, mean_all, std_all])
results.append(row_all)

# Tính cho từng đội với tên đội bóng cụ thể
teams = df['Squad'].unique()
team_id = 1
for team in teams:
    team_data = df[df['Squad'] == team]
    row_team = [team_id, team]  # Sử dụng tên đội bóng thực tế từ cột 'Squad'
    for stat in key_stats:
        median_team = team_data[stat].median()
        mean_team = team_data[stat].mean()
        std_team = team_data[stat].std()
        row_team.extend([median_team, mean_team, std_team])
    results.append(row_team)
    team_id += 1

# Tạo tiêu đề cột theo định dạng yêu cầu
columns = ['', '']
for stat in key_stats:
    columns.extend([f'Median of {stat}', f'Mean of {stat}', f'Std of {stat}'])

# Tạo DataFrame
results_df = pd.DataFrame(results, columns=columns)

# Lưu vào file CSV
results_df.to_csv("E:/Python codeptit/Bai_tap_lon/Bai_2/results2.csv", index=False, encoding="utf-8-sig")
print("Đã lưu kết quả vào results2.csv")

# Chọn một số chỉ số quan trọng để vẽ
key_stats = ['SCA', 'SCA90', 'GCA', 'Tkl', 'Att_x', 'Blocks']

for column in key_stats:
    # Biểu đồ cho toàn bộ cầu thủ
    plt.figure(figsize=(10, 6))
    plt.hist(df[column].dropna(), bins=20, alpha=0.7, color='blue', label='Tất cả cầu thủ')
    plt.title(f'Phân phối của {column} cho Tất cả cầu thủ')
    plt.xlabel(column)
    plt.ylabel('Tần suất')
    plt.legend()
    plt.show()

    # Biểu đồ cho từng đội
    for team in df['Squad'].unique():
        team_data = df[df['Squad'] == team]
        plt.figure(figsize=(10, 6))
        plt.hist(team_data[column].dropna(), bins=20, alpha=0.7, color='green', label=f'{team}')
        plt.title(f'Phân phối của {column} cho {team}')
        plt.xlabel(column)
        plt.ylabel('Tần suất')
        plt.legend()
        plt.show()

# Từ điển để lưu đội tốt nhất cho mỗi chỉ số
best_teams = {}

for column in df.columns:
    if column not in non_numeric_cols and df[column].dtype in [np.float64, np.int64]:
        team_means = df.groupby('Squad')[column].mean()
        best_team = team_means.idxmax()
        best_teams[column] = (best_team, team_means[best_team])

# In kết quả
print("\nĐội có điểm trung bình cao nhất cho mỗi chỉ số:")
for attr, (team, score) in best_teams.items():
    print(f"{attr}: {team} ({score:.2f})")


# Xác định đội xuất sắc nhất trong mùa giải Premier League 2024-2025

# 1: Đếm số lần mỗi đội dẫn đầu (dựa trên best_teams từ Bước 4)
team_counts = pd.Series([team for team, _ in best_teams.values()]).value_counts()
print("\nSố lần mỗi đội dẫn đầu ở các chỉ số:")
print(team_counts)

# Chuẩn hóa số lần dẫn đầu (normalize từ 0 đến 1)
if not team_counts.empty:
    lead_score = (team_counts - team_counts.min()) / (team_counts.max() - team_counts.min())
else:
    lead_score = pd.Series(0, index=df['Squad'].unique())
    print("Không có dữ liệu dẫn đầu để phân tích.")

# 2: Tính điểm hiệu suất ở các chỉ số quan trọng
key_stats = ['Gls', 'Ast', 'xG', 'xAG', 'Tkl', 'Int', 'Cmp%']
performance_scores = pd.DataFrame(index=df['Squad'].unique())

for stat in key_stats:
    # Tính trung bình của chỉ số cho từng đội
    team_means = df.groupby('Squad')[stat].mean()
    # Chuẩn hóa giá trị (normalize từ 0 đến 1)
    if team_means.notna().any():
        normalized_means = (team_means - team_means.min()) / (team_means.max() - team_means.min())
        performance_scores[stat] = normalized_means
    else:
        performance_scores[stat] = 0  # Nếu không có dữ liệu, gán điểm 0

# Tính điểm hiệu suất trung bình cho mỗi đội
performance_score = performance_scores.mean(axis=1)
print("\nĐiểm hiệu suất (trung bình chuẩn hóa) của các chỉ số quan trọng:")
print(performance_score.sort_values(ascending=False))

# 3: Tính độ ổn định (dựa trên độ lệch chuẩn của các chỉ số quan trọng)
stability_scores = pd.DataFrame(index=df['Squad'].unique())

for stat in key_stats:
    # Tính độ lệch chuẩn của chỉ số cho từng đội
    team_std = df.groupby('Squad')[stat].std()
    # Chuẩn hóa độ lệch chuẩn (normalize từ 0 đến 1, đảo ngược vì std thấp = ổn định cao)
    if team_std.notna().any():
        normalized_std = (team_std - team_std.min()) / (team_std.max() - team_std.min())
        # Đảo ngược: std thấp = điểm cao
        stability_scores[stat] = 1 - normalized_std
    else:
        stability_scores[stat] = 0  # Nếu không có dữ liệu, gán điểm 0

# Tính điểm ổn định trung bình cho mỗi đội
stability_score = stability_scores.mean(axis=1)
print("\nĐiểm ổn định (dựa trên độ lệch chuẩn đảo ngược):")
print(stability_score.sort_values(ascending=False))

# 4: Tính điểm tổng với trọng số
# Trọng số: 40% số lần dẫn đầu, 40% hiệu suất, 20% độ ổn định
final_scores = pd.DataFrame({
    'Lead Score': lead_score,
    'Performance Score': performance_score,
    'Stability Score': stability_score
}).fillna(0)  # Điền 0 cho các giá trị NaN

final_score = (0.4 * final_scores['Lead Score'] + 
               0.4 * final_scores['Performance Score'] + 
               0.2 * final_scores['Stability Score'])

print("\nĐiểm tổng (40% Lead + 40% Performance + 20% Stability):")
print(final_score.sort_values(ascending=False))

# 5: Kết luận đội xuất sắc nhất
best_team = final_score.idxmax()
best_team_score = final_score.max()
print(f"\nĐội xuất sắc nhất trong mùa giải Premier League 2024-2025: {best_team}")
print(f"Điểm: {best_team_score:.2f}")
print("Lý do:")
print(f"- Số lần dẫn đầu: {team_counts.get(best_team, 0)} lần")
print(f"- Điểm hiệu suất: {performance_score[best_team]:.2f}")
print(f"- Điểm ổn định: {stability_score[best_team]:.2f}")