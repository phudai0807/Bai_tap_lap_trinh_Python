from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup, Comment
import pandas as pd
import time
from io import StringIO
import os
from functools import reduce
import csv

links_and_filenames = [
    {
        "url": "https://fbref.com/en/comps/9/stats/Premier-League-Stats#all_stats_standard",
        "div_id": "all_stats_standard",
        "file": "E:/Python codeptit/Bai_tap_lon/Bai_1/stats_standard.csv"
    },
    {
        "url": "https://fbref.com/en/comps/9/keepers/Premier-League-Stats#all_stats_keeper_saves",
        "div_id": "all_stats_keeper",
        "file": "E:/Python codeptit/Bai_tap_lon/Bai_1/Goalkeeping.csv"
    },
    {
        "url": "https://fbref.com/en/comps/9/shooting/Premier-League-Stats#all_stats_shooting",
        "div_id": "all_stats_shooting",
        "file": "E:/Python codeptit/Bai_tap_lon/Bai_1/Shooting.csv"
    },
    {
        "url": "https://fbref.com/en/comps/9/passing/Premier-League-Stats#all_stats_passing",
        "div_id": "all_stats_passing",
        "file": "E:/Python codeptit/Bai_tap_lon/Bai_1/stats_passing.csv"
    },
    {
        "url": "https://fbref.com/en/comps/9/gca/Premier-League-Stats#all_stats_gca",
        "div_id": "all_stats_gca",
        "file": "E:/Python codeptit/Bai_tap_lon/Bai_1/stats_gca.csv"
    },
    {
        "url": "https://fbref.com/en/comps/9/defense/Premier-League-Stats#all_stats_defense",
        "div_id": "all_stats_defense",
        "file": "E:/Python codeptit/Bai_tap_lon/Bai_1/stats_defense.csv"
    },
    {
        "url": "https://fbref.com/en/comps/9/possession/Premier-League-Stats#all_stats_possession",
        "div_id": "all_stats_possession",
        "file": "E:/Python codeptit/Bai_tap_lon/Bai_1/stats_possession.csv"
    },
    {
        "url": "https://fbref.com/en/comps/9/misc/Premier-League-Stats#all_stats_misc",
        "div_id": "all_stats_misc",
        "file": "E:/Python codeptit/Bai_tap_lon/Bai_1/stats_misc.csv"
    }
]

# Cấu hình trình duyệt
options = Options()
# options.add_argument("--headless")  # Bật nếu không muốn thấy Chrome chạy

service = Service("D:/chromedriver/chromedriver-win64/chromedriver-win64/chromedriver.exe")
driver = webdriver.Chrome(service=service, options=options)

for item in links_and_filenames:
    url = item["url"]
    div_id = item["div_id"]
    file_path = item["file"]

    driver.get(url)
    time.sleep(5)  # tăng thời gian đợi cho chắc

    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")

    div = soup.find("div", id=div_id)

    # Bảng nằm trong comment?
    if div:
        comment = div.find(string=lambda text: isinstance(text, Comment))
    else:
        comment = None

    table = None

    if comment:
        comment_soup = BeautifulSoup(comment, "html.parser")
        table = comment_soup.find("table")
        print("Bảng nằm trong comment, đã tìm thấy.")
    elif div:
        table = div.find("table")
        print("Bảng nằm ngoài comment, đã tìm thấy.")
    else:
        print(f"Không tìm thấy div id: {div_id}")
        continue

    if table is None:
        print(f"Không tìm thấy bảng trong div id: {div_id}")
        continue

    try:
        df = pd.read_html(StringIO(str(table)))[0]
        print(f"Đọc thành công: {df.shape[0]} dòng, {df.shape[1]} cột")
        df.to_csv(file_path, index=False, encoding="utf-8-sig")
        print(f"Đã lưu vào: {file_path}")
    except Exception as e:
        print(f"Lỗi khi đọc bảng: {e}")

driver.quit()

#Thư mục chứa các file CSV
file_paths = [
    "E:/Python codeptit/Bai_tap_lon/Bai_1/stats_standard.csv",
    "E:/Python codeptit/Bai_tap_lon/Bai_1/Goalkeeping.csv",
    "E:/Python codeptit/Bai_tap_lon/Bai_1/Shooting.csv",
    "E:/Python codeptit/Bai_tap_lon/Bai_1/stats_passing.csv",
    "E:/Python codeptit/Bai_tap_lon/Bai_1/stats_gca.csv",
    "E:/Python codeptit/Bai_tap_lon/Bai_1/stats_defense.csv",
    "E:/Python codeptit/Bai_tap_lon/Bai_1/stats_possession.csv",
    "E:/Python codeptit/Bai_tap_lon/Bai_1/stats_misc.csv"
]
# Lặp qua từng file để xóa dòng đầu
for file_path in file_paths:
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines[1:])  # Bỏ dòng đầu

    print(f"Đã xóa dòng đầu của file: {file_path}")


# BƯỚC 1: Đọc bảng standard
standard_df = pd.read_csv("E:/Python codeptit/Bai_tap_lon/Bai_1/stats_standard.csv")
standard_df.columns = standard_df.columns.str.strip()
print(standard_df.columns.tolist())  # xem tên cột
print(standard_df.head())            # xem vài dòng đầu

# Lọc cầu thủ chơi hơn 90 phút
standard_df["Min"] = pd.to_numeric(standard_df["Min"], errors="coerce")
filtered_df = standard_df[standard_df["Min"] > 90].copy()

# Thêm cột 'First Name'
filtered_df["First Name"] = filtered_df["Player"].apply(lambda x: x.split()[0])
# Sắp xếp theo 'First Name'
filtered_df = filtered_df.sort_values("First Name")

# BƯỚC 2: Định nghĩa hàm gộp các bảng khác 
def merge_stats(main_df, filepath, selected_columns):
    try:
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()
        df = df[["Player", "Squad"] + selected_columns]
        merged_df = pd.merge(main_df, df, on=["Player", "Squad"], how="left")
        return merged_df
    except Exception as e:
        print(f"Lỗi khi gộp từ {filepath}: {e}")
        return main_df


# BƯỚC 3: Gộp từng bảng thống kê cần thiết
#Standard
# Đọc file stats_standard.csv
file_path = "E:/Python codeptit/Bai_tap_lon/Bai_1/stats_standard.csv"
df = pd.read_csv(file_path)

# In ra danh sách các cột để kiểm tra
print("Danh sách cột trong stats_standard.csv:")
print(df.columns.tolist())

# Chọn các cột mong muốn (có thể chỉnh sửa danh sách này tùy nhu cầu)
desired_columns = ["Rk", "Player", "Nation", "Squad", "Pos", "Age", "MP", "Starts", "Min",
                   "Gls", "Ast", "CrdY", "CrdR", "xG", "xAG", "PrgC", "PrgP", "PrgR",
                   "Gls.1", "Ast.1", "xG.1", "xAG.1"]

filtered_df = filtered_df[desired_columns]

# Goalkeeping
filtered_df = merge_stats(filtered_df, "E:/Python codeptit/Bai_tap_lon/Bai_1/Goalkeeping.csv",
                          ["GA90", "Save%", "CS%", "Save%"])

# Shooting
filtered_df = merge_stats(filtered_df, "E:/Python codeptit/Bai_tap_lon/Bai_1/Shooting.csv",
                          ["SoT%", "SoT/90", "G/Sh", "Dist"])

# Passing
filtered_df = merge_stats(filtered_df, "E:/Python codeptit/Bai_tap_lon/Bai_1/stats_passing.csv",
                          ["Cmp", "Cmp%", "TotDist", "Cmp%.1", "Cmp%.2", "Cmp%.3", "KP", "1/3", "PPA", "CrsPA", "PrgP"])

# GCA & SCA
filtered_df = merge_stats(filtered_df, "E:/Python codeptit/Bai_tap_lon/Bai_1/stats_gca.csv",
                          ["SCA", "SCA90", "GCA", "GCA90"])

# Defense
filtered_df = merge_stats(filtered_df, "E:/Python codeptit/Bai_tap_lon/Bai_1/stats_defense.csv",
                          ["Tkl", "TklW", "Att", "Lost", "Blocks", "Sh", "Pass", "Int"])

# Possession
filtered_df = merge_stats(filtered_df, "E:/Python codeptit/Bai_tap_lon/Bai_1/stats_possession.csv",
                          ["Touches", "Def Pen", "Def 3rd", "Mid 3rd", "Att 3rd", "Att Pen",
                           "Att", "Succ%", "Tkld%", "Carries", "PrgDist", "PrgC", "1/3",
                           "CPA", "Mis", "Dis", "Rec", "PrgR"])

# Miscellaneous
filtered_df = merge_stats(filtered_df, "E:/Python codeptit/Bai_tap_lon/Bai_1/stats_misc.csv",
                          ["Fls", "Fld", "Off", "Crs", "Recov", "Won", "Lost", "Won%"])


# BƯỚC 4: Chuẩn hóa dữ liệu
filtered_df = filtered_df.fillna("N/a")
# filtered_df = filtered_df.sort_values("First Name")

# BƯỚC 5: Lưu kết quả
filtered_df.to_csv("E:/Python codeptit/Bai_tap_lon/Bai_1/results.csv", index=False, encoding="utf-8-sig")
print("Đã lưu kết quả vào: results.csv")