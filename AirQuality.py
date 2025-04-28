import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import seaborn as sns
import joblib
from sklearn.impute import KNNImputer

# 1. Đọc dữ liệu từ file CSV với định dạng đúng
print("Đang đọc dữ liệu từ file AirQualityUCI.csv...")
# Use sep=';' to correctly parse the semicolon-separated data
data = pd.read_csv('AirQualityUCI.csv', sep=';')

# 2. Xử lý dữ liệu ban đầu
# Replace -200 values with NaN as they likely represent missing values
data = data.replace(-200, np.nan)
# Loại bỏ các cột không có bất kỳ giá trị nào (tất cả là NaN hoặc bị lỗi đọc)
data = data.dropna(axis=1, how='all')
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]


# Convert comma to dot in numeric columns (European format issue)
for col in data.columns:
    if data[col].dtype == 'object':
        try:
            data[col] = data[col].str.replace(',', '.').astype(float)
        except:
            pass

# 3. Khám phá dữ liệu
print("\nThông tin dữ liệu:")
print(data.info())
print("\nXem 5 dòng đầu tiên:")
print(data.head())
print("\nThống kê mô tả:")
print(data.describe())

# Kiểm tra giá trị null
print("\nKiểm tra giá trị null:")
null_values = data.isnull().sum()
print(null_values)

# 4. Phân tích dữ liệu
# Only include numeric columns in correlation analysis
numeric_data = data.select_dtypes(include=[np.number])
print("\nPhân tích tương quan giữa các biến:")
plt.figure(figsize=(14, 12))
correlation_matrix = numeric_data.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt='.2f')
plt.title('Ma trận tương quan giữa các biến')
plt.tight_layout()
plt.savefig('correlation_matrix.png')

# Phân tích tương quan với biến mục tiêu CO(GT)
if 'CO(GT)' in correlation_matrix.index:
    co_correlation = correlation_matrix['CO(GT)'].sort_values(ascending=False)
    print("\nTương quan của các biến với CO(GT):")
    print(co_correlation)

# 5. Tiền xử lý dữ liệu
# Nếu có cột Date và Time, chuyển đổi thành datetime
if 'Date' in data.columns and 'Time' in data.columns:
    print("\nChuyển đổi cột Date và Time thành DateTime...")
    try:
        # Kiểm tra định dạng của các cột Date và Time
        print("Mẫu giá trị Date:", data['Date'].iloc[0])
        print("Mẫu giá trị Time:", data['Time'].iloc[0])
        
        # Chỉ định format cụ thể (điều chỉnh theo định dạng thực tế của dữ liệu)
        data['DateTime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], 
                                          format='%d/%m/%Y %H.%M.%S')
        
        # Tạo các đặc trưng thời gian
        data['Hour'] = data['DateTime'].dt.hour
        data['Day'] = data['DateTime'].dt.day
        data['Month'] = data['DateTime'].dt.month
        data['DayOfWeek'] = data['DateTime'].dt.dayofweek
        
        # Loại bỏ các cột không cần thiết
        columns_to_drop = ['Date', 'Time', 'DateTime']
        print("Chuyển đổi DateTime thành công!")
    except Exception as e:
        print(f"Không thể chuyển đổi Date và Time thành DateTime. Lỗi: {e}")
        columns_to_drop = ['Date', 'Time']
else:
    columns_to_drop = []

# Chọn đặc trưng và biến mục tiêu
target_column = 'CO(GT)'
# Make sure target column exists
if target_column not in data.columns:
    raise ValueError(f"Cột mục tiêu '{target_column}' không tồn tại trong dữ liệu!")

# Remove any non-numeric columns that remain
numeric_columns = data.select_dtypes(include=[np.number]).columns
non_numeric_cols = [col for col in data.columns if col not in numeric_columns and col not in columns_to_drop]
if non_numeric_cols:
    print(f"Loại bỏ các cột không phải số: {non_numeric_cols}")
    columns_to_drop.extend(non_numeric_cols)

# Drop columns and prepare features/target
X = data.drop([target_column] + columns_to_drop, axis=1, errors='ignore')
y = data[target_column]

# Check for and handle NaN values in both X and y
print("\nXử lý giá trị NaN:")
print(f"Số lượng NaN trong X: {X.isna().sum().sum()}")
print(f"Số lượng NaN trong y: {y.isna().sum()}")

# Remove rows where target variable is NaN
if y.isna().sum() > 0:
    print(f"Loại bỏ {y.isna().sum()} dòng có giá trị NaN trong biến mục tiêu")
    valid_indices = y.notna()
    X = X.loc[valid_indices]
    y = y.loc[valid_indices]

# Handle remaining NaN values in features using imputation
if X.isna().sum().sum() > 0:
    print("Sử dụng KNNImputer để xử lý các giá trị thiếu...")
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)


print(f"\nSố lượng đặc trưng sử dụng: {X.shape[1]}")
print(f"Các đặc trưng: {X.columns.tolist()}")

print("Tạo các đặc trưng tương tác và bậc hai...")

# Thêm đặc trưng bậc hai (polynomial) và tương tác đơn giản
X['PT08.S1_CO*NOx(GT)'] = X['PT08.S1(CO)'] * X['NOx(GT)']
X['PT08.S5_O3**2'] = X['PT08.S5(O3)'] ** 2
X['NO2_GT*RH'] = X['NO2(GT)'] * X['RH']


# 4. Loại bỏ outlier bằng IQR
def remove_outliers_iqr(df, target, multiplier=1.5):
    combined = pd.concat([df, target], axis=1)
    for col in combined.columns:
        Q1 = combined[col].quantile(0.25)
        Q3 = combined[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        combined = combined[(combined[col] >= lower_bound) & (combined[col] <= upper_bound)]
    return combined.drop(columns=[target_column]), combined[target_column]

X, y = remove_outliers_iqr(X, y)

# 6. Chia dữ liệu thành tập huấn luyện và kiểm thử
print("\nChia dữ liệu thành tập huấn luyện và kiểm thử (80% - 20%)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Kích thước tập huấn luyện: {X_train.shape}")
print(f"Kích thước tập kiểm thử: {X_test.shape}")

# Chuẩn hóa dữ liệu
print("\nChuẩn hóa dữ liệu...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Xây dựng mô hình Linear Regression
print("\nHuấn luyện mô hình Linear Regression với ràng buộc không âm...")
model = LinearRegression(positive=True)
model.fit(X_train_scaled, y_train)


# In ra hệ số của mô hình
print("\nHệ số (coefficients) của mô hình:")
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
coef_df = coef_df.sort_values(by='Coefficient', ascending=False)
print(coef_df)

print("\nIntercept của mô hình:", model.intercept_)

# 8. Đánh giá mô hình
print("\nĐánh giá mô hình trên tập kiểm thử...")
y_pred_test = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R²): {r2:.4f}")

# 9. Visualization
# Biểu đồ so sánh giá trị thực tế và dự đoán
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Giá trị CO(GT) thực tế')
plt.ylabel('Giá trị CO(GT) dự đoán')
plt.title('So sánh giá trị thực tế và dự đoán của CO(GT)')
plt.grid(True)
plt.savefig('actual_vs_predicted.png')
plt.close()

# Histogram của sai số
plt.figure(figsize=(10, 6))
residuals = y_test - y_pred_test
sns.histplot(residuals, kde=True)
plt.xlabel('Sai số dự đoán')
plt.ylabel('Tần suất')
plt.title('Phân phối sai số dự đoán')
plt.grid(True)
plt.savefig('residuals_histogram.png')
plt.close()

# Biểu đồ phân tán của sai số
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_test, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Giá trị dự đoán')
plt.ylabel('Sai số')
plt.title('Biểu đồ phân tán sai số')
plt.grid(True)
plt.savefig('residuals_scatter.png')
plt.close()

# Biểu đồ thể hiện mức độ quan trọng của các đặc trưng
plt.figure(figsize=(12, 10))
coef_df = coef_df.sort_values(by='Coefficient')
plt.barh(coef_df['Feature'], coef_df['Coefficient'])
plt.xlabel('Hệ số')
plt.ylabel('Đặc trưng')
plt.title('Mức độ ảnh hưởng của các đặc trưng đến CO(GT)')
plt.grid(True, axis='x')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Top 5 đặc trưng quan trọng nhất dựa trên giá trị tuyệt đối của hệ số
print("\nTop 5 đặc trưng quan trọng nhất:")
coef_abs = coef_df.copy()
coef_abs['Absolute'] = np.abs(coef_abs['Coefficient'])
top_features = coef_abs.sort_values(by='Absolute', ascending=False).head(5)
print(top_features[['Feature', 'Coefficient']])

# 10. Lưu mô hình
print("\nLưu mô hình...")
joblib.dump(model, 'co_prediction_model.pkl')
joblib.dump(scaler, 'co_scaler.pkl')
joblib.dump(list(X.columns), 'co_model_features.pkl')
print("Mô hình đã được lưu thành công!")

# 11. Hàm dự đoán
def predict_co(new_data):
    """
    Hàm dự đoán giá trị CO(GT) cho dữ liệu mới
    
    Parameters:
    new_data (DataFrame): DataFrame chứa các đặc trưng cần thiết
    
    Returns:
    float: Giá trị CO(GT) dự đoán
    """
    required_features = list(X.columns)
    if not all(feature in new_data.columns for feature in required_features):
        missing = [f for f in required_features if f not in new_data.columns]
        raise ValueError(f"Dữ liệu thiếu các đặc trưng: {missing}")
    
    # Chỉ sử dụng các đặc trưng cần thiết
    new_data = new_data[required_features]
    
    # Handle any NaN values in new data
    if new_data.isna().any().any():
        imputer = SimpleImputer(strategy='mean')
        new_data = pd.DataFrame(
            imputer.fit_transform(new_data),
            columns=new_data.columns
        )
    
    # Chuẩn hóa dữ liệu mới
    new_data_scaled = scaler.transform(new_data)
    
    # Dự đoán
    prediction = model.predict(new_data_scaled)
    
    return prediction

# Ví dụ sử dụng
print("\nVí dụ sử dụng mô hình để dự đoán:")
sample_data = X_test.iloc[0:5]  # Lấy 5 mẫu đầu tiên từ tập kiểm thử
sample_predictions = predict_co(sample_data)
actual_values = y_test.iloc[0:5].values

# In kết quả so sánh
comparison = pd.DataFrame({
    'Thực tế': actual_values,
    'Dự đoán': sample_predictions.flatten(),
    'Sai số': np.abs(actual_values - sample_predictions.flatten())
})
print(comparison)
print(f"\nSai số trung bình: {comparison['Sai số'].mean():.4f}")

print("\nKết thúc quá trình xây dựng và đánh giá mô hình Linear Regression cho dự đoán CO(GT).")