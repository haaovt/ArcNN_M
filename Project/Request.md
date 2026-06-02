# Tổng thời gian thực hiện: 15 tuần (từ tuần 26 đến tuần 40), tuần 41 sẽ báo cáo kết thúc
# Kiến thức cơ bản (4 tuần):
1. Linear Regression and Logistic Regression
    - Nội dung: Cơ bản về các phương pháp hồi quy, dùng hồi quy để phân lớp, gradient descent
    - Công việc: 
        + Tìm hiểu về phương pháp hồi quy, ý nghĩa, ý tưởng của phương pháp Linear Regression và Logistic Regression, hàm tối ưu và cách giải nghiệm cho hàm tối ưu.
        + Phân tích sự khác nhau của 2 phương pháp, khi nào cần dùng cái nào
    - Kết quả:
        + Hiểu và triển khai được 2 phương pháp trên pytorch (có thể tự tạo dữ liệu hoặc sử dụng dữ liệu có sẵn, ví dụ như MNIST)

2. MLP
    - Nội dung: Cơ bản về lan truyền thông tin trong MLP, gradient descent, over/underfitting, parameter search
    - Công việc:
        + Tìm hiểu về cách thức hoạt động của MLP (forward, backward propagation), ý nghĩa của hàm kích hoạt, hàm loss, một số hàm loss phổ biến, ý nghĩa? Tại sao cần dùng gradient descent?
        + Over/Underfitting là gì?
    - Kết quả:
        + Tương tự phần trên 


# Triển khai trên dataset lựa chọn (tạm thời là Glasgow) (11 tuần)
1. FMCW radar (2 tuần)
    - Nội dung: Tìm hiểu nguyên lý hoạt động của FMCW radar
    - Công việc: 
        + Tìm hiểu cách trích xuất khoảng cách và vận tốc của vật thể từ tín hiệu

2. Tìm hiểu các kiến trúc dùng để trích xuất đặc trưng và phân lớp (4 tuần):
    - Nội dung: dựa trên kiến thức nền tảng và đặc tính của FMCW radar, tìm hiểu các kiến trúc phát triển hơn để xây dựng mô hình ở phần sau
    - Công việc:
        + Tìm hiểu một số kiến trúc để làm feature extractor. Gợi ý: MLP, CNN
        + Tìm hiểu một số kiến trúc để làm classifer. Gợi ý: MLP, LSTM, BiLSTM, SVM, kNN, Decision Tree
        (không bắt buộc dùng toàn bộ gợi ý, có thể dùng các kiến trúc khác nếu thấy phù hợp)
    - Kết quả:
        + Hiểu và triển khai lại được các kiến trúc lựa chọn trên pytorch
    - Nâng cao: 
        + Sử dụng Transformer, Auto Encoder, Variational Auto Encoder, ResNet...

3. Xây dựng mô hình nhận dạng hành động (HAR) sử dụng FMCW radar (5 tuần):
    - Nội dung: Từ các đặc trưng đầu vào, xây dựng mô hình để dự đoán hành động
    - Công việc:
        + Triển khai mô hình từ các kiến trúc ở trên
        + Thực hiện chia dữ liệu để đánh giá
        + Thực hiện hyperparameter search để có kiến trúc tối ưu
        + Đánh giá mô hình trên các tiêu chí lựa chọn
        + So sánh và nhận xét các kết quả
    - Kết quả:
        + Xây dựng mô hình trên pytorch, chạy dự đoán và đánh giá kết quả


# Tài liệu tham khảo:
    - Youtube - Coursera Neural Network and Deep Learning Specialization Course: https://www.youtube.com/watch?v=CS4cs9xVecg&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0&index=1
    - Dive into Deep Learning - d2l.ai
    - The Elements of Statistical Learning Book by Jerome H. Friedman, Robert Tibshirani, and Trevor Hastie
    - https://www.youtube.com/watch?v=2S1dgHpqCdk&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz





















