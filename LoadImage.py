import requests
from bs4 import BeautifulSoup
import os
from pdf2image import convert_from_path

# URL trang cần truy cập
url = 'https://vanban.chinhphu.vn/he-thong-van-ban?classid=1&mode=1'

# Gửi yêu cầu tới trang web
response = requests.get(url)
response.raise_for_status()

# Phân tích HTML
soup = BeautifulSoup(response.text, 'html.parser')

# Tìm danh sách các đường dẫn tới PDF trong cột "Trích yếu"
documents = soup.find_all('a', href=True)

# Tạo thư mục để lưu file PDF và ảnh JPG (trong D:\NCKH\Loaded_Images)
pdf_save_dir = 'D:/NCKH/Loaded_Images/pdf'
jpg_save_dir = 'D:/NCKH/Loaded_Images/jpg'
os.makedirs(pdf_save_dir, exist_ok=True)
os.makedirs(jpg_save_dir, exist_ok=True)

# Chọn tối đa 500 văn bản để tải xuống
for idx, doc in enumerate(documents[:500]):
    doc_url = doc['href']
    
    # Nếu link là tương đối, nối thêm base_url
    if not doc_url.startswith('http'):
        # Assuming the relative URL is within the same domain
        doc_url = 'https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/8/' + doc_url

    # Kiểm tra xem liên kết có phải là file PDF hay không
    if doc_url.endswith('.pdf'):
        try:
            # Tải xuống file PDF
            pdf_response = requests.get(doc_url)
            original_file_name = doc_url.split('/')[-1]  # Lấy tên file gốc từ URL
            pdf_path = f'{pdf_save_dir}/{original_file_name}'  # Sử dụng tên file gốc để lưu
            with open(pdf_path, 'wb') as pdf_file:
                pdf_file.write(pdf_response.content)
            
            print(f'Tải xuống thành công file PDF: {pdf_path}')

            # Chuyển đổi PDF sang JPG
            images = convert_from_path(pdf_path)
            for i, image in enumerate(images):
                # Sử dụng tên file gốc để tạo tên ảnh
                img_output_path = f'{jpg_save_dir}/{original_file_name.replace(".pdf", f"_page_{i}.jpg")}'
                image.save(img_output_path, 'JPEG')
                print(f'Chuyển đổi thành công {pdf_path} trang {i} thành {img_output_path}')
        
        except Exception as e:
            print(f'Không thể tải hoặc chuyển đổi file PDF từ {doc_url}: {e}')
