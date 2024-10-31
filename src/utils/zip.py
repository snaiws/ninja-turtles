import os
import zipfile

def zip_folder(folder_path, zip_file_path):
    # 압축 파일 생성
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 폴더 내의 모든 파일과 폴더를 순회
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # 파일을 압축 파일에 추가
                file_path = os.path.join(root, file)
                # 압축 파일 내에서의 상대 경로를 계산하여 저장
                zipf.write(file_path, os.path.relpath(file_path, folder_path))  