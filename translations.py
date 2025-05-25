# translations.py
translations = {
    'en': {
        'site_title': 'Car Identifier', # Updated to be more general
        'lang_english': 'English',
        'lang_korean': '한국어',
        
        # General UI & Labels
        'page_header': 'Upload Car Image for Brand Prediction', # This might be too specific if you have tabs now
        'select_image_label': 'Click to select an image or drag & drop',
        'predict_button': 'Identify Car', # More generic
        'result_header': 'Identification Result',
        'predicted_brand_is': 'Predicted Brand:',
        'predicted_model_is': 'Predicted Model:',
        'confidence_prefix': 'Confidence:',
        'future_work_header': 'Future Work',
        'future_work_text': 'This is a placeholder for future improvements, model accuracy display, more brands, user accounts, etc.',
        'selected_file_prefix_js': 'Selected file: ',
        'no_file_selected_js': 'No file selected',

        # Tab Navigation
        'tab_image_upload': 'Identify Your Image',
        'tab_video_analysis': 'Identify From Video',

        # Image Upload Tab Specific
        'header_image_upload': 'Upload Image for Identification',
        'button_identify_full': 'Identify Full Image',
        'button_identify_crop': 'Identify Cropped Area',
        'button_start_crop': 'Select Area to Crop',
        'button_reset_crop': 'Reset Crop Selection',
        'alert_no_crop_selected': 'Please select an area on the image to crop first.',


        # Video Analysis Tab Specific
        'header_video_analysis': 'Analyze Cars in Video',
        'label_select_video': 'Select Video',
        'video_instructions': 'Play the video and pause it to identify cars in the current frame.',
        
        # Error Messages (Flash and Prediction Results)
        'error_no_file_part': 'No file part in the request.',
        'error_no_file_selected_flash': 'No file selected for uploading.',
        'error_could_not_save': 'Could not save file. Error: {error}',
        'error_invalid_file_type': 'Invalid file type. Allowed types: png, jpg, jpeg, gif.',
        'error_low_brand_confidence': "Could not reliably identify brand. Best guess: {brand} ({conf:.2f}%) - confidence too low.",
        'error_no_models_for_brand': "Brand identified: {brand}, but no specific models are listed for it in our database.",
        'error_inconsistent_model_list': "Brand identified: {brand}, but its model list is inconsistent or empty after validation.",
        'error_no_valid_models_for_brand_logits': "Brand: {brand}. No valid model logits found after filtering for this brand.",
        'error_low_model_confidence': "Model identified: {model} ({conf:.2f}%) - confidence too low for reliable prediction.",
        'error_prediction_failed_general': "An error occurred during the identification process. Please try again.",
        'error_no_car_detected': 'No car was detected in the uploaded image or selected crop.', # For YOLO
        'error_cropping_failed': 'Failed to process the cropped area from the image.', # For image crop issues
        'error_detection_failed': 'Object detection process failed. Error: {error}', # For YOLO failures
        'error_video_frame_decode': 'Could not decode the video frame image data.',
        'error_video_prediction_failed': 'Prediction process failed for the video frame.',
        'error_no_videos_found': 'No videos found in the video directory.',


    },
    'ko': {
        'site_title': '자동차 식별기', # 업데이트됨
        'lang_english': 'English',
        'lang_korean': '한국어',

        # 일반 UI 및 레이블
        'page_header': '자동차 이미지 업로드하여 브랜드 예측', # 탭이 있다면 너무 구체적일 수 있음
        'select_image_label': '이미지를 선택하거나 드래그 앤 드롭하세요',
        'predict_button': '자동차 식별', # 더 일반적
        'result_header': '식별 결과',
        'predicted_brand_is': '예측된 브랜드:',
        'predicted_model_is': '예측된 모델:',
        'confidence_prefix': '신뢰도:',
        'future_work_header': '향후 작업',
        'future_work_text': '향후 개선 사항 (모델 정확도 표시, 더 많은 브랜드, 사용자 계정 등)을 위한 공간입니다.',
        'selected_file_prefix_js': '선택된 파일: ',
        'no_file_selected_js': '선택된 파일 없음',

        # 탭 네비게이션
        'tab_image_upload': '내 이미지 식별',
        'tab_video_analysis': '영상에서 식별',

        # 이미지 업로드 탭 관련
        'header_image_upload': '식별할 이미지 업로드',
        'button_identify_full': '전체 이미지 식별',
        'button_identify_crop': '자른 영역 식별',
        'button_start_crop': '자를 영역 선택',
        'button_reset_crop': '영역 선택 초기화',
        'alert_no_crop_selected': '먼저 이미지에서 자를 영역을 선택해주세요.',

        # 비디오 분석 탭 관련
        'header_video_analysis': '영상 속 자동차 분석',
        'label_select_video': '비디오 선택',
        'video_instructions': '비디오를 재생하고 현재 프레임의 자동차를 식별하려면 일시정지하세요.',

        # 오류 메시지 (플래시 및 예측 결과)
        'error_no_file_part': '요청에 파일 부분이 없습니다.',
        'error_no_file_selected_flash': '업로드할 파일이 선택되지 않았습니다.',
        'error_could_not_save': '파일을 저장할 수 없습니다. 오류: {error}',
        'error_invalid_file_type': '잘못된 파일 형식입니다. 허용되는 형식: png, jpg, jpeg, gif.',
        'error_low_brand_confidence': "브랜드를 확실히 식별할 수 없습니다. 최선 추정: {brand} ({conf:.2f}%) - 신뢰도가 너무 낮습니다.",
        'error_no_models_for_brand': "브랜드 식별됨: {brand}, 하지만 데이터베이스에 해당 브랜드의 특정 모델 목록이 없습니다.",
        'error_inconsistent_model_list': "브랜드 식별됨: {brand}, 하지만 유효성 검사 후 모델 목록이 일치하지 않거나 비어 있습니다.",
        'error_no_valid_models_for_brand_logits': "브랜드: {brand}. 이 브랜드에 대해 필터링 후 유효한 모델 로짓을 찾을 수 없습니다.",
        'error_low_model_confidence': "모델 식별됨: {model} ({conf:.2f}%) - 신뢰도가 너무 낮아 확실한 예측이 어렵습니다.",
        'error_prediction_failed_general': "식별 과정 중 오류가 발생했습니다. 다시 시도해주세요.",
        'error_no_car_detected': '업로드된 이미지 또는 선택한 영역에서 자동차가 감지되지 않았습니다.', # YOLO 용
        'error_cropping_failed': '이미지에서 선택한 영역을 처리하지 못했습니다.', # 이미지 자르기 문제 용
        'error_detection_failed': '객체 감지 과정에 실패했습니다. 오류: {error}', # YOLO 실패 용
        'error_video_frame_decode': '비디오 프레임 이미지 데이터를 디코딩할 수 없습니다.',
        'error_video_prediction_failed': '비디오 프레임에 대한 예측 과정에 실패했습니다.',
        'error_no_videos_found': '비디오 디렉토리에서 비디오를 찾을 수 없습니다.',
    }
}