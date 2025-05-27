from flask import Blueprint, request, jsonify
from app.services.videoAnalysis import analyze_video
from app.utils.logger import setup_logger
import os
import tempfile

logger = setup_logger(__name__)

api = Blueprint('api', __name__)

@api.route('/video-analysis-batch', methods=['POST'])
def video_analysis_batch_api():
    try:
        if 'videos' not in request.files:
            logger.warning("Missing video files in request.")
            return jsonify({'success': False, 'error': 'At least one video file is required'}), 400

        video_files = request.files.getlist('videos')
        results = []

        for video_file in video_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp:
                video_file.save(tmp.name)
                temp_path = tmp.name

            try:
                avg_conf, avg_eye, feedback = analyze_video(temp_path)
                results.append({
                    'filename': video_file.filename,
                    'average_confidence': round(avg_conf, 2),
                    'average_eye_contact': round(avg_eye, 2),
                    'feedback': feedback
                })
            except Exception as ve:
                logger.error(f"Error analyzing video {video_file.filename}: {ve}", exc_info=True)
                results.append({
                    'filename': video_file.filename,
                    'error': 'Failed to analyze this video'
                })
            finally:
                os.remove(temp_path)

        return jsonify({'success': True, 'results': results})

    except Exception as e:
        logger.error(f"Error during batch video analysis: {e}", exc_info=True)
        return jsonify({'success': False, 'error': 'Internal server error'}), 500