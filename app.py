import cv2
import proctor as ct
import Audio as ad
from flask import Flask, render_template, Response

app = Flask(__name__)

camera = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, img = camera.read()
        if not success:
            break
        else:
            img = ct.face_checker(img) # Face spoofing

            frame = ct.head_position(img) #  Head position
            
            frame = ct.object_detection(img) #Object detection - Yolo V3 models

            ad.Audio() # Audio detection
            
            ret, buffer = cv2.imencode('.jpg', frame)

            frame = buffer.tobytes()
            
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/welcome')
def welcome():
    # camera.release()
    # cv2.destroyAllWindows()
    return render_template('welcome.html')

if __name__ == '__main__':
    app.run(debug=True)