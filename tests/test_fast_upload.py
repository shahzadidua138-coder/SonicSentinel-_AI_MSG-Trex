import io
import json
import urllib.request
import numpy as np
import soundfile as sf

def test():
    sr = 22050
    t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
    # Impulsive wave
    data = (0.7 * np.sin(2 * np.pi * 300 * t) * np.exp(-t * 3)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, data, sr, format='WAV')
    buf.seek(0)
    wav_bytes = buf.read()

    boundary = '----WebKitFormBoundarySample123'
    body = (
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="audio"; filename="impulse_test.wav"\r\n'
        f'Content-Type: audio/wav\r\n\r\n'
    ).encode('latin1') + wav_bytes + f'\r\n--{boundary}--\r\n'.encode('latin1')

    req = urllib.request.Request('http://127.0.0.1:5000/api/audio/upload', data=body)
    req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
    res = urllib.request.urlopen(req)
    print("HTTP Status:", res.getcode())
    resp_data = json.loads(res.read().decode('utf-8'))
    print("Upload Result:", json.dumps({
        "success": resp_data.get("success"),
        "audio_id": resp_data.get("audio_id"),
        "prediction": resp_data.get("prediction", {}).get("final_class"),
        "confidence": resp_data.get("prediction", {}).get("python_confidence"),
        "status": resp_data.get("prediction", {}).get("consistency_status"),
        "quality": resp_data.get("quality", {}).get("quality_grade")
    }, indent=2))

if __name__ == '__main__':
    test()
