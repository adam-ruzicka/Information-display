from datetime import datetime
import importlib.util
import json
import os
import serial
import sys
import time

from threading import Thread
import threading

import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

import zlib
import pyzbar.pyzbar
import base45
import base64
import cbor2
from cose.headers import Algorithm, KID
from cose.messages import CoseMessage
from cose.keys import cosekey, ec2, keyops, curves
from typing import Dict, Tuple, Optional
from pyasn1.codec.ber import decoder as asn1_decoder
from cryptojwt import jwk as cjwtk
from cryptojwt import utils as cjwt_utils

from create_img import create_info_image


class VideoStream:
    """Camera object that controls video streaming from the USB camera"""

    def __init__(self, resolution=(480, 640), framerate=30):
        # Initialize the USB camera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        # print(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])

        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

        # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
        # Start the thread that reads frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Return the most recent frame
        return self.frame

    def stop(self):
        # Indicate that the camera and thread should be stopped
        self.stopped = True


def start_qr_detection():
    """Start QR code detection process"""

    global detecting_qr

    detecting_qr = True
    timer = threading.Timer(60.0, end_qr_detection)
    timer.start()


def end_qr_detection():
    """End QR code detection process"""

    global detected_qr
    global detecting_qr
    global qr_signed
    global qr_verified
    global qr_valid
    global displaying

    detected_qr = None
    detecting_qr = False

    qr_signed = False
    qr_verified = False
    qr_valid = False
    displaying = False


def center_text(image, text, color, offset=0):
    """Draw centered text"""

    text_w, text_h = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
    center_coordinates = (int(image.shape[1] / 2) - int(text_w / 2), int(image.shape[0] / 2) - int(text_h / 2) + offset)

    ft = cv2.freetype.createFreeType2()
    ft.loadFontData(fontFileName='arial.ttf', id=0)
    ft.putText(img=image, text=text, org=center_coordinates,
               fontHeight=40, color=color, thickness=-1, line_type=cv2.LINE_AA,
               bottomLeftOrigin=True)


def center_rectangle(image, text, color, start=0, end=0):
    """Draw rectangle behind text"""

    text_w, text_h = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 2, 3)[0]
    text_x, text_y = int(image.shape[1] / 2) - int(text_w / 2), int(image.shape[0] / 2) - int(text_h / 2)
    output = image.copy()
    cv2.rectangle(image, (text_x - 10, text_y - 50 + start), (text_x + text_w + 10, text_y + text_h + 10 + end), color,
                  -1)

    return output


def detection_process(frame_resized):
    """Detect face mask"""

    global dst
    global frame1
    global object_name
    global detected_temps
    global detected_objects
    global scanner_index
    global go_closer
    global go_left
    global go_right

    input_data = np.expand_dims(frame_resized, axis=0)
    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects
    num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects
    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        object_name = ""
        if ((scores[0] > 0.95) and (scores[0] <= 1.0)):
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1, (boxes[0][0] * imH)))
            xmin = int(max(1, (boxes[0][1] * imW)))
            ymax = int(min(imH, (boxes[0][2] * imH)))
            xmax = int(min(imW, (boxes[0][3] * imW)))
            if 110 <= xmin <= 170 and ymin <= 200 and 300 <= xmax <= 420 and 560 <= ymax <= 640:
                if xmin >= 160:
                    plus_value = 12
                elif xmin >= 140:
                    plus_value = 7
                else:
                    plus_value = 5
                edited_temp = detected_temperature + plus_value
                if 34 < edited_temp < 45 and edited_temp not in detected_temps:
                    detected_temps.append(edited_temp)
                object_name = labels[int(classes[0])]  # Look up object name from "labels" array using class index
                detected_objects.append(object_name)
                # print(object_name)
                go_closer = False
                go_left = False
                go_right = False
            else:
                if 130 <= xmin <= 220 and ymin <= 300 and 250 <= xmax <= 340 and 500 <= ymax <= 640:
                    go_closer = True
                if 180 <= xmin <= 230 and ymin <= 300 and 290 <= xmax <= 420 and 500 <= ymax <= 640:
                    go_left = True
                if xmin <= 110 and ymin <= 300 and 200 <= xmax <= 300 and 500 <= ymax <= 640:
                    go_right = True
                detected_temps.clear()
                detected_objects.clear()
                scanner_index = 0


def show_video():
    """Handle every possible state"""

    global dst
    global frame1
    global object_name
    global detecting_qr
    global detected_temperature
    global detected_temps
    global detected_objects
    global displaying
    global displaying_counter
    global scanner_index
    global merged
    global merged2
    global go_closer
    global go_left
    global go_right

    detected_avg = 0
    detected_obj = ''
    detected_avg_ok = False
    while True:
        frame1 = videostream.read()
        frame1 = cv2.flip(frame1, 1)
        if not displaying:
            if not detecting_qr and (object_name == "mask" or object_name == "no mask") and len(detected_temps) > 0:
                center_text(image=frame1, text="Prebieha meranie...", color=(255, 255, 0), offset=0)

                scanner = cv2.resize(scanner_images[int(scanner_index % SCANNER_IMAGES_COUNT * 2)],
                                     (frame1.shape[1], frame1.shape[0]))

                dst = cv2.addWeighted(frame1, 0.5, scanner, 0.2, 0)

                scanner_index += SCANNER_INDEX_SPEED
            elif detecting_qr:
                center_text(image=frame1, text="Naskenuj COVID-19 certifikát", color=(255, 255, 0), offset=0)

                dst = cv2.addWeighted(frame1, 0.6, merged2, 0.8, 0)
            else:
                if go_closer:
                    center_text(image=frame1, text="Pristúp bližšie", color=(255, 255, 255), offset=0)
                elif go_left:
                    center_text(image=frame1, text="Posuň sa doľava", color=(255, 255, 255), offset=0)
                elif go_right:
                    center_text(image=frame1, text="Posuň sa doprava", color=(255, 255, 255), offset=0)

                dst = cv2.addWeighted(frame1, 0.6, merged, 0.8, 0)
        else:
            displaying_counter += 1

            if displaying_counter > 100:
                displaying_counter = 0
                displaying = False

            if detected_obj != "" and not detecting_qr:
                measured_temperature = 'Nameraná teplota: {0:.1f}'.format(detected_avg) + '°C'
                output = center_rectangle(frame1, measured_temperature, (0, 0, 0), -50, 50)
                center_text(image=frame1, text=measured_temperature, color=(255, 255, 0), offset=-50)
                center_text(image=frame1, text=metadata[detected_obj][detected_avg_ok]['title'],
                            color=metadata[detected_obj][detected_avg_ok]['color_temp'], offset=0)
                center_text(image=frame1, text=metadata[detected_obj][detected_avg_ok]['name'],
                            color=metadata[detected_obj][detected_avg_ok]['color_mask'], offset=50)

                dst = cv2.addWeighted(frame1, 0.5, output, 0.3, 0)

            if detected_qr is not None:
                if qr_signed and qr_verified and qr_valid:
                    cert_text = 'Certifikát je platný'
                    output2 = center_rectangle(frame1, cert_text, (0, 0, 0))
                    center_text(image=frame1, text=cert_text, color=(0, 255, 0), offset=0)
                else:
                    cert_text = 'Certifikát je neplatný!'
                    output2 = center_rectangle(frame1, cert_text, (0, 0, 0))
                    center_text(image=frame1, text=cert_text, color=(0, 0, 255), offset=0)

                dst = cv2.addWeighted(frame1, 0.5, output2, 0.3, 0)

            scanner_index = 0

        if len(detected_temps) > 5 and len(detected_objects) > 10:
            detected_avg = sum(detected_temps) / len(detected_temps)
            detected_avg_ok = 30 <= detected_avg <= 37
            detected_temps.clear()
            detected_obj = list(set([x for x in detected_objects if detected_objects.count(x) > 10 - 1]))[0]
            detected_objects.clear()
            displaying = True

            timer = threading.Timer(2.0, start_qr_detection)
            timer.start()

        # cv2.namedWindow("Object detector", cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty("Object detector", cv2.WND_PROP_FULLSCREEN, 1)
        cv2.imshow('Detekcia masky a teploty', dst)
        if cv2.waitKey(1) == ord('q'):
            break


def detect_frame():
    """Detect frames in an infinite loop"""

    global frame

    while True:
        frame = videostream.read()
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        detection_process(frame_resized)


def detect_temperature():
    """Detect temperature in an infinite loop"""

    while True:
        global detected_temperature
        global serial_port
        line = serial_port.readline().decode('utf-8').rstrip()
        if line:
            detected_temperature = json.loads(line)["ObjectTemp"] + 2


def find_key(key: Algorithm, keys_file: str) -> Optional[cosekey.CoseKey]:
    """Read the JSON-database of all known keys"""

    with open(keys_file, encoding='utf-8') as f:
        known_keys = json.load(f)

    jwt_key = None
    for key_id, key_data in known_keys.items():
        key_id_binary = base64.b64decode(key_id)
        if key_id_binary == key:
            # print("Found the key from DB!")
            # check if the point is uncompressed rather than compressed
            x, y = public_ec_key_points(base64.b64decode(key_data['publicKeyPem']))
            key_dict = {'crv': key_data['publicKeyAlgorithm']['namedCurve'],  # 'P-256'
                        'kid': key_id_binary.hex(),
                        'kty': key_data['publicKeyAlgorithm']['name'][:2],  # 'EC'
                        'x': x,  # 'eIBWXSaUgLcxfjhChSkV_TwNNIhddCs2Rlo3tdD671I'
                        'y': y,  # 'R1XB4U5j_IxRgIOTBUJ7exgz0bhen4adlbHkrktojjo'
                        }
            jwt_key = cosekey_from_jwk_dict(key_dict)
            break

    if not jwt_key:
        return None

    if jwt_key.kid.decode() != key.hex():
        raise RuntimeError("Internal error: No key for {0}!".format(key.hex()))

    return jwt_key


def public_ec_key_points(public_key: bytes) -> Tuple[str, str]:
    """Adapted from: https://stackoverflow.com/a/59537764/1548275"""

    public_key_asn1, _remainder = asn1_decoder.decode(public_key)
    public_key_bytes = public_key_asn1[1].asOctets()

    off = 0
    if public_key_bytes[off] != 0x04:
        raise ValueError("EC public key is not an uncompressed point")
    off += 1

    size_bytes = (len(public_key_bytes) - 1) // 2

    x_bin = public_key_bytes[off:off + size_bytes]
    x = int.from_bytes(x_bin, 'big', signed=False)
    off += size_bytes

    y_bin = public_key_bytes[off:off + size_bytes]
    y = int.from_bytes(y_bin, 'big', signed=False)
    off += size_bytes

    bl = (x.bit_length() + 7) // 8
    bytes_val = x.to_bytes(bl, 'big')
    x_str = base64.b64encode(bytes_val, altchars='-_'.encode()).decode()

    bl = (y.bit_length() + 7) // 8
    bytes_val = y.to_bytes(bl, 'big')
    y_str = base64.b64encode(bytes_val, altchars='-_'.encode()).decode()

    return x_str, y_str


def cosekey_from_jwk_dict(jwk_dict: Dict) -> cosekey.CoseKey:
    """Read key and return CoseKey"""

    if jwk_dict["kty"] != "EC":
        raise ValueError("Only EC keys supported")
    if jwk_dict["crv"] != "P-256":
        raise ValueError("Only P-256 supported")

    from pprint import pprint
    key = ec2.EC2(
        crv=curves.P256,
        x=cjwt_utils.b64d(jwk_dict["x"].encode()),
        y=cjwt_utils.b64d(jwk_dict["y"].encode()),
    )
    key.key_ops = [keyops.VerifyOp]
    if "kid" in jwk_dict:
        key.kid = bytes(jwk_dict["kid"], "UTF-8")

    return key


def read_cosekey_from_pem_file(cert_file: str) -> cosekey.CoseKey:
    """Read certificate, calculate kid and return EC CoseKey"""

    if not cert_file.endswith(".pem"):
        raise ValueError("Unknown key format. Use .pem keyfile")

    with open(cert_file, 'rb') as f:
        cert_data = f.read()
        # Calculate Hash from the DER format of the Certificate
        cert = x509.load_pem_x509_certificate(cert_data, hazmat.backends.default_backend())
        keyidentifier = cert.fingerprint(hazmat.primitives.hashes.SHA256())
    f.close()
    key = cert.public_key()

    jwk = cjwtk.ec.ECKey()
    jwk.load_key(key)
    # Use first 8 bytes of the hash as Key Identifier (Hex as UTF-8)
    jwk.kid = keyidentifier[:8].hex()
    jwk_dict = jwk.serialize(private=False)

    return cosekey_from_jwk_dict(jwk_dict)


def verify_signature(cose_msg: CoseMessage, key: cosekey.CoseKey) -> bool:
    """Verify certificate signature"""

    cose_msg.key = key
    if not cose_msg.verify_signature():
        print("Podpis nepasuje ku kľúču s ID {0}!".format(key.kid.decode()))
        return False

    print("Podpis úspešne overený")

    return cose_msg.verify_signature()


def detect_qr_certificate():
    """Detect QR certificate in an infinite loop when needed"""

    global frame
    global displaying
    global detecting_qr
    global detected_qr
    global qr_signed
    global qr_verified
    global qr_valid

    while True:
        frame = videostream.read()
        frame = cv2.flip(frame, 1)
        if detecting_qr:
            detected = pyzbar.pyzbar.decode(frame)

            if detected:
                cert = detected[0].data.decode()
                b45data = cert.replace("HC1:", "")
                zlibdata = base45.b45decode(b45data)
                cbordata = zlib.decompress(zlibdata)
                cose_msg = CoseMessage.decode(cbordata)

                if KID in cose_msg.phdr:
                    print("COVID-19 certifikát je podpísaný X.509 certifikátom")
                    print("X.509 vo formáte DER má SHA-256 začínajúci na: {0}".format(cose_msg.phdr[KID].hex()))
                    key = find_key(cose_msg.phdr[KID], DEFAULT_CERTIFICATE_DB_JSON)
                    qr_signed = True
                    if key:
                        verify_signature(cose_msg, key)
                        qr_verified = True
                        print("V databáze bol nájdený príslušný kľúč, certifikát je pravý")
                    else:
                        print("V databáze nebol nájdený príslušný kľúč, certifikát nie je pravý")
                else:
                    print("Certifikát nie je podpísaný")

                cbor = cbor2.loads(cose_msg.payload)
                cbor_json = json.dumps(cbor, indent=2, default=str, ensure_ascii=False)
                # print("Certificate as JSON: {0}".format(cbor_json))
                if 'v' in json.loads(cbor_json)["-260"]["1"]:
                    certificate_date = json.loads(cbor_json)["-260"]["1"]["v"][0]["dt"]
                    vaccine_num = json.loads(cbor_json)["-260"]["1"]["v"][0]["sd"]
                    vaccine_code = json.loads(cbor_json)["-260"]["1"]["v"][0]["vp"]
                    days_since = (datetime.utcnow() - datetime.strptime(certificate_date, "%Y-%m-%d")).days
                    print("Dátum vakcinácie je: " + certificate_date + ", čo je pred " + str(days_since) + " dňami")
                    print("Ide o " + str(vaccine_num) + ". vakcínu v poradí (kód " + str(vaccine_code) + ")")
                    if vaccine_num >= 3:
                        qr_valid = True
                        print("Certifikát je platný bez obmedzenia")
                    else:
                        qr_valid = 14 <= days_since <= 270
                        if qr_valid:
                            print("Rozdiel dní dátumov je v intervale medzi 14 a 270 dní -> certifikát je platný")
                        else:
                            print("Rozdiel dní dátumov nie je v intervale medzi 14 a 270 dní -> certifikát je neplatný")
                elif 'r' in json.loads(cbor_json)["-260"]["1"]:
                    overcome_date = json.loads(cbor_json)["-260"]["1"]["r"][0]["df"]
                    overcome_code = json.loads(cbor_json)["-260"]["1"]["r"][0]["tg"]
                    days_since = (datetime.utcnow() - datetime.strptime(overcome_date, "%Y-%m-%d")).days
                    print("Dátum prekonania je: " + overcome_date + ", čo je pred " + str(days_since) + " dňami")
                    qr_valid = days_since <= 180
                    if qr_valid:
                        print("Rozdiel dní dátumov je do 180 dní -> certifikát je platný")
                    else:
                        print("Rozdiel dní dátumov nie je do 180 dní -> certifikát je neplatný")
                else:
                    print("Neznámy certifikát")
                detected_qr = True
                displaying = True
                timer = threading.Timer(2.0, end_qr_detection)
                timer.start()


def get_scanner_images():
    """Read scanner images"""

    return [cv2.imread('./images/scanner' + str(sc) + '.png') for sc in range(SCANNER_IMAGES_COUNT)] + [
        cv2.imread('./images/scanner' + str(sc) + '.png') for sc in reversed(range(SCANNER_IMAGES_COUNT))]


def get_labels():
    """Read possible labels"""

    return [line.strip() for line in open(PATH_TO_LABELS, 'r')]


global dst
CWD_PATH = os.getcwd()

DEFAULT_CERTIFICATE_DB_JSON = 'certs/Digital_Green_Certificate_Signing_Keys.json'

SCANNER_IMAGES_COUNT = 10
SCANNER_INDEX_SPEED = 0.5

MODEL_NAME = 'TFLite_model/'
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, 'labelmap.txt')

imW, imH = 480, 640

info_image_name = create_info_image()
info_image = cv2.resize(cv2.imread(info_image_name), (640, 480))
dst_image = cv2.resize(cv2.imread('./images/blank_face.png'), (640, 480))
merged = cv2.addWeighted(dst_image, 0.3, info_image, 0.7, 0)
dst_image2 = cv2.resize(cv2.imread('./images/scanner.png'), (640, 480))
merged2 = cv2.addWeighted(dst_image2, 0.3, info_image, 0.7, 0)

# Load loading scanner images
scanner_images = get_scanner_images()
# Load the label map
labels = get_labels()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'detect.tflite')

# Load the Tensorflow Lite model.
interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()

# Get model details (quantization, quantization_parameters, quantized_dimensions)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# 300x300
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# Initialize video stream
videostream = VideoStream(resolution=(imW, imH), framerate=30).start()
time.sleep(1)

object_name = ""
detecting_qr = False
detected_qr = None

qr_signed = False
qr_verified = False
qr_valid = False

frame = None
frame1 = None

go_closer = False
go_left = False
go_right = False

detected_temperature = 0

detected_temps = []
detected_objects = []

displaying = False
displaying_counter = 0

scanner_index = 0

metadata = {
    'mask': {
        True: {
            'name': 'S maskou',
            'title': 'Teplota v poriadku',
            'color_temp': (0, 255, 0),
            'color_mask': (0, 255, 0)
        },
        False: {
            'name': 'S maskou',
            'title': 'Teplota je vysoká',
            'color_temp': (0, 0, 255),
            'color_mask': (0, 255, 0)
        }
    },
    'no mask': {
        True: {
            'name': 'Bez masky',
            'title': 'Teplota v poriadku',
            'color_temp': (0, 255, 0),
            'color_mask': (0, 0, 255)
        },
        False: {
            'name': 'Bez masky',
            'title': 'Teplota je vysoká',
            'color_temp': (0, 0, 255),
            'color_mask': (0, 0, 255)
        }
    }
}

Thread(target=detect_frame).start()
Thread(target=detect_qr_certificate).start()

serial_port = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
serial_port.reset_input_buffer()

Thread(target=detect_temperature).start()

show_video()

cv2.destroyAllWindows()
videostream.stop()
