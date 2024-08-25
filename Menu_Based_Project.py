import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk, ImageDraw, ImageFilter, ImageEnhance
import numpy as np
import pyttsx3
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from googlesearch import search
import geocoder
from twilio.rest import Client

# Initialize the text-to-speech engine
engine = pyttsx3.init()

def capture_image():
    filename = 'captured_image.jpg'
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not access the webcam.")
        return
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(filename, frame)
        messagebox.showinfo("Success", f"Image saved as {filename}")
    else:
        messagebox.showerror("Error", "Could not capture an image.")
    cap.release()
    cv2.destroyAllWindows()

def detect_face(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    return (x, y, w, h)

def overlay_face_on_image(original_image_path, face_image_path, face_bbox):
    original_image = Image.open(original_image_path)
    face_image = Image.open(face_image_path)
    face_image = face_image.resize((face_bbox[2], face_bbox[3]))
    overlay = Image.new('RGBA', original_image.size, (0, 0, 0, 0))
    overlay.paste(face_image, (face_bbox[0], face_bbox[1]))
    combined_image = Image.alpha_composite(original_image.convert('RGBA'), overlay)
    combined_image = combined_image.convert('RGB')
    combined_image.save('overlayed_face_image.jpg')
    combined_image.show()

def process_image():
    capture_image()
    face_bbox = detect_face('captured_image.jpg')
    if face_bbox:
        image = Image.open('captured_image.jpg')
        cropped_face = image.crop((face_bbox[0], face_bbox[1], face_bbox[0] + face_bbox[2], face_bbox[1] + face_bbox[3]))
        cropped_face.save('cropped_face.jpg')
        overlay_face_on_image('captured_image.jpg', 'cropped_face.jpg', face_bbox)
        messagebox.showinfo("Success", "Face detected and overlay completed.")

def text_to_speech():
    text = "Heyy, this is Rohan"
    print(text)
    engine.say(text)
    engine.runAndWait()

def set_volume():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume.iid, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volume.SetMasterVolumeLevel(-20.0, None)
    messagebox.showinfo("Volume", "Volume set to -20 dB.")

def google_search():
    query = "top singers"
    results = search(query, num_results=5)
    result_text = "\n".join(f"{i + 1}. {result}" for i, result in enumerate(results))
    messagebox.showinfo("Google Search Results", result_text)

def get_location():
    g = geocoder.ip('me')
    location_info = f"Latitude: {g.lat}\nLongitude: {g.lng}\nAddress: {g.address}"
    messagebox.showinfo("Location", location_info)

def apply_filters():
    image_path = filedialog.askopenfilename()
    if not image_path:
        return
    image = Image.open(image_path)
    blurred_image = image.filter(ImageFilter.BLUR)
    sharpened_image = image.filter(ImageFilter.SHARPEN)
    edge_enhanced_image = image.filter(ImageFilter.EDGE_ENHANCE)
    grayscale_image = image.convert("L")
    enhancer = ImageEnhance.Brightness(image)
    brightened_image = enhancer.enhance(1.5)
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.imshow(image)
    plt.title("Original")
    plt.axis('off')
    plt.subplot(2, 3, 2)
    plt.imshow(blurred_image)
    plt.title("Blurred")
    plt.axis('off')
    plt.subplot(2, 3, 3)
    plt.imshow(sharpened_image)
    plt.title("Sharpened")
    plt.axis('off')
    plt.subplot(2, 3, 4)
    plt.imshow(edge_enhanced_image)
    plt.title("Edge Enhanced")
    plt.axis('off')
    plt.subplot(2, 3, 5)
    plt.imshow(grayscale_image, cmap='gray')
    plt.title("Grayscale")
    plt.axis('off')
    plt.subplot(2, 3, 6)
    plt.imshow(brightened_image)
    plt.title("Brightened")
    plt.axis('off')
    plt.show()

def display_color_image():
    color_image = np.zeros((100, 100, 3), dtype=np.uint8)
    color_image[40:60, 40:60, 0] = 255
    plt.imshow(color_image)
    plt.axis('off')
    plt.show()

def send_sms():
    account_sid = 'AC32d3821fd8dbab5ee714c9f5dd364901'
    auth_token = 'b246756cb4b4df713d3ff6751c75f863'
    from_phone = '+16183074236'
    client = Client(account_sid, auth_token)
    message = client.messages.create(
        body='Hello, this is a test message sent from Python!',
        from_=from_phone,
        to='+918504855535'
    )
    messagebox.showinfo("SMS Sent", f"Message sent with SID: {message.sid}")

def make_call():
    account_sid = 'AC32d3821fd8dbab5ee714c9f5dd364901'
    auth_token = 'b246756cb4b4df713d3ff6751c75f863'
    from_phone_number = '+16183074236'
    to_phone_number = '+918504855535'
    twiml_url = 'http://demo.twilio.com/docs/voice.xml'
    client = Client(account_sid, auth_token)
    call = client.calls.create(
        to=to_phone_number,
        from_=from_phone_number,
        url=twiml_url
    )
    messagebox.showinfo("Call Initiated", f"Call SID: {call.sid}")

def send_whatsapp_message():
    account_sid = 'AC32d3821fd8dbab5ee714c9f5dd364901'
    auth_token = 'b246756cb4b4df713d3ff6751c75f863'
    from_phone = 'whatsapp:+14155238886'  # Twilio Sandbox WhatsApp number
    to_phone = 'whatsapp:+918504855535'
    client = Client(account_sid, auth_token)
    message = client.messages.create(
        body='Hello Rohan, this is a test message from Python!',
        from_=from_phone,
        to=to_phone
    )
    messagebox.showinfo("WhatsApp Message Sent", f"Message sent with SID: {message.sid}")

def apply_realistic_sunglasses_filter():
    image_path = 'captured_image.jpg'
    image = Image.open(image_path).convert('RGBA')
    width, height = image.size
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    glasses_width = width // 3
    glasses_height = height // 10
    bridge_width = width // 10
    bridge_height = glasses_height // 2
    left_eye_center = (width // 3, height // 3)
    right_eye_center = (2 * width // 3, height // 3)
    bridge_center = (width // 2, height // 3 + glasses_height // 2)
    left_bbox = [left_eye_center[0] - glasses_width // 2, left_eye_center[1] - glasses_height // 2,
                 left_eye_center[0] + glasses_width // 2, left_eye_center[1] + glasses_height // 2]
    draw.ellipse(left_bbox, fill=(0, 0, 0, 128))
    right_bbox = [right_eye_center[0] - glasses_width // 2, right_eye_center[1] - glasses_height // 2,
                  right_eye_center[0] + glasses_width // 2, right_eye_center[1] + glasses_height // 2]
    draw.ellipse(right_bbox, fill=(0, 0, 0, 128))
    bridge_bbox = [bridge_center[0] - bridge_width // 2, bridge_center[1] - bridge_height // 2,
                   bridge_center[0] + bridge_width // 2, bridge_center[1] + bridge_height // 2]
    draw.rectangle(bridge_bbox, fill=(0, 0, 0, 128))
    combined_image = Image.alpha_composite(image, overlay)
    combined_image.save('sunglasses_filter_image.jpg')
    combined_image.show()

def create_main_window():
    window = tk.Tk()
    window.title("Application")
    window.geometry("800x600")

    # Add welcome message and name at the top
    tk.Label(window, text="ROHAN SINGH CHOUHAN", font=('Roboto', 16, 'bold'), bg='white').pack(pady=10)
    tk.Label(window, text="Welcome to my project!", font=('Roboto', 14), bg='white').pack(pady=5)

    # Set a solid background color
    window.configure(bg="#f0f0f0")

    # Create and place the buttons with a dark green background
    button_color = "#006400"  # Dark Green

    tk.Button(window, text="Capture Image", command=capture_image, bg=button_color, fg="white").pack(pady=10)
    tk.Button(window, text="Process Image", command=process_image, bg=button_color, fg="white").pack(pady=10)
    tk.Button(window, text="Text to Speech", command=text_to_speech, bg=button_color, fg="white").pack(pady=10)
    tk.Button(window, text="Set Volume", command=set_volume, bg=button_color, fg="white").pack(pady=10)
    tk.Button(window, text="Google Search", command=google_search, bg=button_color, fg="white").pack(pady=10)
    tk.Button(window, text="Get Location", command=get_location, bg=button_color, fg="white").pack(pady=10)
    tk.Button(window, text="Apply Filters", command=apply_filters, bg=button_color, fg="white").pack(pady=10)
    tk.Button(window, text="Display Color Image", command=display_color_image, bg=button_color, fg="white").pack(pady=10)
    tk.Button(window, text="Send SMS", command=send_sms, bg=button_color, fg="white").pack(pady=10)
    tk.Button(window, text="Make Call", command=make_call, bg=button_color, fg="white").pack(pady=10)
    tk.Button(window, text="Send WhatsApp Message", command=send_whatsapp_message, bg=button_color, fg="white").pack(pady=10)
    tk.Button(window, text="Apply Sunglasses Filter", command=apply_realistic_sunglasses_filter, bg=button_color, fg="white").pack(pady=10)

    window.mainloop()

create_main_window()
