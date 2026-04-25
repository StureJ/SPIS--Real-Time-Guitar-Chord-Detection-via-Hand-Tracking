import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import glob

# --- Config ---
SAVE_FOLDER = "captures"
FILE_PREFIX = "photo"

os.makedirs(SAVE_FOLDER, exist_ok=True)


def get_next_index():
    existing = glob.glob(os.path.join(SAVE_FOLDER, f"{FILE_PREFIX}_*.jpg"))
    if not existing:
        return 1
    indices = []
    for f in existing:
        try:
            num = int(os.path.splitext(os.path.basename(f))[0].split("_")[-1])
            indices.append(num)
        except ValueError:
            pass
    return max(indices) + 1 if indices else 1


class WebcamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("📸 Webcam Capture")
        self.root.configure(bg="#1a1a1a")
        self.root.resizable(False, False)

        #Camera index set to 1 before else it connects to iPhone.
        self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam.")
            self.root.destroy()
            return

        # Flip
        self.mirror = True
        self.label = tk.Label(root, bg="#1a1a1a", bd=0)
        self.label.pack(padx=10, pady=10)

        btn_frame = tk.Frame(root, bg="#1a1a1a")
        btn_frame.pack(pady=(0, 12))

        self.capture_btn = tk.Button(
            btn_frame,
            text="📷  Take Photo",
            command=self.capture,
            font=("Helvetica", 14, "bold"),
            bg="#e63946",
            fg="white",
            activebackground="#c1121f",
            activeforeground="white",
            relief="flat",
            padx=20,
            pady=8,
            cursor="hand2",
        )
        self.capture_btn.pack(side="left", padx=6)

        self.mirror_btn = tk.Button(
            btn_frame,
            text="🔄  Mirror: ON",
            command=self.toggle_mirror,
            font=("Helvetica", 11),
            bg="#333",
            fg="white",
            activebackground="#555",
            activeforeground="white",
            relief="flat",
            padx=12,
            pady=8,
            cursor="hand2",
        )
        self.mirror_btn.pack(side="left", padx=6)

        self.status = tk.Label(
            root,
            text=f"Saves to: ./{SAVE_FOLDER}/",
            bg="#1a1a1a",
            fg="#888",
            font=("Helvetica", 10),
        )
        self.status.pack(pady=(0, 8))

        self.update_frame()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            if self.mirror:
                frame = cv2.flip(frame, 1)
            self.current_frame = frame  # Save for capture
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img = img.resize((640, 480), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)
        self.root.after(15, self.update_frame)

    def capture(self):
        if not hasattr(self, "current_frame"):
            return
        idx = get_next_index()
        filename = os.path.join(SAVE_FOLDER, f"{FILE_PREFIX}_{idx}.jpg")
        cv2.imwrite(filename, self.current_frame)
        self.status.config(text=f"✅  Saved: {filename}", fg="#4caf50")
        self.root.after(3000, lambda: self.status.config(
            text=f"Saves to: ./{SAVE_FOLDER}/", fg="#888"
        ))

    def toggle_mirror(self):
        self.mirror = not self.mirror
        label = "ON" if self.mirror else "OFF"
        self.mirror_btn.config(text=f"🔄  Mirror: {label}")

    def on_close(self):
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = WebcamApp(root)
    root.mainloop()