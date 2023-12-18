import tkinter as tk
import customtkinter
from tkinter import filedialog
from PIL import Image, ImageTk
import zipfile
import rarfile
import cv2

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

import os

import torch.nn.functional as F
import json


ans_dict = {0: "Бетон", 1: "Грунт", 2: "Дерево", 3: "Кирпич", -1: "Неверный файл"}


class CustomRegnet(nn.Module):
    def __init__(self, num_freeze_layers=0):
        super(CustomRegnet, self).__init__()
        self.res = models.regnet.regnet_y_400mf(weights=None)
        self.res.fc = nn.Sequential(nn.Linear(440, 4))

    def forward(self, x):
        return self.res(x)


class Wastes_Classifier(nn.Module):
    def __init__(self):

        super(Wastes_Classifier, self).__init__()

        self.reses = CustomRegnet()
        self.classifier = nn.Linear(12, 4)

    def forward(self, img):
        out = []
        for i in range(3):
            out.append(self.reses(img[:, i, :, :, :]))
        out = torch.cat(out, dim=1)
        out = F.dropout(out, p=0.65, training=self.training)

        return self.classifier(out)

    def give_one_photo(self, file_name):
        trans = transforms.Compose([
            transforms.Resize(448),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        name = file_name
        video_file = []

        start_time = 119  # начальное время (2 минуты)
        end_time = 135  # конечное время (3 минуты)

        # Открываем исходное видеофайл
        video = cv2.VideoCapture(name)

        if int(video.get(cv2.CAP_PROP_FRAME_COUNT)) != 240 * 12:
            print("Video has wrong size!!!", name)
            return torch.Tensor([-1])

        # Вычисляем FPS (кадры в секунду)
        fps = int(video.get(cv2.CAP_PROP_FPS))

        # Вычисляем количество кадров, которые нужно пропустить
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        # Устанавливаем текущий кадр на начальный
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Обрабатываем и отображаем кадры в заданном временном промежутке
        for frame_number in range(start_frame, end_frame):
            ret, frame = video.read()
            if not ret:
                break
            video_file.append(frame / 255)

        # Закрываем видеопоток
        video.release()
        first = 16 * fps - 1
        second = first - fps * 5
        third = second - fps * 10

        video_itg = torch.from_numpy(np.stack([video_file[third], video_file[second], video_file[first]])).permute(0, 3,
                                                                                                                   1, 2)
        video_itg = trans(video_itg)
        vidos = video_itg.to(torch.float32)

        return vidos

    def infer(self, dir_file, is_folder=False, device='cpu'):

        if is_folder:
            data = []
            for name in dir_file:
                photo = self.give_one_photo(name)
                if photo.shape == (3, 3, 224, 224):
                    data.append(self.give_one_photo(name))
            data = torch.stack(data)
        else:
            data = self.give_one_photo(dir_file).unsqueeze(0)
        data = data.to(device)
        device = device

        if device == 'cpu':
            dataset = DataLoader(data, batch_size=4, num_workers=os.cpu_count())
        else:
            print('here')
            dataset = DataLoader(data, batch_size=16)

        answer_itg = []
        for batch in dataset:
            out = []
            for i in range(3):
                out.append(self.reses(batch[:, i, :, :, :]))
            out = torch.cat(out, dim=1)

            answer = self.classifier(out)
            answer_itg.append(torch.argmax(answer, dim=1))

        return torch.cat(answer_itg)


class VideoClassifierApp:
    def __init__(self, root, device="cpu"):
        self.root = root
        self.root.title("Video Classifier App")
        self.list_dict = dict()
        self.json_dict = dict()

        string_primary = "#1a1625"
        sting_surface = "#2f2b3a"
        string_accent = "#7a5af5"
        self.all_selected = False

        # Установка размеров окна по разрешению экрана пользователя
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{int(screen_width * 0.8)}x{int(screen_height * 0.8)}")

        self.root.configure(bg=string_primary)
        if device == 'cuda' and torch.cuda.is_available():
            self.device = device
        else:
            self.device = 'cpu'

        self.model = Wastes_Classifier()

        self.model.load_state_dict(torch.load("model.pt", map_location=torch.device(self.device)))

        self.model.eval()

        # Панель с кнопкой "Файл"
        # self.menu_bar = tk.Menu(self.root)
        # self.root.config(menu=self.menu_bar)

        # self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        # self.menu_bar.add_cascade(label="Файл", menu=self.file_menu)
        # self.file_menu.add_command(label="Загрузить", command=self.load_file)

        self.result_text = tk.Text(self.root, font=("Arial", 14), bg="#575757", fg="white")
        self.result_text.place(relx=0.77, rely=0.05, relwidth=0.2, relheight=0.8)

        # Список загруженных файлов с чекбоксами
        self.file_list = tk.Listbox(self.root, selectmode=tk.MULTIPLE, background="#575757", bd=0, highlightthickness=0,
                                    selectbackground="#717171", selectborderwidth=10, border=20, fg="white")
        self.file_list.place(relx=0.02, rely=0.05, relwidth=0.3, relheight=0.8)
        self.file_list.bind("<<ListboxSelect>>", self.on_file_select)

        # Экран для просмотра первого кадра выбранного видео
        self.video_canvas = tk.Canvas(self.root, bg="#575757", height=580, width=580, bd=0, highlightthickness=0)
        self.video_canvas.place(relx=0.35, rely=0.05)

        my_font = customtkinter.CTkFont(family="Arial", size=14, weight="bold")

        # Кнопка "Выбрать все"
        self.upload_button = customtkinter.CTkButton(self.root, text="Загрузить", width=210, height=40,
                                                     command=self.load_file,
                                                     fg_color=string_accent,
                                                     corner_radius=16,
                                                     text_color="#000", font=my_font)
        self.upload_button.place(relx=0.02, rely=0.9)

        self.select_all_button = customtkinter.CTkButton(self.root, text="Выбрать все", width=210, height=40,
                                                         command=self.select_all,
                                                         fg_color=string_accent,
                                                         corner_radius=16,
                                                         text_color="#000", font=my_font)
        self.select_all_button.place(relx=0.18, rely=0.9)

        self.to_json = customtkinter.CTkButton(self.root, text="Скачать в json", width=300, height=40,
                                               command=self.download,
                                               fg_color=string_accent,
                                               corner_radius=16,
                                               text_color="#000", font=my_font)
        self.to_json.place(relx=0.77, rely=0.9)

        # Кнопка "Предсказать"
        # self.predict_button = tk.Button(self.root, text="Предсказать", command=self.predict, bg=string_primary)
        # self.predict_button.place(relx=0.35, rely=0.9, width=720, relheight=0.05)

        self.predict_button = customtkinter.CTkButton(self.root, width=580,
                                                      height=40, text="Предсказать", command=self.predict,
                                                      fg_color=string_accent,
                                                      corner_radius=16,
                                                      text_color="#000", font=my_font)

        self.predict_button.place(relx=0.35, rely=0.9)

        # Переменная для хранения выбранного видео
        self.selected_video = None


    def load_file(self):
        file_paths = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4"), ("Zip files", "*.zip"), ("Rar files", "*.rar")], multiple=True)
        for file_path in file_paths:
            if file_path:
                # Если это zip файл
                if file_path.lower().endswith('.zip'):
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall()
                        extracted_files = zip_ref.namelist()

                # Если это rar файл
                elif file_path.lower().endswith('.rar'):
                    with rarfile.RarFile(file_path, 'r') as rar_ref:
                        rar_ref.extractall()
                        extracted_files = rar_ref.namelist()

                # Если это mp4 файл
                elif file_path.lower().endswith('.mp4'):
                    extracted_files = [file_path]

                for file in extracted_files:
                    name = file[file.rfind("/") + 1:]
                    self.list_dict[name] = file
                    self.file_list.insert(tk.END, name)

    def on_file_select(self, event):
        selected_index = self.file_list.curselection()
        if selected_index:
            self.selected_video = self.list_dict[self.file_list.get(selected_index)]
            self.display_first_frame()

    def display_first_frame(self):
        if self.selected_video:

            cap = cv2.VideoCapture(self.selected_video)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 12 * 130)
            ret, frame = cap.read()
            cap.release()

            if ret:
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # Apply center crop to the image
                center_crop_size = 580
                transform = transforms.CenterCrop(center_crop_size)
                frame = transform(frame)

                # Display the cropped image
                frame = ImageTk.PhotoImage(frame)

                # Отображение на экране видео

                self.video_canvas.config(width=580, height=580)
                self.video_canvas.create_image(0, 0, anchor=tk.NW, image=frame)
                self.video_canvas.image = frame

    def select_all(self):
        if not self.all_selected:
            self.file_list.selection_set(0, tk.END)
            self.all_selected = True
        else:
            self.file_list.selection_clear(0, tk.END)
            self.all_selected = False

    def download(self):
        file_path = filedialog.asksaveasfilename(defaultextension="predictions.json",
                                                 filetypes=[("JSON files", "*.json")])

        if file_path:
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(self.json_dict, file, indent=4, ensure_ascii=False)
            print(f"JSON file saved to: {file_path}")

    def predict(self):
        selected_files = [self.list_dict[self.file_list.get(i)] for i in self.file_list.curselection()]

        with torch.inference_mode():
            results = []
            predictions = self.model.infer(selected_files, True, self.device)

            for i, prediction in enumerate(predictions):
                prediction = ans_dict[prediction.item()]
                results.append(f"{os.path.basename(selected_files[i])}: {prediction}")

                self.json_dict[os.path.basename(selected_files[i])] = prediction

            # Очищаем Text перед обновлением результатов
            self.result_text.delete(1.0, tk.END)

            # Вставляем новые результаты в Text
            self.result_text.insert(tk.END, "\n".join(results))


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoClassifierApp(root)
    root.mainloop()

