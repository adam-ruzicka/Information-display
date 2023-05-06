import cv2
import os
from datetime import datetime
import csv
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
import requests as requests
import csv_names

days = ["pondelok", "utorok", "streda", "štvrtok", "piatok", "sobota", "nedeľa"]
info_file_name = "./images/covid/" + str(datetime.now().strftime('%Y-%m-%d')) + ".png"
csv_names.create()


def create_info_image() -> str:
    if not os.path.exists(info_file_name):
        with open('names.csv', mode='r') as file:
            names = [row for row in csv.reader(file)]

        day = datetime.now().day - 1
        month = datetime.now().month - 1

        response = requests.get('https://covid.blanksite.eu/.index.php').text
        response_json = json.loads(response)

        def format_number(number):
            return f"{number:,}".replace(',', ' ')

        cases = format_number(response_json['cases'])
        todayCases = format_number(response_json['todayCases'])
        deaths = format_number(response_json['deaths'])
        todayDeaths = format_number(response_json['todayDeaths'])

        def draw_text(text, position, background_color, anchor):
            left, top, right, bottom = draw.textbbox(position, text, anchor=anchor, font=font)
            draw.rectangle((left - 10, top - 10, right + 10, bottom + 10), fill=background_color)
            draw.text(position, text, font=font, anchor=anchor, fill=(255, 255, 255))

        def format_cases(no_cases):
            if no_cases == "1":
                return "Pribudol " + str(no_cases) + " pozitívny"
            elif no_cases == "2" or no_cases == "3" or no_cases == "4":
                return "Pribudli " + str(no_cases) + " pozitívni"
            else:
                return "Pribudlo " + str(no_cases) + " pozitívnych"

        def format_deaths(no_deaths):
            if no_deaths == "1":
                return "Pribudla " + str(no_deaths) + " obeť"
            elif no_deaths == "2" or no_deaths == "3" or no_deaths == "4":
                return "Pribudli " + str(no_deaths) + " obete"
            else:
                return "Pribudlo " + str(no_deaths) + " obetí"

        width, height = (640, 480)
        margin = 20

        pil_im = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(pil_im, "RGBA")
        font = ImageFont.truetype("arial.ttf", width // 25)

        # top left
        draw_text(
            "Aktuálne je " + str(datetime.now().strftime('%d.%m.')) + " (" + str(days[datetime.now().weekday()]) + ")",
            (margin, margin), background_color=(10, 20, 222, 130), anchor="la")
        # top right
        draw_text("Meniny má " + names[month][day], (width - margin, margin), background_color=(228, 20, 222, 130),
                  anchor="ra")

        # bottom left
        pcr_positive = format_cases(todayCases) + "\n(celkovo " + str(cases) + ")"
        draw_text(pcr_positive, (margin, height - margin), background_color=(20, 210, 4, 130), anchor="ld")

        # bottom right
        deaths_positive = format_deaths(todayDeaths) + "\n(celkovo " + str(deaths) + ")"
        draw_text(deaths_positive, (width - margin, height - margin), background_color=(210, 20, 4, 130), anchor="rd")

        info_img = np.array(pil_im)
        red = info_img[:, :, 0].copy()
        info_img[:, :, 0] = info_img[:, :, 2].copy()
        info_img[:, :, 2] = red

        cv2.imwrite(info_file_name, info_img)

    return info_file_name

