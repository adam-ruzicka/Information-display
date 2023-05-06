import csv
import os

names = [
    ["(Nový rok, deň vzniku SR)",
     "Alexandra, Karina",
     "Daniela",
     "Drahoslav",
     "Andrea",
     "Antónia (Zjavenie Pána - Traja králi)",
     "Bohuslava",
     "Severín",
     "Alexej",
     "Dáša",
     "Malvína",
     "Ernest",
     "Rastislav",
     "Radovan",
     "Dobroslav",
     "Kristína",
     "Nataša",
     "Bohdana",
     "Drahomíra, Mário",
     "Dalibor",
     "Vincent",
     "Zora",
     "Miloš",
     "Timotej",
     "Gejza",
     "Tamara",
     "Bohuš",
     "Alfonz",
     "Gašpar",
     "Ema",
     "Emil"],
    ["Tatiana",
     "Erik, Erika",
     "Blažej",
     "Veronika",
     "Agáta",
     "Dorota",
     "Vanda",
     "Zoja",
     "Zdenko",
     "Gabriela",
     "Dezider",
     "Perla",
     "Arpád",
     "Valentín",
     "Pravoslav",
     "Ida, Liana",
     "Miloslava",
     "Jaromír",
     "Vlasta",
     "Lívia",
     "Eleonóra",
     "Etela",
     "Roman, Romana",
     "Matej",
     "Frederik, Frederika",
     "Viktor",
     "Alexander",
     "Zlatica",
     "Radomír"],
    ["Albín",
     "Anežka",
     "Bohumil, Bohumila",
     "Kazimír",
     "Fridrich",
     "Radoslav, Radoslava",
     "Tomáš",
     "Alan, Alana",
     "Františka",
     "Branislav, Bruno",
     "Angela, Angelika",
     "Gregor",
     "Vlastimil",
     "Matilda",
     "Svetlana",
     "Boleslav",
     "Ľubica",
     "Eduard",
     "Jozef",
     "Víťazoslav, Klaudius",
     "Blahoslav",
     "Beňadik",
     "Adrián",
     "Gabriel",
     "Marián",
     "Emanuel",
     "Alena",
     "Soňa",
     "Miroslav",
     "Vieroslava",
     "Benjamín"],
    ["Hugo",
     "Zita (Veľký piatok)",
     "Richard",
     "Izidor",
     "Miroslava (Veľkonočný pondelok)",
     "Irena",
     "Zoltán",
     "Albert",
     "Milena",
     "Igor",
     "Július",
     "Estera",
     "Aleš",
     "Justína",
     "Fedor",
     "Dana, Danica",
     "Rudolf, Rudolfa",
     "Valér",
     "Jela",
     "Marcel",
     "Ervín",
     "Slavomír",
     "Vojtech",
     "Juraj",
     "Marek",
     "Jaroslava",
     "Jaroslav",
     "Jarmila",
     "Lea",
     "Anastázia"],
    ["(Sviatok práce)",
     "Žigmund",
     "Galina, Timea",
     "Florián",
     "Lesana, Lesia",
     "Hermína",
     "Monika",
     "Ingrida (Deň víťazstva nad fašizmom)",
     "Roland",
     "Viktória",
     "Blažena",
     "Pankrác",
     "Servác",
     "Bonifác",
     "Žofia, Sofia",
     "Svetozár",
     "Gizela, Aneta",
     "Viola",
     "Gertrúda",
     "Bernard",
     "Zina",
     "Júlia, Juliána",
     "Želmíra",
     "Ela",
     "Urban, Vivien",
     "Dušan",
     "Iveta",
     "Viliam",
     "Vilma, Elmar, Maxim, Maxima",
     "Ferdinand",
     "Petrana, Petronela"],
    ["Žaneta",
     "Xénia, Oxana",
     "Karolína",
     "Lenka",
     "Laura",
     "Norbert",
     "Róbert, Róberta",
     "Medard",
     "Stanislava",
     "Margaréta, Gréta",
     "Dobroslava",
     "Zlatko",
     "Anton",
     "Vasil",
     "Vít",
     "Bianka, Blanka",
     "Adolf",
     "Vratislav",
     "Alfréd",
     "Valéria",
     "Alojz",
     "Paulína",
     "Sidónia",
     "Ján",
     "Tadeáš, Olívia",
     "Adriána",
     "Ladislav, Ladislava",
     "Beáta",
     "Peter, Pavol, Petra",
     "Melánia"],
    ["Diana",
     "Berta",
     "Miloslav",
     "Prokop",
     "Cyril, Metod (Sviatok)",
     "Patrik, Patrícia",
     "Oliver",
     "Ivan",
     "Lujza",
     "Amália",
     "Milota",
     "Nina",
     "Margita",
     "Kamil",
     "Henrich",
     "Drahomír, Rút",
     "Bohuslav",
     "Kamila",
     "Dušana",
     "Eliáš, Iľja",
     "Daniel",
     "Magdaléna",
     "Oľga",
     "Vladimír",
     "Jakub, Timur",
     "Anna, Hana, Anita",
     "Božena",
     "Krištof",
     "Marta",
     "Libuša",
     "Ignác"],
    ["Božidara",
     "Gustáv",
     "Jerguš",
     "Dominik, Dominika",
     "Hortenzia",
     "Jozefína",
     "Štefánia",
     "Oskar",
     "Ľubomíra",
     "Vavrinec",
     "Zuzana",
     "Darina",
     "Ľubomír",
     "Mojmír",
     "Marcela",
     "Leonard",
     "Milica",
     "Elena, Helena",
     "Lýdia",
     "Anabela, Liliana",
     "Jana",
     "Tichomír",
     "Filip",
     "Bartolomej",
     "Ľudovít",
     "Samuel",
     "Silvia",
     "Augustín",
     "Nikola, Nikolaj (Výročie SNP)",
     "Ružena",
     "Nora"],
    ["Drahoslava (Deň ústavy SR)",
     "Linda, Rebeka",
     "Belo",
     "Rozália",
     "Regína",
     "Alica",
     "Marianna",
     "Miriama",
     "Martina",
     "Oleg",
     "Bystrík",
     "Mária, Marlena",
     "Ctibor",
     "Ľudomil",
     "Jolana (Sedembolestná Panna Mária)",
     "Ľudmila",
     "Olympia",
     "Eugénia",
     "Konštantín",
     "Ľuboslav, Ľuboslava",
     "Matúš",
     "Móric",
     "Zdenka",
     "Ľubor, Ľuboš",
     "Vladislav, Vladislava",
     "Edita",
     "Cyprián",
     "Václav",
     "Michal, Michaela",
     "Jarolím"],
    ["Arnold",
     "Levoslav",
     "Stela",
     "František",
     "Viera",
     "Natália",
     "Eliška",
     "Brigita",
     "Dionýz",
     "Slavomíra",
     "Valentína",
     "Maximilián",
     "Koloman",
     "Boris",
     "Terézia",
     "Vladimíra",
     "Hedviga",
     "Lukáš",
     "Kristián",
     "Vendelín",
     "Uršuľa",
     "Sergej",
     "Alojzia",
     "Kvetoslava",
     "Aurel",
     "Demeter",
     "Sabína",
     "Dobromila",
     "Klára",
     "Šimon, Simona",
     "Aurélia"],
    ["Denis, Denisa (Sviatok všetkých svätých)",
     "Pamiatka zosnulých",
     "Hubert",
     "Karol",
     "Imrich",
     "Renáta",
     "René",
     "Bohumír",
     "Teodor",
     "Tibor",
     "Martin, Maroš",
     "Svätopluk",
     "Stanislav",
     "Irma",
     "Leopold",
     "Agnesa",
     "Klaudia (Deň boja za slobodu a demokraciu)",
     "Eugen",
     "Alžbeta",
     "Félix",
     "Elvíra",
     "Cecília",
     "Klement",
     "Emília",
     "Katarína",
     "Kornel",
     "Milan",
     "Henrieta",
     "Vratko",
     "Andrej, Ondrej"],
    ["Edmund",
     "Bibiána",
     "Oldrich",
     "Barbora, Barbara",
     "Oto",
     "Mikuláš",
     "Ambróz",
     "Marína",
     "Izabela",
     "Radúz",
     "Hilda",
     "Otília",
     "Lucia",
     "Branislava, Bronislava",
     "Ivica",
     "Albína",
     "Kornélia",
     "Sláva",
     "Judita",
     "Dagmara",
     "Bohdan",
     "Adela",
     "Nadežda",
     "Adam, Eva (Štedrý deň)",
     "1. sviatok vianočný",
     "Štefan (2. sviatok vianočný)",
     "Filoména",
     "Ivana, Ivona",
     "Milada",
     "Dávid",
     "Silvester"]
]


def create():
    if not os.path.exists('names.csv'):
        with open('names.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(names)
