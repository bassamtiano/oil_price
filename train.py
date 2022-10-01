import pandas as pd

class helo():
    def test(self):
        print('hello')

def load_data():
    bulan2id = {
        "Januari": 1,
        "Februari": 2,
        "Maret": 3,
        "April": 4,
        "Mei": 5,
        "Juni": 6,
        "Juli": 7,
        "Agustus": 8,
        "September": 9,
        "Oktober": 10,
        "November": 11,
        "Desember": 12
    }

    data = pd.read_csv('./datasets/bps_harga_beras.csv', sep =';')
    # Mengambil nama kolom di dataset
    print(data.keys())
    # Mengambil nilai / value di kolom tertentu
    beras_data = data[['nama_tahun', 'nama_turunan_tahun', 'data_content']]

    beras_data.nama_turunan_tahun =  beras_data.nama_turunan_tahun.map(bulan2id)
    # Untuk load file excel
    beras_2021 = pd.read_excel('./datasets/data_beras_2021.xlsx')
    # untuk transpose / putar dari horizontal ke vertikal bentuk table nya
    beras_2021_raw = beras_2021[:5].T
    beras_2021_raw = beras_2021_raw[1:13]
    beras_2021_raw = beras_2021_raw.loc[:, 1:4]
    
    beras_2021_raw = beras_2021_raw.rename(columns={1: "bulan", 2: "premium", 3: "medium", 4: "luar_kualitas"})
    beras_2021_raw["premium"] = pd.to_numeric(beras_2021_raw["premium"])
    beras_2021_raw["medium"] = pd.to_numeric(beras_2021_raw["medium"])
    beras_2021_raw["luar_kualitas"] = pd.to_numeric(beras_2021_raw["luar_kualitas"])
    beras_2021_raw["tahun"] = "2021"
    # Mean dengan axis 1 dia rata rata horizontal  (kiri ke kanan)
    # Mean dengan axis 0 dia rata rata vertical (atas ke bawah)
    beras_2021_raw["harga_beras"] = beras_2021_raw[["premium", "medium", "luar_kualitas"]].mean(axis=1).round()
    beras_2021_raw = beras_2021_raw[["tahun", "bulan", "harga_beras"]].reset_index(drop=True)
    beras_2021_raw.bulan =  beras_2021_raw.bulan.map(bulan2id)
    print(beras_2021_raw)


if __name__ == '__main__':
    # Memanggil method di class
    h = helo()
    h.test()

    # Memanggil method
    load_data()

