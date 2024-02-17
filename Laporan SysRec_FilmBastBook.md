# Laporan Proyek Machine Learning -  Rekomendasi Film
by Handerson Loriano

## Project Overview
### Pendahuluan
[<sup>[1]</sup>](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9269752/)Sistem rekomendasi Film bertujuan untuk memberikan sugesti kepada *customer* berdasarkan fitur-fitur yang disukai pada *Customer*. Sistem rekomendasi yang terbaik adalah mampu memberikan sugesti film yang cocok dengan kesamaan *Customer* dalam performa yang tinggi.

Seiringan dengan perkembangan teknologi yang pesat, beberapa pelaku bisnis perusahaan ditantang untuk memberikan rekomendasi cepat dan relevan. Untuk itu sistem rekomendasi dibutuhkan untuk memecahkan masalah yang kompleks karena, penyesuaian dari masing-masing pihak pengguna berbeda-beda, waktu, lingkungan dan juga kebudayaannya masing-masing.

Contohnya, Sistem rekomendasi *youtube* (Nama popularnya '*Youtube Algorithm*') merekomendasikan pengunjung yang berada disuatu tempat

![image](https://drive.google.com/uc?id=1Le8PvVE9AHgEdbP1xiCPprB5m64VS6Rj&export=download)

Hal ini menyebabkan pengguna ingin melihat bagian yang ingin dieksplorasi dan meningkatkan peluang untuk video yang banyak ingin dilihat. Tentunya ini akan menambah *revenue* bagi pemilik video dan *Channel*-nya

### Rumusan Permasalahan Awal
1. Mempengaruhi seseorang dengan kesamaan relevansi dalam waktu singkat
2. Analisa dari sudut pandang *customer* sangatlah sulit :
  - *Customer* memiliki pandangan yang berbeda-beda dan data yang sangat besar
  - *Customer* tidak semua data bisa menjadi relevan untuk diolah sistem rekomendasi karena, beberapa data-data yang dimiliki *customer* tidak bisa dijadikan acuan yang tepat.

## Business Understanding
Bila ingin merekomendasi suatu film tentunya harus ada kedekatan konten yang ingin dipertunjukan dari segi manapun. Misalnya, bila seseorang menikmati film bergenre '*Animation*' tentunya ia akan memilih film '*Animation*' yang lainnya. Tentu saja penikmat film bisa saja berubah, bisa saja ia ingin melihat film tentang 'Documentary'. Akan tetapi, rekomendasi harus mendekati diantara '*Animation*' dan '*Documentary*' maka, penikmat film akan selalu menikmatinya

### Problem Statements
1. *Genre* apa yang memiliki rating tertinggi?
2. Apa rekomendasi film untuk orang yang ber*genre* *Horror*?

### Goals
1. Mengetahui '*Genre*' yang mempunyai rating tertinggi
2. Membuat model *machine learning Content-Based filtering* yang dapat memprediksi film rekomendasi dan diukur berdasarkan metrik *Precision @K*.
Semakin banyak *genre* yang mempunyai kesamaan pada film maka semakin besar peluang penikmat film akan menonton film ber*genre* kesukaannya.

Pada contoh ini adalah film ber*genre* horror

## Data Understanding
Ini adalah tahapan yang dilakukan pada *Data Understanding* :
1. Melakukan *Data Loading* supaya bisa membaca sumber dari yang sudah ada dan mengikuti arti data yang baik
2. Mendeskripsikan data supaya bisa melihat pola apa saja yang bisa disiapkan pada tahapan *Data Preparation*
3. Menyaring *missing Value* dan *Outliers*
4. Melakukan *EDA Univariate analysis* supaya dapat melihat total data yang dapat lihat
5. Melakukan *EDA Multivariate analysis* untuk fitur kategorikal untuk mencari hubungan yang sesuai

Info Dataset :
1. Nama Dataset : NetflixOriginals
2. total : 584

Berikut ini adalah maksud / arti dari nama kolom tersebut :
1. *Title* = judul film
2. *Genre* = jenis film
3. *Premiere* = tanggal film tayangan
4. *Runtime* = lamanya film berjalan (dalam menit)
5. *IMDB Score* = nilai rating film dari IMDB (0 - 10)
6. *Language* = bahasa film tayangan

proses yang akan dilakukan pada tahapan ini adalah :
1. *Data loading*
2. *Exploratory Data Analysis*

Pada penggalian data pada tahap *Exploratory Data analysis* :
1. Menemukan Kolom cell *Language* yang sepertinya berganda dan beberapa dari bahasa tersebut ditemukan menyatu dengan satu sama lain. Data unik ini bisa dinamakan *multilabel classification*, yang dimana beberapa data menjadi satu tapi masih terkaitan dengan maknanya. Sehingga bisa terlihat pada perhitungan *univariate analysis* lumayan banyak dan komplex
![image](https://drive.google.com/uc?id=1e7pjEL1LZzxQECHOmojRANT2MkpzD5Zq&export=download)
2. Uniknya pada data ini memiliki korelasi lemah terhadap '*Runtime*' dan '*Premiere*'. Ini berarti, penayangan tanggal film-film yang telat berpotensi mempunyai film dengan jangka tayang yang lebih lama dari tanggal-tanggal yang sebelumnya
![image](https://drive.google.com/uc?id=1BJGTs-eSAWm1YU-zzoAqojffPlC4TwI4&export=download)

### Data Loading
Data yang berasal dari sumber ini[<sup>[6]</sup>](https://www.kaggle.com/datasets/luiscorter/netflix-original-films-imdb-scores) :
1. setelah diunduh, harus diunggah ke *google drive*
2. dan mengubah akses supaya bisa pakai di *google colab*

Pada tahap data loading akan melakukan pengambilan data dan menyeleksi kolom yang perlu diperhatikan

### Exploratory Data Analysis
Ini adalah tahapan yang akan dilakukan :
1. Mendeskripsikan variabel kolom
2. menganalisis untuk mencari *missing value* dan *outliers*
3. Menggunakan cara *univariate analysis* untuk analisis data dan menjawab pertanyaan bisnis

*Exploratory data analysis* dapat membantu mencari *outliers* dan menjawab pertanyaan pengertian bisnis yang disampaikan.
Tahapan yang dilakukan oleh *Exploratory Data Analysis* :
1. Melihat *missing value*. Tidak ada *missing value* yang relevan
2. Mengubah data tanggal menjadi angka yang bisa diproses untuk pengecekan *outliers*
3.Memproses data *numerical* yang berada di jangkauan *outlier* memakai, metode *IQR*. Untuk memproses *value* yang dapat mengacaukan pada tahap *modelling*. Contohnya, kolom '*Premiere*' dan '*Runtime*'

Setelah itu melakukan *Exploratory Data Analysis - Univariate Analysis*. Dari sini bisa memproses data kategorikal.

Terlihat ada beberapa data pada kolom '*Language*' terlihat ganda dan beberapa *variable* dari kolom tersebut sama dengan beberapa *variable* yang sendiri.

Contohnya :
![image](https://drive.google.com/uc?id=1AiF5jdMCpBUcNhOpjnYGlpDSPD5bSaqK&export=download)

Kategori yang di kotak merah adalah *Outlier Categorical*. Kesalahan pada '*Making-of*' ini adalah tidak tahu apakah ini jenis film bertipe '*Documentary*', '*Behind the Screen*' atau sebagainya. Maka, dari itu dihapus.

lalu :
![image](https://drive.google.com/uc?id=1RWnL0YI-GzT0ojw6DBbyVq-CU-JfOjHb&export=download)

Pada gambar ini menunjukan, adanya kesalahan pada pengetikan (*typo*). Karenanya, bisa diperbaiki. Maka bisa, diganti data tersebut dari 'Thia' menjadi 'Thai'.

Selanjutnya, ini adalah pertanyaan bisnis yang dapat dijawab dalam analisanya.

|Data|Value|
|:------|:------|
|Title|David Attenborough : ALife on Our Planet|
|Genre|Documentary|
|Premiere|1601769600000000000|
|Runtime|83|
|IMDB Score|9.0|
|Language|English|

Terlihat genre '*Documentary*' mempunyai *IMDB Score* yang paling tinggi yang berjudul '*David Attenborough: A Life on Our Planet*'

## Data Preparation
Beberapa data sudah diproses untuk *missing value* dan *outliers* pada tahap *Exploratory data analysis* :
1. Menggunakan Metode *IQR* untuk mengatasi *outliers* pada kolom *Runtime* dan *Premiere*.
2. Tidak ada *Missing Value* atau data yang kosong. Setelah, menggunakan '*.isnull().sum()*' pada library pandas.
3. Mengubah kolom *Premiere* menjadi tanggal dan diolah memakai *value* untuk dianalisa pada tahapan *Data Understanding*
4.  Lalu, ini adalah tahapan yang digunakan pada *Data Preparation*</br>
  Membersihkan data yang sudah dianalisa pada tahapan *Exploratory Data Analysis*
    - Kesalahan pada '*Making-of*'. Tidak tahu apakah ini jenis film bertipe '*Documentary*', '*Behind the Screen*' atau sebagainya. Maka, dari itu data boleh dihapus
    - Kesalahan penulisan 'Thia'. Tidak ada bahasa yang bernama Thia. Jadi harus diubah dan bisa menganalisa memakai *google* menjadi '*Thai*'

## Modelling
Sistem rekomendasi akan menampilkan 10 hal yang direkomendasikan dan menggunakan model Pendekatan *Content Based Filtering*.

Kolom yang digunakan adalah *Genre* karena, Ini adalah hal yang membutuhkan kesesuaian penikmat film. Bila mereka ingin melihat film layaknya bahasa inggris, bisa mengatur nilai k didalam fungsi

Dikutip dari halaman page *SKLearn* tentang [*TfidfVectorizer*](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) adalah Mengkonversikan kumpulan kata-kata dan dijadikan *matrix of TF-IDF features* (Fitur ini adalah fitur yang paling tepat karena, pada Kolom '*Genre*' juga mempunyai *multilabel classification*). Dengan ini, bisa dicek kesamaan memakai [*Cosine Similarity*](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html) pada sampel matriks x dan y -nya

Ini adalah tahap yang dibuat :
1. Meng*initilize* dan Memakai *TFIDF-Vectorizer* untuk melihat kata-kata yang dikategorikan
2. Menggunakan *Cosine Similarity* untuk menghitung derajat Kesamaan-nya
3. Membuat fungsi rekomendasi dari dataset pada *Data Preparation*

Pada perhitungan *TFIDF-Vectorizer* memakai perhitungan *default*nya dan *Cosine Similarity* juga memakai perhitungan *default*nya saja.

### Hasil
Inilah Hasilnya bila seseorang pernah melihat film "*The Larva Island Movie*"

|index|Title|Genre|Language|Runtime|Premiere|IMDB Score|
|---|---|---|---|---|---|---|
|0|Octonauts & the Caves of Sac Actun|Animation|English|72|14/Aug 2020|6\.2|
|1|The Willoughbys|Animation/Comedy/Adventure|English|90|22/Apr 2020|6\.4|
|2|Fearless|Animation/Superhero|English|89|14/Aug 2020|4\.9|
|3|Invader Zim: Enter the Florpus|Animation/Science Fiction|English|71|16/Aug 2019|7\.5|
|4|Over the Moon|Animation/Musical/Adventure|English|95|23/Oct 2020|6\.4|
|5|Klaus|Animation/Christmas/Comedy/Adventure|English|97|15/Nov 2019|8\.2|
|6|The Babysitter: Killer Queen|Comedy/Horror|English|102|10/Sep 2020|5\.8|
|7|The After Party|Comedy|English|89|24/Aug 2018|5\.8|
|8|Rising High|Satire|German|94|17/Apr 2020|5\.8|
|9|The Claus Family|Fantasy|Dutch|96|07/Dec 2020|5\.8|

## Evaluation
Ini adalah tahapan yang akan dilakukan oleh evaluation :
1. Membuat metrik evaluasi berdasarkan *Precision @K* [<sup>[3]</sup>](https://datascience.stackexchange.com/questions/92247/precisionk-and-recallk)

  > $\texttt{Recommender system precision: }P = \frac{\texttt{# of our recommendations that are relevant}}{\texttt{# of items we recommend}}$

  Sederhananya, Fungsi ini menghitung berapa jumlah rekomendasi yang relevan dibagi dengan semua rekomendasi. $P$ adalah persentase yang dimana semakin mendekati 100% adalah mesin rekomendasi yang terbaik.

  Rekomendasi Persen = jumlah total rekomendasi yang relevan / jumlah semua item rekomendasi
2. Membuat perbandingan antar *genre* dan persentase yang tepat

Perhitungan ini efektif mengevaluasi model *machine-learning* yang memakai lebih dari satu algoritma *machine-learning*. Akan tetapi, evaluasi dapat membias kearah sistem rekomendasi pada salah satu *Genre*. Maka, bisa disempurnakan memakai perhitungan beberapa *Genre*

Terlihat bila data pada genre yang mempunyai dataset sedikit perhitungan persentase presisi menurun ini dikarenakan data yang digunakan sangat kurang.

Tentunya Perhitungan *Precision @K* ini masih membias ke konten yang terbanyak oleh karena itu bisa dihitung menggunakan rata-rata dari beberapa *genre* :

|Genre|Persentase|
|:---|---:|
|Animation|60%|
|Documentary|100%|
|Horror|100%|
|Musical|60%|

lalu perhitungannya :
> $\\P = \frac{\texttt{60% + 100% + 100% + 60%}}{4}$

> $\\P = \texttt{80%}$

Terlihat pada Genre '*Documentary*' dan '*Horror*' Memberikan evaluasi yang 100% karena, dataset yang dilatih banyak dari *Genre* '*Documentary*' dan '*Horror*'. Sedangkan, pada *genre* *Animation* dan *Musical* memberikan evaluasi 60%. Maka untuk itu, dipakailah perhitungan rata-rata antara banyak *genre* sehingga perhitungan tersebut mengurangi bias dalam evaluasi '*Precision @K*'

Hasilnya evaluasi sistem rekomendasinya adalah $\pm$ 80%. Hasil yang cukup bagus dan relevan.

Inilah rekomendasi seseorang untuk melihat film *horror*

|index|Title|Genre|Language|Runtime|Premiere|IMDB Score|
|---|---|---|---|---|---|---|
|0|Apostle|Horror-thriller|English|129|12/Oct 2018|6\.3|
|1|The Perfection|Horror-thriller|English|90|24/May 2019|6\.1|
|2|Gerald's Game|Horror thriller|English|103|29/Sep 2017|6\.5|
|3|Nobody Sleeps in the Woods Tonight|Horror|Polish|103|28/Oct 2020|4\.8|
|4|Cadaver|Horror|Norwegian|86|22/Oct 2020|5\.1|
|5|Rattlesnake|Horror|English|85|25/Oct 2019|4\.6|
|6|I Am the Pretty Thing That Lives in the House|Horror|English|89|28/Oct 2016|4\.6|
|7|In the Tall Grass|Horror|English|101|04/Oct 2019|5\.4|
|8|Bulbbul|Horror|Hindi|94|24/Jun 2020|6\.6|
|9|Things Heard & Seen|Horror|English|121|29/Apr 2021|5\.3|

#References
[<sup>[1]</sup>](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9269752/) Paweł Pławiak, Academic Editor - Movie Recommender Systems: Concepts, Methods, Challenges, and Future Directions</br>
[<sup>[2]</sup>](https://medium.com/@m_n_malaeb/recall-and-precision-at-k-for-recommender-systems-618483226c54) Maher Malaeb - Recall and Precision at k for Recommender Systems</br>
[<sup>[3]</sup>](https://datascience.stackexchange.com/questions/92247/precisionk-and-recallk) drorhun - precision@k and recall@k</br>
[<sup>[4]</sup>](https://www.shaped.ai/blog/evaluating-recommendation-systems-part-1) Tullie Murrell \[February 7, 2023\] - Evaluating Recommendation Systems - Precision@k, Recall@k, and R-Precision</br>
[<sup>[5]</sup>](https://www.dicoding.com/academies/319/discussions/134402) Kurnia Sari Sitanggang \[12 Oct 2021\] - Penerapan metric precision pada evaluasi Content Based Filtering ?

[<sup>[6]</sup>](https://www.kaggle.com/datasets/luiscorter/netflix-original-films-imdb-scores) LUIS - Netflix Original Films & IMDB Scores</br>
[<sup>[7]</sup>](https://en.wikipedia.org/wiki/Lists_of_Netflix_original_films)Lists of Netflix original films