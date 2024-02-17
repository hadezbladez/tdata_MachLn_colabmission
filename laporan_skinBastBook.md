# Laporan Proyek Machine Learning - Handerson Loriano
## Domain proyek
Pendahuluan yang mendasarkan dari kutipan [<sup>[1]</sup>](https://www.kaggle.com/datasets/farjanakabirsamanta/skin-cancer-dataset) [<sup>[2]</sup>](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) [<sup>[3]</sup>](https://www.nature.com/articles/sdata2018161#Sec1)</br>
</br>
*Dermatoscopic* banyak digunakan oleh pengguna tanpa memakai alat bantuan lihat lain karena gambar dari *dermatoscopic* sesuai untuk pelatihan *AI* untuk mendiagnosa keadaan kulit. Karena *Training dataset* untuk *machine learning* karena beberapa ukuran gambar data sampel kecil dan cenderung tidak terlalu beragam dari gambar *dermatoscopic*.</br>
 
Dikutip dari [Wikipedia](https://en.wikipedia.org/wiki/Dermatoscopy) </br>
*Dermatoscopic* adalah pemeriksaan lesi kulit memakai *dermatoscope*. Alat ini sangat mirip dengan kamera untuk memeriksa lesi kulit tanpa ada halangi disekitarnya. *Dermatoscope* ini terdiri dari :
1. lensa kaca pembesar
2. cahaya polarisasi dan biasa
3. plat transparansi dan beberap gel menjadi medium diantara alatnya dan kulit pasien
 
### Rumusan permasalahan awal
- permasalahan karena alat dermatoscope adalah alat yang mempunyai biaya yang cukup besar sehingga ini menjadi pertimbangan untuk para ahli menggunakan alatnya. </br>
- Tidak mudah bagi manusia yang ahli mendiagnosa kulit dari kasat mata karena, alat *dermatoscopy* membutuhkan penilaian dan pertimbangan dari mata manusia sehingga ini menjadi hal yang perlu diselesaikan. Untungnya beberapa gambar dari alat *dermatoscopy* sangat sesuai untuk *machine learning*
- karena manusia lebih mudah melihat dari angka dan data visualisasi yang tepat dibanding gambaran maka, alat ini adalah sebagai alat penunjang untuk memastikan kondisi kulit tersebut. </br>
- Sampel data yang tidak terlalu banyak dapat menyebabkan hasil analisa *AI* yang tidak sesuai karenanya, dimunculkan data-data pasien dari populasi yang berbeda-beda
 
Seiring perkembangan jaman proses *machine learning* dan alat seperti *graphic card* semakin canggih. Maka hal ini bisa menyelesaikan data2 yang tadinya sangat banyak menjadi lebih ringkas dan bisa divisualisasikan oleh manusia supaya mendapat laporan lesi kulit yang lebih baik.</br>
Dengan adanya hal tersebut :
1. Mempermudah dan dapat berpotensi mempersingkat waktu dalam analisa penyakit lesi kulit pengecekan analisis pada penyakit lesi kulit
2. Pengecekan yang memakan waktu sedikit akan berpotensi mengurangi biaya yang besar seperti alatnya *dermatoscope*</br>
 
### *Background*
Sangat tidak mudah untuk mendeteksi dini tentang kesehatan kulit :
1. Dari biaya yang lumayan mahal hanya untuk mendeteksi beberapa jenis kulit
2. Dari sekian kulit memang belum tentu membahayakan seseorang ini tidak sebanding dengan biaya mahal hanya untuk mendeteksi dan bahayanya lesi kulit yang lebih ganas. Maka dari hal itu diperlukan ada alat deteksi otomatis
3. Ada artikel tentang penyakit kulit ganas dari [ini](https://www.foxnews.com/health/olympic-swimmer-potentially-saved-from-skin-cancer-thanks-to-eagle-eyed-fan). </br>
([Catatan](https://www.worldlifeexpectancy.com/id/indonesia-skin-disease) Karena, sumber data dari WHO tidak bisa dilihat secara langsung, maka pengambilan data mentah berasal dari sumber website yang bukan dari sumber data WHO)</br>
walaupun tidak mematikan tetapi, ini bisa menjadikan perhatian bahwa penyakit lesi kulit itu bukan hal yang remeh-temeh.
 
## Business Understanding
### Problem Statements
1. Apakah usia mempengaruhi seseorang terkena penyakit kulit?
2. Bagaimana cara membedakan penyakit kulit dengan beberapa fitur yang ada?
 
### Goals
1. Menganalisa seseorang terkena penyakit kulit berdasarkan usia dan mengetahui korelasinya </br>
2. Membuat model machine learning yang dapat memprediksi seseorang jenis penyakit kulit apa bila seseorang mempunyai fitur tertentu</br>
 
## Data Understanding
Ini adalah tahapan yang dilakukan pada *Data Understanding* :
1. Membaca sumber dari yang sudah ada sehingga bisa mengikuti arti data yang baik
2. Mendeskripsikan data supaya bisa melihat pola apa saja yang bisa disiapkan pada tahapan Data Preparation
3. Menyaring missing Value dan Outliers
4. Tidak ada korelasi dengan kolom lain karena itu usia menjadi patokan data yang bernilai
5. Melakukan EDA Univariate analysis supaya dapat melihat total data yang dapat lihat
6. Melakukan EDA Multivariate analysis untuk Categorical features karena, numerical features hanya usia(age) saja
 
Info Dataset :
1. Nama Dataset : HAM10000 dataset </br>
2. total : 10.015</br>
 
Berikut ini adalah maksud / arti dari nama kolom tersebut :
1. lesion_id : nilai unik untuk data
2. image_id : nama file image (Tidak akan digunakan)
3. dx : nama penyakit kulit kanker </br>
  - akiec = Actinic keratoses and intraepithelial carcinoma / Bowen's disease
  - bcc = basal cell carcinoma
  - nkl = benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses)
  - df = dermatofibroma
  - mel = melanoma
  - nv = melanocytic nevi
  - vasc = vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage)
4. dx_type : data pertimbangan penyakit kulit secara detail
5. age : umur
6. sex : jenis kelamin
7. localization : lokasi bagian kulit tubuh
</br>
 
proses yang akan dilakukan pada tahapan ini adalah :
1. Data loading
2. Exploratory Data Analysis
 
### Data Loading
Data yang berasal dari sumber [ini](https://www.kaggle.com/datasets/farjanakabirsamanta/skin-cancer-dataset) :
1. setelah diunduh, harus diunggah ke *google drive*
2. dan mengubah akses supaya bisa pakai di *google colab*</br>
 
Pada tahap data loading akan melakukan pengambil dataan dan menyeleksi kolom yang perlu diperhatikan </br>
(Catatan! karena beberapa dari data tersebut memerlukan data dari gambar maka datai tersebut tidak dipakai dan hanya data yang berada ditabel yang dipakai)</br>
 
### Exploratory Data Analysis
Ini adalah tahapan yang dilakukan *Exploratory Data Analysis* :
1. Mendeskripsikan variabel kolom
2. menganalisis untuk mencari missing value dan outliers
3. Menggunakan cara univariate analysis dan Multivariate analysis untuk analisis data</br>
 
Exploratory data analysis dapat membantu mencari outliers dan menjawab pertanyaan pengertian bisnis yang disampaikan.
Seperti pada gambar dibawah ini:</br>
![image](https://drive.google.com/uc?id=1rFtI_CCwAERk7XYUEfIo1PnUzLL-9mBG&export=download)</br>
Tidak bisa dipungkiri lagi sex dengan variable *unknown* adalah data kategorikal yang bersifat *outlier* karena data yang kebetulan tidak ada ( atau Kesetaraan gender LGBT masih belum mempunyai gender bersifat unknown) maka data harus di*cleaning* data lebih lanjut lagi.</br>
 
Ada beberapa data tahapan seperti pada kolom variabel *Localization* data tersebut mengandung *Unknown* dan ini adalah hal yang tidak diperlukan maka harus melakukan data *cleaning* lagi
 
 
Beberapa data yang dianalisa dapat memberikan jawaban :
*   ternyata usia mempengaruhi seseorang terkena penyakit kulit ![image](https://drive.google.com/uc?id=1051RBRzr4-qj1oX5xG65BADcDAydDhL0&export=download)
*   Laki-laki lebih rentan terkena penyakit kulit![image](https://drive.google.com/uc?id=14eN52k_8orUrgrbH_FkUkVnSLf7rW8nB&export=download)</br>
 
## Data Preparation
Ini adalah tahapan yang dilakukan pada *Data Preparation* :
1. Encoding fitur kategori *OHE* (*One-Hot Encoding*) </br>
karena, Data kategorikal yang sebenarnya tidak
Beberapa dari data ini akan di-*Encode* dengan cara *OHE* untuk kolom : *sex* dan *localization*</br>
2. pembagian dataset dengan fungsi *train_test_split*</br>
Data *training* dan data *test* akan dibagi menjadi +/- 60 : 30 karena, mendapatkan hasil yang lumayan stabil dalam evaluasi model
Dalam pembagian dataset ini terbagi menjadi </br>
X adalah data yang bisa diprediksi berdasarkan *age, sex & localization* </br>
y adalah data target yang untuk diprediksi (dx)</br>
 
## Modelling
Penggunaan model
1. *Linear SVC* (lsvc : clf)
2. *K Neighbors Classifier* (knc : knn)
3. *Random Forest Classifier* (randfc : clfr)
4. *Boosting Algorithm* (gradboostc : clfg)</br>
 
Karena ada 4 model tahapan 1 adalah tahapan paling buruk dan terakhir adalah tahapan terbaik tetapi, pada kenyataannya tidak selalu begitu. Random forest terpilih yang baik karena akurasinya. walaupun linear SVC adalah terburuk tetapi, akurasi tersebut bisa menyaingi beberapa model setingkatnya.
 
Model-model tesebut memiliki data default dan masih belum *tuning* hyperparameter, karena masih tahapan belajar dan model-model tersebut memiliki akurasi yang bagus.
 
Terlihat model linearSVC hampir sesuai dengan akurasinya dan model yang paling mendekati akurasinya.
 
Sewaktu pertama kali melihat tentang prediksi akurasi pertama kali, praduga awal yang terbaik adalah model random forest classifier karena, memang hasil akurasinya tertinggi. Setelah di-evaluasi pada beberapa sampel bisa diketahui bahwa Gradient boosting classifier adalah hasil yang terbaik walaupun dia bukan mendapat akurasi yang terbaik tetapi hasil koreksi sangat mendekati dengan data yang diuji!
 
## Evaluation
Model metrik dievaluasi berdasarkan *accuracy-score*</br>
Berdasarkan dari [Scikit-Page tentang *accuracy-score*](https://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score)</br></br>
 
If $\hat{y}_i$ is the predicted value of the $i$-th sample and $y_i$ is the corresponding true value, then the fraction of correct predictions over $n_\text{samples}$ is defined as</br>
$\texttt{accuracy}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples}-1} 1(\hat{y}_i = y_i)$</br></br>
 
Fungsi *Accuracy-score* menghitung akurasi, dari beberapa deret atau menghitung nilai prediksi yang benar.</br>
 
Pada pengkelasan *multilabel*, fungsi tersebut mengembalikan nilai dari sebagian deret nilai akurasi. Bila semua data diprediksi pada label untuk sampel yang cocok dengan deret nilai, maka akurasi tersebut bernilai 1; yang lain adalah 0</br>
 
![image](https://drive.google.com/uc?id=1oq-2nH9TIIYIxitFGSKXAaT7xeNoHQWy&export=download)</br></br>
Terlihat model *Gradient Boost Classifier* adalah model yang mempunyai *accuracy-score* yang bagus. </br>
Ternyata sewaktu dievaluasi Algoritma *Gradient Boost Classifier* adalah hasil yang paling mempunyai penilaian akurasi besar walaupun, algoritma *Random Forest Classifier* adalah akurasi model yang terbesar.
 
# Referensi
https://www.kaggle.com/datasets/farjanakabirsamanta/skin-cancer-dataset, </br>
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T</br>
https://www.nature.com/articles/sdata2018161#Sec1</br>
https://en.wikipedia.org/wiki/Dermatoscopy</br>
https://www.foxnews.com/health/olympic-swimmer-potentially-saved-from-skin-cancer-thanks-to-eagle-eyed-fan</br>
https://www.worldlifeexpectancy.com/id/indonesia-skin-disease</br>
https://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score</br>