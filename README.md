# EFT Company Matcher

**EFT açıklamalarından şirket eşleştirme · Hibrit aday arama · Yerel model çıkarımı**

EFT Company Matcher, ödeme açıklamalarında eksik, hatalı veya kısaltılmış biçimde yazılan şirket adlarını, referans şirket ve alternatif ad (alias) kayıtlarıyla eşleştirmek için geliştirilen bir Python projesidir. Metin normalizasyonu, karakter tabanlı benzerlik ve anlamsal aramayı bir araya getirir; bulunan adayları yeniden değerlendirerek eşleşme kararını ve alternatifleri kullanıcıya sunar.

Sistem, yeterli eşleşme kanıtı bulunmadığında `UNKNOWN` sonucu üretecek şekilde tasarlanmıştır. Kullanıcı arayüzü Streamlit, servis katmanı FastAPI, kalıcı veri katmanı PostgreSQL ile geliştirilmiştir.

## Problem ve yaklaşım

Aynı şirket, farklı ödeme açıklamalarında kısaltılmış unvanıyla, yazım hatalarıyla veya işlem numaralarıyla birlikte yer alabilir. Benzer unvanlara sahip farklı şirketler de adayların ayrıştırılmasını zorlaştırır.

Projede iki aşamalı bir yaklaşım kullanılır:

1. **Aday üretimi:** Şirket ve alias havuzundan metinsel ve anlamsal benzerlik sinyalleriyle güçlü adaylar seçilir.
2. **Yeniden değerlendirme:** Seçilen adaylar ayrıntılı benzerlik özellikleri, yeniden sıralama ve karar eşikleriyle değerlendirilir.

Örneğin, referans havuzuna eklenmiş temsili bir şirket için aşağıdaki gibi bir eşleştirme hedeflenir:

| Alan | Temsili değer |
| --- | --- |
| Referans şirket | ÖRNEK ANKA YAZILIM VE TEKNOLOJİ A.Ş. |
| EFT açıklaması | ORNEK ANKA YAZLM TEKNOLOJI FATURA ODEME 4821 |
| Amaç | Yazım farklılıkları ve işlem ifadeleri içinden doğru şirket adayını bulmak |

Bu örnek, kullanım senaryosunu anlatır; ölçülmüş bir test sonucu değildir.

## Temel özellikler

- EFT açıklamalarının ve şirket ifadelerinin normalizasyonu.
- Şirket unvanları ile alternatif yazımların birlikte değerlendirilmesi.
- Qwen3 Embedding 0.6B ile anlamsal aday arama.
- Karakter tabanlı TF-IDF, prefix uyumu ve Jaro-Winkler gibi metinsel benzerlik sinyalleri.
- En güçlü adayın, alternatiflerin, skorların ve karar türünün gösterilmesi.
- Otomatik eşleştirme, inceleme gerektiren sonuçlar ve `UNKNOWN` karar yaklaşımı.
- Backend, veritabanı ve model durumuna yönelik sağlık kontrolü bileşenleri.
- Pozitif, negatif ve şirket bazlı holdout değerlendirme betikleri.
- CatBoostRanker için deneysel aday özellikleri, eğitim ve değerlendirme bileşenleri.

CatBoost çalışmaları deneysel geliştirme kapsamındadır. Eğitim betiklerinin bulunması, eğitilmiş bir modelin depoda dağıtıldığı veya bütün sorgularda etkin olduğu anlamına gelmez.

## Mimari

Streamlit arayüzü, EFT açıklamasını HTTP üzerinden FastAPI servisine iletir. Backend, eşleştirme bileşenlerini çalıştırır ve sonuçları yapılandırılmış bir yanıt olarak döndürür. PostgreSQL, şirket ve alias kayıtlarını saklar.

Aşağıdaki şema eşleştirme hattının kavramsal akışını gösterir:

```mermaid
flowchart TD
    input["EFT açıklaması"] --> norm["Metin normalizasyonu"]
    norm --> lexical["Metinsel aday üretimi"]
    norm --> semantic["Anlamsal aday üretimi"]
    lexical --> fusion["Aday birleştirme"]
    semantic --> fusion
    fusion --> ranking["Yeniden sıralama"]
    ranking --> decision{"Karar eşikleri"}
    decision --> matched["Otomatik eşleştirme"]
    decision --> review["İnceleme"]
    decision --> unknown["UNKNOWN"]
```

Model çıkarımı yerel ortamda çalışacak şekilde geliştirilmiştir. İlk kurulumda model dosyalarının indirilmesi ve şirket verilerinin hazırlanması gerekir.

## Teknolojiler

| Teknoloji | Projedeki görevi |
| --- | --- |
| Python | Uygulama, veri hazırlama ve değerlendirme betikleri |
| FastAPI / Uvicorn | API ve geliştirme sunucusu |
| Streamlit / Requests | Kullanıcı arayüzü ve backend iletişimi |
| PostgreSQL / SQLAlchemy / Psycopg | Şirket ve alias verilerinin saklanması ve erişimi |
| Pydantic / Pydantic Settings / python-dotenv | Veri doğrulama ve uygulama ayarları |
| Qwen3 Embedding 0.6B / Sentence Transformers | Metinlerin anlamsal vektörlere dönüştürülmesi |
| Transformers / PyTorch | Yerel model yükleme ve çıkarım |
| scikit-learn / NumPy / RapidFuzz | Metinsel özellikler ve sayısal işlemler |
| CatBoostRanker | Deneysel öğrenilen sıralama yaklaşımı |
| pytest | Test altyapısı |

Embedding modelinin kullanım bilgileri için [Qwen3 Embedding 0.6B model kartına](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) bakılabilir.

## Dosya yapısı

| Konum | İçerik |
| --- | --- |
| `backend/app/api/` | API bileşenleri |
| `backend/app/database/` | Veritabanı bağlantısı, modeller ve veri erişimi |
| `backend/app/normalization/` | Metin normalizasyonu |
| `backend/app/retrieval/` | Aday üretimi ve birleştirme |
| `backend/app/ranking/` | Aday yeniden sıralama |
| `backend/app/decision/` | Karar, eşik ve kalibrasyon için ayrılmış boş taslak dosyalar |
| `backend/app/models/` | Model yükleme ve kullanım bileşenleri |
| `backend/scripts/` | Veri hazırlama, indeks, model ve benchmark betikleri |
| `backend/tests/` | Mevcut test dosyaları |
| `frontend/` | Streamlit uygulaması, sayfalar ve API istemcisi |
| `docker/` | Backend ve frontend için ayrılmış boş Dockerfile taslakları |
| `data/` | Yerel veri dosyaları için ayrılan dizinler |
| `models/` | Yerel model dosyaları için ayrılan dizin |
| `.env.example` | Örnek ortam ayarları |
| `requirements.txt` | Ana Python bağımlılık listesi |

## Geliştirme ortamını hazırlama

Aşağıdaki komutlar Windows PowerShell ve yeni bir geliştirme ortamı içindir. Komutları proje kökünde çalıştırın. Uygulamayı kullanmak için Python bağımlılıklarının yanı sıra PostgreSQL, şirket verileri ve gerekli model/indeks dosyaları da hazırlanmalıdır.

### 1. Sanal ortam

Seçilen PyTorch ve diğer paketlerle uyumlu bir Python sürümü kullanın. Yeni sanal ortam oluşturup etkinleştirin:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. PyTorch ve bağımlılıklar

GPU ile çalıştırmak için işletim sisteminize ve NVIDIA sürücünüze uygun PyTorch kurulumunu [resmî PyTorch kurulum seçicisinden](https://pytorch.org/get-started/locally/) belirleyin ve komutu etkin sanal ortamda çalıştırın.

Projenin ana bağımlılıklarını kökteki dosyadan kurun:

```powershell
python -m pip install -r requirements.txt
```

`backend/requirements.txt` ve `frontend/requirements.txt` mevcut durumda boştur; bağımlılık kaynağı kökteki dosyadır.

CatBoost bileşenleri için gereken ek paket, mevcut `requirements.txt` içinde listelenmemiştir. Bu bileşenleri kullanacak ortamda [CatBoost kurulum yönergesine](https://catboost.ai/docs/en/installation/python-installation-method-pip-install) uygun olarak kurun:

```powershell
python -m pip install catboost
```

Paket bağımlılıklarını ve PyTorch GPU görünürlüğünü kontrol edin:

```powershell
python -m pip check
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

GPU ile çalıştırılacak ortamda `CUDA: True` beklenir. Bu kontrol, uygulamanın bütün bileşenlerinin çalıştığını tek başına doğrulamaz.

### 3. Ortam ayarları

Yerel `.env` dosyası henüz yoksa örnek dosyadan oluşturun:

```powershell
if (-not (Test-Path -LiteralPath .env)) {
    Copy-Item -LiteralPath .env.example -Destination .env
}
```

Örnek yapılandırma:

```dotenv
APP_NAME=EFT Company Matcher
APP_DATA_SOURCE=postgres
ALLOW_CSV_FALLBACK=true
DATABASE_URL=postgresql+psycopg://YOUR_DB_USER:YOUR_DB_PASSWORD@127.0.0.1:5432/eft_matcher
```

| Değişken | Açıklama |
| --- | --- |
| `APP_NAME` | Uygulama adı |
| `APP_DATA_SOURCE` | Veri kaynağı seçimi; örnek ayar `postgres` |
| `ALLOW_CSV_FALLBACK` | CSV alternatifine geçişle ilgili ayar; kullanılabilmesi için uygun CSV verisinin hazır olması gerekir |
| `DATABASE_URL` | PostgreSQL bağlantı adresi; örnekte Psycopg sürücüsü kullanılır |

PostgreSQL üzerinde uygulamanın kullanacağı veritabanını hazırlayın. `.env` içindeki örnek kullanıcı adı, parola, sunucu, port ve veritabanı adını kendi ortamınıza göre düzenleyin. Örnek bağlantı `eft_matcher` adlı veritabanını kullanır.

### 4. Şirket verileri, modeller ve indeksler

Şirket ve alias verileri uygulamanın referans havuzunu oluşturur. Model ağırlıkları, hazırlanmış veri setleri ve indeksler yerel olarak üretilir veya temin edilir. Git takibinde bu dosyalar yerine gerekli dizinleri koruyan `.gitkeep` dosyaları bulunur.

Hazırlıkla ilgili, içeriği bulunan betik dosyaları:

| İşlem | İlgili betik |
| --- | --- |
| Veritabanı başlangıç hazırlığı | [init_database.py](backend/scripts/init_database.py) |
| Şirket verisi aktarımı | [import_csv_to_database.py](backend/scripts/import_csv_to_database.py) |
| Embedding modelinin indirilmesi | [download_embedding_model.py](backend/scripts/download_embedding_model.py) |
| Yeniden sıralama modeli hazırlığı | [download_reranker_model.py](backend/scripts/download_reranker_model.py) |

`import_companies.py`, `generate_alias_embeddings.py` ve `build_tfidf_index.py` mevcut durumda boş taslaklardır. Bu dosyalar, çalıştırılabilir veri veya indeks hazırlama adımları olarak kullanılmamalıdır. İndeks ve alias embedding hazırlığının güncel uygulama içindeki çağrı noktaları ayrıca doğrulanmalıdır.

**Kurulum kapsamı:** Bu bölüm ilgili bileşenleri gösterir. CSV kolonları, betiklerin zorunlu argümanları ve model dosyalarının kesin konumları henüz bu README'de ayrıntılandırılmamıştır. Temiz bir ortamda uçtan uca kurulum için bu bilgiler, ilgili betikler ve `backend/app/config.py` ile birlikte tamamlanmalıdır.

`docker/backend.Dockerfile` ve `docker/frontend.Dockerfile` da boş taslaklardır; bu dosyalarla uygulama imajı oluşturma akışı henüz hazır değildir.

## Uygulamayı başlatma

Veritabanı, veri ve gerekli modeller hazır olduğunda backend ile frontend'i ayrı terminallerde çalıştırın. Her iki terminalde de proje kökünde olun ve sanal ortamı etkinleştirin.

Backend:

```powershell
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Streamlit ana uygulaması:

```powershell
python -m streamlit run frontend/Home.py --server.address 127.0.0.1 --server.port 8501
```

Bu komutlarla backend adresi [127.0.0.1:8000](http://127.0.0.1:8000), arayüz adresi [127.0.0.1:8501](http://127.0.0.1:8501) olur. Frontend API istemcisinin kullandığı backend adresi, [api_client.py](frontend/services/api_client.py) içindeki yapılandırmayla uyumlu olmalıdır.

## Kullanım

1. Streamlit arayüzünde EFT eşleştirme sayfasını açın.
2. Şirket ifadesi içeren açıklamayı girin ve eşleştirmeyi başlatın.
3. Karar türünü, en güçlü adayı ve alternatif şirketleri inceleyin.
4. Yakın skorlu adaylarda inceleme kararını; yeterli ilişki bulunmayan sorgularda `UNKNOWN` sonucunu değerlendirin.

Skorlar adayların karşılaştırılmasına yardımcı olur. Ayrıca kalibre edilmedikçe bu skorlar doğrudan doğru eşleşme olasılığı olarak yorumlanmamalıdır.

## Test ve değerlendirme

Mevcut test dosyaları `backend/tests/` altında normalizasyon, şirket indeksi, aday yeniden sıralama ve belirsiz tam eşleşme senaryolarını kapsayan dosyalardan oluşur.

Proje kökünde, `backend` dizinini yalnızca test sürecinin modül arama yoluna ekleyerek testleri çalıştırmak için:

```powershell
python -c "import sys, pytest; sys.path.insert(0, 'backend'); raise SystemExit(pytest.main(['backend/tests', '-q']))"
```

Benchmark üretme ve çalıştırma betikleri `backend/scripts/` altında yer alır. Değerlendirme yaklaşımında aşağıdaki ölçütler izlenir:

| Ölçüt | Değerlendirdiği davranış |
| --- | --- |
| Recall@K | Beklenen şirketin ilk K aday arasında bulunması |
| Top-1 doğruluğu | Beklenen şirketin ilk sıraya yerleşmesi |
| UNKNOWN recall | Eşleşmemesi gereken sorguların doğru biçimde reddedilmesi |
| Yanlış otomatik eşleşmeler | Hatalı bir şirket için otomatik eşleşme kararı verilmesi |

Şirket bazlı holdout yaklaşımında, değerlendirme sorguları için eğitimde görülmeyen şirketler ayrılır. Sonuçlar yorumlanırken örnek sayısı, pozitif/negatif dağılımı, aday limiti ve veri ayrımı birlikte ele alınmalıdır.

CatBoost geliştirmesi için aday özellik dışa aktarma, eğitim, shadow benchmark ve holdout üretimi betikleri bulunur. Yeniden üretilebilir değerlendirme için veri hazırlama parametreleri ile kullanılan model ve karar ayarlarının birlikte kaydedilmesi gerekir.

## Geliştirici

Berk Ilgın Öner
