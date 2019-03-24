Uczenie maszynowe - notatniki
==========================

Niniejszy projekt ma na celu zapoznanie Cię z podstawami uczenia maszynowego w środowisku Python. Notatniki te stanowią materiały dodatkowe i rozwiązania ćwiczeń zamieszczonych w książce [Hands-on Machine Learning with Scikit-Learn and TensorFlow](http://shop.oreilly.com/product/0636920052289.do).


# Instalacja

Jeżeli chcesz poeksperymentować z kodem uczenia przez wzmacnianie opisanym w rozdziale 16., musisz zainstalować [narzędzia OpenAI gym](https://gym.openai.com/docs) i ich zależności pozwalające na symulowanie środowiska Atari.

Jeśli znasz środowisko Python i potrafisz je zainstalować, przejdź do pliku `wymagania.txt`, zainstaluj wymienione tam biblioteki i przejdź do poniższej sekcji [Uruchamianie notatników Jupyter](#Uruchamianie-notatników-Jupyter). W przeciwnym wypadku przeczytaj niniejszy plik do końca.

## Środowisko Python i wymagane biblioteki
Jest oczywiste, że będziemy potrzebować środowiska Python. Obecnie w większościach systemów jest już domyślnie zainstalowane środowisko Python 2, a w niektórych przypadkach również Python 3. Możesz sprawdzić wersję występującą u Ciebie za pomocą następujących poleceń:

    $ python --version   # for Python 2
    $ python3 --version  # for Python 3

Każda wersja środowiska Python 3., zwłaszcza ≥3.5, powinna być dobra. W przypadku jej braku zalecam jej zainstalowanie (środowisko Python ≥2.6 również powinno działać, zawsze jednak lepiej korzystać z najbardziej aktualnej wersji). Możesz tego dokonać na kilka sposobów: w systemach Windows lub MacOSX wystarczy pobrać je ze strony [python.org](https://www.python.org/downloads/). W systemie MacOSX możesz ewentualnie użyć menedżerów [MacPorts](https://www.macports.org/) lub [Homebrew](https://brew.sh/). W przypadku stosowania środowiska Python 3.6 na systemie MacOSX musisz wpisać poniższą komendę, aby zainstalować pakiet certyfikatów `certifi`, ponieważ ta wersja nie zawiera certyfikatów uwierzytelniających połączenia SSL (patrz [pytanie w serwisie StackOverflow](https://stackoverflow.com/questions/27835619/urllib-and-ssl-certificate-verify-failed-error)):

    $ /Applications/Python\ 3.6/Install\ Certificates.command

W dystrybucji Linux należy korzystać z domyślnego menedżera pakietów, chyba że wiesz, co robisz. Na przykład w dystrybucjach Debian lub Ubuntu należy wpisać:

    $ sudo apt-get update
    $ sudo apt-get install python3

Alternatywnym rozwiązaniem jest pobranie i instalacja platformy [Anaconda](https://www.continuum.io/downloads). Jest to pakiet zawierający środowisko Python i wiele bibliotek naukowych. Zalecana jest wersja wykorzystująca środowisko Python 3.

Jeśli interesuje Cię platforma Anaconda, przeczytaj następną sekcję, w przeciwnym wypadku przejdź do sekcji [Korzystanie z menedżera pip](#Korzystanie-z-menedżera-pip) section.

## Using Anaconda
Podczas stosowania platformy Anaconda możemy ewentualnie stworzyć izolowane środowisko Python przeznaczone dla określonego projektu. Jest to zalecane rozwiązanie, ponieważ możemy w ten sposób definiować osobne środowisko dla każdego projektu (np. dla tych notatników), w których mogą być używane odmienne biblioteki lub różne wersje tej samej biblioteki:

    $ conda create -n umksiążka python=3.5 anaconda
    $ source activate umksiążka

Powyższe polecenia tworzą i uaktywniają świeże środowisko Python 3.5 nazwane `umksiążka` (możesz dowolnie zmienić nazwę). Środowisko to zawiera wszystkie biblioteki naukowe stanowiące część platformy Anaconda, w tym wszystkie potrzebne nam moduły (NumPy, Matplotlib, Pandas, Jupyter i kilka innych), oprócz biblioteki TensorFlow, dlatego zainstalujmy ją już teraz:

    $ conda install -n umksiążka -c conda-forge tensorflow=1.4.0

W ten sposób zainstalowaliśmy moduł TensorFlow 1.4.0 w środowisku `umksiążka` (pobraliśmy je z repozytorium `conda-forge`). Jeśli nie zamierzasz korzystać ze środowiska `umksiążka`, pomiń opcję `-n umksiążka`.

Teraz możemy ewentualnie zainstalować rozszerzenia aplikacji. Pozwalają one korzystać z ładnych spisów treści w notatnikach, nie jest to jednak niezbędne.

    $ conda install -n umksiążka -c conda-forge jupyter_contrib_nbextensions

Jesteśmy gotowi! Przejdź teraz do sekcji [Uruchamianie notatników Jupyter](#Uruchamianie-notatników-Jupyter).

## Korzystanie z menedżera pip 
Jeżeli nie korzystasz z platformy Anaconda, musisz zainstalować kilka bibliotek naukowych środowiska Python, bez których nasz projekt nie będzie działać - dotyczy to w szczególności modułów NumPy, Matplotlib, Pandas, Jupyter i TensorFlow (a także kilku innych). W tym celu możesz użyć zintegrowanego ze środowiskiem Python menedżera pakietów pip lub menedżera stanowiącego część danego systemu (jeśli jest dostępny, np. w dystrybucjach Linux lub w systemie MacOS X, jeśli korzystasz z menedżerów MacPorts lub Homebrew). Zaletą używania menedżera pip jest łatwość tworzenia wielu izolowanych środowisk Python korzystających z różnych bibliotek i wersji danej biblioteki (tj. po jednym  środowisku na każdy projekt). Z kolei dzięki korzystaniu z wbudowanego menedżera pakietów istnieje mniejsze ryzyko występowania konfliktów pomiędzy bibliotekami środowiska Python, a innymi pakietami. Osobiście mam wiele projektów mających odmienne wymagania dotyczące bibliotek, dlatego wolę korzystać z menedżera pip i definiować środowiska izolowane.

Poniżej wymieniam polecenia, jakie należy wpisać w terminalu, jeśli chcesz za pomocą menedżera pip zainstalować wymagane biblioteki. Uwaga: jeżeli korzystasz ze środowiska Python 2, należy zastąpić wyrażenia `pip3` i `python3` wyrażeniami, odpowiednio: `pip` oraz `python`.

Upewnijmy się najpierw, że mamy zainstalowaną najnowszą wersję menedżera pip:

    $ pip3 install --user --upgrade pip

Opcja `--user` pozwala zainstalować najnowszą wersję menedżera pip jedynie dla bieżącego użytkownika. Jeżeli chcesz tego dokonać dla wszystkich użytkowników, musisz mieć prawa administratora (np. skorzystać z polecenia  `sudo pip3` w dystrybucji Linux), i usunąć opcję `--user`. To samo dotyczy każdej poniższej komendy zawierającej opcję `--user` option.

Następnie możesz (ale nie musisz) stworzyć środowisko izolowane. Jest to zalecane rozwiązanie, ponieważ dzięki temu możesz definiować oddzielne środowisko dla każdego projektu (np. dla bieżącego), mogące diametralnie różnić się rodzajami i wersjami bibliotek:

    $ pip3 install --user --upgrade virtualenv
    $ virtualenv -p `which python3` izo

Zostaje wygenerowany nowy katalog `izo` w bieżącym folderze, przechowujący izolowane środowisko bazujące na wersji Python 3. Jeśli masz zainstalowanych kilka różnych wersji środowiska Python 3, możesz zastąpić wyrażenie `` `which python3` `` ścieżką do preferowanego pliku wykonywalnego.

Teraz musimy uaktywić to środowisko. Musimy wpisywać poniższe polecenie za każdym razem, gdy chcemy korzystać z tego środowiska.

    $ source ./izo/bin/activate

Teraz zainstalujemy wymagane pakiety środowiska Python za pomocą menedżera pip. Jeżeli nie korzystasz ze środowiska virtualenv, dodaj opcję  `--user` (ewentualnie możesz zainstalować te biblioteki dla wszystkich użytkowników, prawdopodobnie jednak wymagane będą do tego uprawnienia administratora, np. korzystanie z polecenia `sudo pip3` w dystrybucji Linux).

    $ pip3 install --upgrade -r requirements.txt

Świetnie! Wszystko już jest gotowe, czas uruchomić aplikację Jupyter.

## Uruchamianie notatników Jupyter
Jeśli chcesz korzystać z rozszerzeń aplikacji Jupyter (nie jest to niezbędne, ale dzięki nim otrzymujemy ładne spisy treści), musisz je najpierw zainstalować:

    $ jupyter contrib nbextension install --user

Teraz możemy uaktywnić rozszerzenie, takie jak, na przykład, Table of Contents (2):

    $ jupyter nbextension enable toc2/main

W porządku! Aby uruchomić aplikację Jupyter, wystarczy wpisać:

    $ jupyter notebook

W Twojej przeglądrce powinno otworzyć się nowe okno zawierające listę elementów katalogu roboczego. Jeżeli przeglądarka nie uruchomi się automatycznie, wpisz adres [localhost:8888](http://localhost:8888/tree). Aby rozpocząć przeglądanie notatników, kliknij plik `indeks.ipynb`!

Uwaga: możesz również uaktywniać i konfigurować rozszerzenia pod adresem [http://localhost:8888/nbextensions](http://localhost:8888/nbextensions).

Gratulacje! Jesteś gotowa/gotów chłonąć wiedzę z zakresu uczenia maszynowego!

# Ofiarodawcy
Chciałbym podziękować wszystkich osobom przyczyniającym się do powstania tego projektu, czy to poprzez sugestie, wykrywanie błędów albo dzielenie się wątpliwościami. Specjalne podziękowania kieruję do użytkowników Steven Bunkley i Ziembla za stworzenie katalogu `docker`.
