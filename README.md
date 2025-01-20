# Opis Problemu

W dzisiejszym świecie popularność artykułów online ma kluczowe znaczenie dla wydawców treści, platform medialnych, a nawet biznesów. Przewidywanie liczby udostępnień artykułu pozwala na optymalizację treści, lepszą personalizację oraz większą szansę na dotarcie do wielu osób. Problem polega na zbudowaniu modelu uczenia maszynowego, który będzie w stanie przewidywać liczbę udostępnień artykułu na podstawie szeregu zmiennych, które można wziąć pod uwagę przy pisaniu i udostępnianiu treści przez twórców internetowych.

# Założenia i Cele Projektu

Celem projektu jest stworzenie aplikacji przewidującej liczbę udostępnień artykułu w internecie, wykorzystującej algorytmy uczenia maszynowego.
Założenia projektu:
 - Wykorzystanie zbioru danych Online News Popularity z repozytorium UCI.
 - Implementacja i porównanie modeli regresyjnych takich jak regresja liniowa, drzewo decyzyjne, las losowy oraz gradient boosting.
 - Wykorzystanie narzędzi takich jak Streamlit, Kedro oraz Weights and Biases.
 - Wdrożenie aplikacji przewidującej popularność artykułów.

# Rezultaty

Projekt dostarczy aplikację webową umożliwiającą przewidywanie liczby udostępnień artykułów na podstawie podanych zmiennych. Opracowane modele regresyjne zostaną ocenione i zoptymalizowane, wskazując na najbardziej efektywne podejście. Wnioski uzyskane z analizy danych zostaną wykorzystane do udzielenia rekomendacji zwiększających popularność treści.
Wykonane elementy:
 - Wykorzystanie zbioru danych Online News Popularity z repozytorium UCI.
 - Implementacja i porównanie modeli takich jak regresja liniowa, drzewo decyzyjne, las losowy oraz gradient boosting.
 - Modularizacja kodu zgodnie z wytycznymi MlOps Level 0.
 - Implementacja potoku ML w Kedro
 - Logowanie wyników eksperymentów w Weights & Biases.
 - Wykorzystanie AutoGluon do wyboru najlepszych algorytmów i dostrajania hiperparametrów
 - Stworzenie oraz zintegrowanie lokalnej bazy danych MySQL do przechowywania wyników predykcji i danych wejściowych.
 - Wystawienie endpointu FastAPI do predykcji w czasie rzeczywistym..
 - Wdrożenie aplikacji Streamlit umożliwiającej załadowanie danych, uruchomienie pipeline'u i wizualizację wyników.

# Instalacja

1. stworzenie i uruchominie virtualnego środowiska
uruchom polecenie w katalogu głównym projektu
```
python -m venv .venv
pip install -r requirements.txt
.\.venv\Scripts\activate
```
jeśli Conda
```
conda env create --file news-online-popularity\src\news_online_popularity\environment.yaml
conda activate news-popularity
```
2. uruchom streamlit
```
streamlit run news-online-popularity\interface\Predictions.py
```
3. w oddzielnym terminalu uruchom fastapi
```
fastapi dev .\news-online-popularity\api\FastApi.py
```
4. Wewnątrz okna przeglądarki otwórz "Settings"
6. Wypełnij pola: "W and B project name", "W and B api key" i "Connection string"
7. Naciśnij "Create database"

# Integracja Bazy Danych

Dane przechowywane są w bazie danych PostgreSQL, do której aplikacja łączy się za pomocą biblioteki SQLAlchemy. Struktura bazy danych obejmuje tabele dla surowych danych
Dane przechowywane są w bazie danych MySQL, do której aplikacja łączy się za pomocą biblioteki SQLAlchemy. Struktura bazy danych obejmuje tabele dla surowych danych. Połączenie w pliku catalog.yml
```
news_data_table:
    type: pandas.SQLTableDataset
    credentials: my_mysql_creds
    table_name: newspop
```
Dane są później przekazywane do potoku w pliku pipeline.py
```
    node(
    func=prepare_data,
    inputs="news_data_table",
    outputs=["X","y"],
    name="data_praparation_node",
),
```
# Streamlit

Aplikacja składa się z trzech widoków. W pierwszym  możliwa jest predykcja danych w podanym pliku .csv, lub dodanych ręcznie. Po predykcji danych ręcznie dodanych możliwe jest ich dodanie wraz z wynikiem. Jeśli jakieś pole nie zostanie wypełnione, jego wartość traktowana jest jako 0.
Widok kolejny to “Database view” pozwalający na eksplorację oraz modyfikację danych w bazie.
Ostatnim widokiem jest widok “Settings” w którym użytkownik może manipulować parametrami modeli, dodawać dane wymagane przez bazę danych i “WadnB” oraz uruchamiać potok Kedro.

# Kedro

Kedro jest narzędziem ułatwiającym tworzenie modeli uczenia maszynowego. Na użytek w tej aplikacji stworzony został pipeline “ASI”. Pipeline ten pozwala na wytrenowanie dwóch modeli uczenia maszynowego: “Drzewa decyzji” i “regresji liniowej. Start potoku następuje po wciśnięciu przez urzytkownika przycisku “Run kedro” w widoku “Settings”. Po jego zakończeniu na górze widoku pojawią się nowo wytrenowane modele.

# Struktura Potoku
Potok podzielony jest na node-y. Każdy node odpowiada za operacje na danych, tworzenie modelu, lub wyliczanie metryk. 
```
node(
    func=prepare_data,
    inputs="news_data_table",
    outputs=["X","y"],
    name="data_praparation_node",
),
node(
    func=split_data,
    inputs=["X","y","params:test_size","params:random_state"],
    outputs=["X_train", "X_test", "y_train", "y_test"],
    name="data_split_node",
),
node(
    func=train_linear_regression,
    inputs=["X_train","y_train", "y_test", "X_test","X","y"],
    outputs="linear_regression",
    name="linear_model_node",
),
node(
    func=get_predictions,
    inputs=["linear_regression","X_test"],
    outputs="linear_predictions",
    name="linear_predictions_node",
),
node(
    func=calculate_metrics,
    inputs=["linear_regression", "X_train", "y_train","y_test","linear_predictions"],
    outputs="linear_metrics",
    name="linear_metrics_node",
),
node(
    func=print_metrics,
    inputs=["linear_metrics","params:linear_regression_name"],
    outputs=None,
    name="linear_metrics_print_node",
),
node(
    func=train_decision_tree,
    inputs=["X_train","y_train", "y_test", "X_test","X","y"],
    outputs="decision_tree",
    name="tree_model_node",
),
node(
    func=get_predictions,
    inputs=["decision_tree","X_test"],
    outputs="tree_predictions",
    name="tree_predictions_node",
),
node(
    func=calculate_metrics,
    inputs=["decision_tree", "X_train", "y_train","y_test","tree_predictions"],
    outputs="tree_metrics",
    name="tree_metrics_node",
),
node(
    func=print_metrics,
    inputs=["tree_metrics","params:decision_tree_name"],
    outputs=None,
    name="tree_metrics_print_node",
),
```
 - data_preparation_node - pobiera dane z bazy, obrabia je i dzieli na zbiory X i y.
 - data_split_node - dzieli dane ze zbiorów X i y na zbiory testowe i treningowe
 - linear_model_node - trenuje model regresji liniowej
 - linear_predictions_node - tworzy zbiór predykcji ze zbioru testowego
 - linear_metrics_node - zwraca statystyki modelu na bazie zbiorów testowych( mse, r2, abs_error, rmse, medae, evs, scores )
 - linear_metrics_print_node - wyświetla podane metryki w konsoli
 - tree_model_node - trenuje model drzewa decyzyjnego
 - tree_predictions_node - tworzy zbiór predykcji ze zbioru testowego
 - tree_metrics_node - zwraca statystyki modelu na bazie zbiorów testowych( mse, r2, abs_error, rmse, medae, evs, scores )
 - tree_metrics_print_node - wyświetla podane metryki w konsoli

# Weights and Biases

Logowanie danych do serwisu “Weights and Biases” odbywa się z użyciem “Kedro Hooks”. W pliku hooks.py utworzona została funkcja “after_node_run”. Funkcja wołana jest za każdym razem kiedy jakiś node zakończy swoje działanie. Do funkcji przesyłane są dwasłowniki: “inputs” w którym znajdują się dany przyjęte przez dany node, oraz “outputs” z danymi które dany node zwraca. Przy użyciu instrukcji logicznych możemy sprawić by dana część funkcji wykonywana była tylko gdy zakończy działanie dany node. Przykład funkcji poniżej.
```
class WandBCallHook:

    @hook_impl
    def after_node_run(self, node: Node, inputs: dict[str, Any], outputs: dict[str, Any]):

        if node.name == "data_praparation_node":

            with open('news-online-popularity\\conf\\local\\credentials.yml', 'r') as file:
                conn_str_service = yaml.safe_load(file)

            wandb.login(key=conn_str_service['wandbapikey'])
            wandb.init(project=conn_str_service['wandbprojectname'])
            df = pd.DataFrame(inputs['news_data_table'])
            df.columns = [str(col) for col in df.columns]
            wandb.run.log({"Dataset": wandb.Table(dataframe=df)})

```

# Wizualizacja Struktury Projektu
Struktura projektu
![Alt text](/screenshots/Structure.png "Potok Kedro")

Potok Kedro
![Alt text](/screenshots/kedro-pipeline.png "Potok Kedro")
